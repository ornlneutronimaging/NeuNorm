import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipp as sc
from astropy.io import fits
from PIL import Image
from scitiff.io import load_scitiff

from neunorm import __version__
from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline


class TestMarsCCDPipeline:
    """Tests for the MARS CCD pipeline."""

    @classmethod
    def setup_class(cls):
        """Create tiff files for testing once for all tests in this class."""
        cls.sample_paths = []
        cls.sample_paths_bad_pixels = []
        cls.ob_paths = []
        cls.dark_paths = []

        cls._tmpdir = tempfile.TemporaryDirectory(delete=False)
        tmp_dir = Path(cls._tmpdir.name)

        # create 5 sample tiffs with values 81-85 and metadata.
        for i in range(5):
            data = np.full((32, 32), 81 + i, dtype=np.float32)
            img = Image.fromarray(data)
            exif = img.getexif()
            exif[65027] = "ExposureTime:30.000000"
            exif[65022] = f"RunNo:{1000 + i}"
            exif[65025] = "ModelStr:DW936_BV"
            filename = tmp_dir / f"sample_{i:03}.tiff"
            img.save(filename, exif=exif)
            cls.sample_paths.append(filename)

        # create 5 sample tiffs with values 81-85 and metadata. These have a dead pixel and a gamma spike.
        for i in range(5):
            data = np.full((32, 32), 81 + i, dtype=np.float32)
            # add dead pixel at (22, 8)
            data[22, 8] = 0
            # add pixel for gamma spike at (7, 19)
            data[7, 19] = 1000
            img = Image.fromarray(data)
            exif = img.getexif()
            exif[65027] = "ExposureTime:30.000000"
            exif[65022] = f"RunNo:{1000 + i}"
            exif[65025] = "ModelStr:DW936_BV"
            filename = tmp_dir / f"sample_bad_{i:03}.tiff"
            img.save(filename, exif=exif)
            cls.sample_paths_bad_pixels.append(filename)

        # create 3 OB tiffs with values 99, 100, 101 and metadata
        for i in range(3):
            data = np.full((32, 32), 99 + i, dtype=np.float32)
            img = Image.fromarray(data)
            filename = tmp_dir / f"ob_{i:03}.tiff"
            img.save(filename, exif=exif)
            cls.ob_paths.append(filename)

        # create 2 dark tiffs with values 4 and 6 and metadata
        for i in range(2):
            data = np.full((32, 32), 4 + 2 * i, dtype=np.float32)
            img = Image.fromarray(data)
            filename = tmp_dir / f"dark_{i:03}.tiff"
            img.save(filename, exif=exif)
            cls.dark_paths.append(filename)

    @classmethod
    def teardown_class(cls):
        """Remove all temp test files after all tests in this class have run."""
        cls._tmpdir.cleanup()

    def test_mars_ccd_pipeline_hdf5(self):
        """
        Test the MARS CCD pipeline end-to-end with HDF5 output and verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            run_mars_ccd_pipeline(
                sample_paths=self.sample_paths,
                ob_paths=self.ob_paths,
                dark_paths=self.dark_paths,
                output_path=output_path,
            )

            assert output_path.exists()

            # Read back the file and check contents
            with h5py.File(output_path, "r") as hf:
                # Check transmission data
                assert "transmission" in hf
                assert hf["transmission"].shape == (5, 32, 32)
                # check that transmission values are correct based on the formula
                # T = (S - AVERAGE(D)) / (AVERAGE(OB) - AVERAGE(D))
                for i in range(5):
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i - 5) / (100 - 5))
                assert hf["transmission"].attrs["units"] == "dimensionless"
                assert hf["transmission"].dtype == np.float32
                # Check uncertainty data exists and is reasonable
                assert "uncertainty" in hf
                assert hf["uncertainty"].dtype == np.float32
                np.testing.assert_allclose(hf["uncertainty"], 0.1, rtol=0.15)
                # Check coordinates
                assert "x" in hf
                np.testing.assert_equal(hf["x"], np.arange(32))
                assert "y" in hf
                np.testing.assert_equal(hf["y"], np.arange(32))
                # Check masks
                assert "masks/dead" in hf
                np.testing.assert_equal(hf["masks/dead"], np.zeros((32, 32), dtype=bool))
                # Check metadata
                assert "metadata/sample_paths" in hf
                np.testing.assert_equal(hf["metadata/sample_paths"].asstr()[:], [str(p) for p in self.sample_paths])
                assert "metadata/ob_paths" in hf
                np.testing.assert_equal(hf["metadata/ob_paths"].asstr()[:], [str(p) for p in self.ob_paths])
                assert "metadata/dark_paths" in hf
                np.testing.assert_equal(hf["metadata/dark_paths"].asstr()[:], [str(p) for p in self.dark_paths])
                assert "metadata/gamma_filter_applied" in hf
                np.testing.assert_equal(hf["metadata/gamma_filter_applied"][()], True)
                assert "metadata/processing_timestamp" in hf
                assert "metadata/version" in hf
                np.testing.assert_equal(hf["metadata/version"].asstr()[()], __version__)
                assert "metadata/roi_applied" not in hf  # ROI not applied in this test
                # check metadata from files, RunNo, ExposureTime and Model
                assert "RunNo" in hf
                np.testing.assert_equal(hf["RunNo"][:], [1000, 1001, 1002, 1003, 1004])
                assert "ExposureTime" in hf
                np.testing.assert_equal(hf["ExposureTime"][:], [30] * 5)
                assert "ModelStr" in hf
                np.testing.assert_equal(hf["ModelStr"].asstr()[()], "DW936_BV")

    def test_mars_ccd_pipeline_roi_and_tiff(self):
        """
        Test the MARS CCD pipeline with ROI applied and TIFF output, then verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
            output_path = Path(f.name)

            run_mars_ccd_pipeline(
                sample_paths=self.sample_paths,
                ob_paths=self.ob_paths,
                dark_paths=self.dark_paths,
                output_path=output_path,
                roi=(5, 5, 25, 25),
            )

            assert output_path.exists()

            # Load back the file to verify contents
            dg = load_scitiff(output_path)

        assert isinstance(dg, sc.DataGroup)
        assert "image" in dg
        assert "daq" in dg
        assert "extra" in dg

        # Check image data and metadata
        image = dg["image"]
        assert image.dtype == sc.DType.float32
        assert image.dims == ("z", "y", "x")
        assert image.values.shape == (5, 20, 20)
        # check that transmission values are correct based on the formula
        # T = (S - AVERAGE(D)) / (AVERAGE(OB) - AVERAGE(D))
        for i in range(5):
            np.testing.assert_allclose(image.values[i], (81 + i - 5) / (100 - 5))
        # Check uncertainty data exists and is reasonable
        np.testing.assert_allclose(image.variances, 0.011, rtol=0.15)
        assert "scitiff-mask" in image.masks
        assert image.masks["scitiff-mask"].shape == (5, 20, 20)
        np.testing.assert_array_equal(image.masks["scitiff-mask"].values, False)

        # Check DAQ metadata
        daq = dg["daq"]  # this is type scitiff.DAQMetadata
        assert daq.facility == "HFIR"
        assert daq.instrument == "MARS"
        assert daq.detector_type == "DW936_BV"
        assert daq.source_type == "neutron"

        # Check extra metadata
        extra = dg["extra"]
        assert len(extra["sample_paths"].values) == 5
        assert len(extra["ob_paths"].values) == 3
        assert len(extra["dark_paths"].values) == 2
        assert extra["gamma_filter_applied"]

        assert "processing_timestamp" in extra
        np.testing.assert_equal(extra["roi_applied"].value, (5, 5, 25, 25))
        assert extra["version"] == __version__

    def test_mars_ccd_pipeline_dark_gamma_spike_pixels(self):
        """Test that the dark and gamma spike pixels are correctly handled.

        For this test just check the returned DataArray instead of looking at the output file.
        """

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_mars_ccd_pipeline(
                sample_paths=self.sample_paths_bad_pixels,
                ob_paths=self.ob_paths,
                dark_paths=self.dark_paths,
                output_path=output_path,
                roi=(5, 5, 25, 25),
            )

            assert output_path.exists()

        assert transmission.shape == (5, 20, 20)
        for i in range(5):
            expected_value = np.full((20, 20), (81 + i - 5) / (100 - 5))
            expected_value[22 - 5, 8 - 5] = 0  # dead pixel should be zero after dark subtraction and normalization
            # gamma spike should be replaced with the same values from that image
            np.testing.assert_allclose(transmission.values[i], expected_value)

        # check that the variances are higher for the gamma spike pixel and near zero for the dead pixel
        approximate_variances = np.full((5, 20, 20), 0.011)
        approximate_variances[:, 22 - 5, 8 - 5] = 0  # dead pixel should have almost zero variance
        approximate_variances[:, 7 - 5, 19 - 5] = 0.114  # gamma spike pixel should have higher variance
        np.testing.assert_allclose(transmission.variances, approximate_variances, atol=0.002)

        # The mask should only have the dead pixel masked
        expected_dead_pixel_mask = np.zeros((20, 20), dtype=bool)
        expected_dead_pixel_mask[22 - 5, 8 - 5] = True
        np.testing.assert_array_equal(transmission.masks["dead_pixels"].values, expected_dead_pixel_mask)


class TestMarsCCDPipelineFITS:
    """Tests for the MARS CCD pipeline using FITS files."""

    @classmethod
    def setup_class(cls):
        """Create FITS files for testing once for all tests in this class."""
        cls.sample_paths = []
        cls.ob_paths = []
        cls.dark_paths = []

        cls._tmpdir = tempfile.TemporaryDirectory(delete=False)
        tmp_dir = Path(cls._tmpdir.name)

        # create 5 sample FITS files with values 81-85 and metadata.
        for i in range(5):
            data = np.full((32, 32), 81 + i, dtype=np.float32)
            hdu = fits.PrimaryHDU(data)
            hdu.header["EXPOSURETIME"] = 30.0
            hdu.header["RUNNO"] = 1000 + i
            hdu.header["MODEL"] = "DW936_BV"
            filename = tmp_dir / f"sample_{i:03}.fits"
            hdu.writeto(filename, overwrite=True)
            cls.sample_paths.append(filename)

        # create 3 OB FITS files with values 99, 100, 101 and metadata
        for i in range(3):
            data = np.full((32, 32), 99 + i, dtype=np.float32)
            hdu = fits.PrimaryHDU(data)
            filename = tmp_dir / f"ob_{i:03}.fits"
            hdu.writeto(filename, overwrite=True)
            cls.ob_paths.append(filename)

        # create 2 dark FITS files with values 4 and 6 and metadata
        for i in range(2):
            data = np.full((32, 32), 4 + 2 * i, dtype=np.float32)
            hdu = fits.PrimaryHDU(data)
            filename = tmp_dir / f"dark_{i:03}.fits"
            hdu.writeto(filename, overwrite=True)
            cls.dark_paths.append(filename)

    @classmethod
    def teardown_class(cls):
        """Remove all temp test files after all tests in this class have run."""
        cls._tmpdir.cleanup()

    def test_mars_ccd_pipeline_fits_output(self):
        """
        Test the MARS CCD pipeline end-to-end with FITS input and HDF5 output and verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            run_mars_ccd_pipeline(
                sample_paths=self.sample_paths,
                ob_paths=self.ob_paths,
                dark_paths=self.dark_paths,
                output_path=output_path,
            )

            assert output_path.exists()

            # Read back the file and check contents
            with h5py.File(output_path, "r") as hf:
                # Check transmission data
                assert "transmission" in hf
                assert hf["transmission"].shape == (5, 32, 32)
                # check that transmission values are correct based on the formula
                # T = (S - AVERAGE(D)) / (AVERAGE(OB) - AVERAGE(D))
                for i in range(5):
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i - 5) / (100 - 5))
                assert hf["transmission"].attrs["units"] == "dimensionless"
                assert hf["transmission"].dtype == np.float32
                # Check uncertainty data exists and is reasonable
                assert "uncertainty" in hf
                assert hf["uncertainty"].dtype == np.float32
                np.testing.assert_allclose(hf["uncertainty"], 0.1, rtol=0.15)
                # Check coordinates
                assert "x" in hf
                np.testing.assert_equal(hf["x"], np.arange(32))
                assert "y" in hf
                np.testing.assert_equal(hf["y"], np.arange(32))
                # Check masks
                assert "masks/dead" in hf
                np.testing.assert_equal(hf["masks/dead"], np.zeros((32, 32), dtype=bool))
                # Check metadata
                assert "metadata/sample_paths" in hf
                np.testing.assert_equal(hf["metadata/sample_paths"].asstr()[:], [str(p) for p in self.sample_paths])
                assert "metadata/ob_paths" in hf
                np.testing.assert_equal(hf["metadata/ob_paths"].asstr()[:], [str(p) for p in self.ob_paths])
                assert "metadata/dark_paths" in hf
                np.testing.assert_equal(hf["metadata/dark_paths"].asstr()[:], [str(p) for p in self.dark_paths])
                assert "metadata/gamma_filter_applied" in hf
                np.testing.assert_equal(hf["metadata/gamma_filter_applied"][()], True)
                assert "metadata/processing_timestamp" in hf
                assert "metadata/version" in hf
                np.testing.assert_equal(hf["metadata/version"].asstr()[()], __version__)
                assert "metadata/roi_applied" not in hf  # ROI not applied in this test
                # check metadata from files, RunNo, ExposureTime and Model
                assert "RUNNO" in hf
                np.testing.assert_equal(hf["RUNNO"][:], [1000, 1001, 1002, 1003, 1004])
                assert "EXPOSURETIME" in hf
                np.testing.assert_equal(hf["EXPOSURETIME"][()], 30)
                assert "MODEL" in hf
                np.testing.assert_equal(hf["MODEL"].asstr()[()], "DW936_BV")
