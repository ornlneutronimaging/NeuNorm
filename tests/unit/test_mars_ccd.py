import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
from astropy.io import fits
from PIL import Image
from scitiff.io import load_scitiff

from neunorm import __version__
from neunorm.loaders.stack_loader import load_stack
from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline
from neunorm.processing.normalizer import normalize_transmission
from neunorm.processing.reference_preparer import prepare_reference
from neunorm.processing.run_combiner import combine_runs


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
            exif[65025] = "ManufacturerStr:DW936_BV"
            exif[65052] = "MotSlitVB.RBV:42.3"
            exif[65054] = "MotSlitVT.RBV:42.8"
            exif[65056] = "MotSlitHR.RBV:41.4"
            exif[65058] = "MotSlitHL.RBV:42.4"
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
            exif[65025] = "ManufacturerStr:DW936_BV"
            filename = tmp_dir / f"sample_bad_{i:03}.tiff"
            img.save(filename, exif=exif)
            cls.sample_paths_bad_pixels.append(filename)

        # create 3 OB tiffs with values 99, 100, 101 and metadata
        for i in range(3):
            data = np.full((32, 32), 99 + i, dtype=np.float32)
            img = Image.fromarray(data)
            exif = img.getexif()
            exif[65027] = "ExposureTime:30.000000"
            exif[65025] = "ManufacturerStr:DW936_BV"
            exif[65052] = "MotSlitVB.RBV:42.3"
            exif[65054] = "MotSlitVT.RBV:42.8"
            exif[65056] = "MotSlitHR.RBV:41.4"
            exif[65058] = "MotSlitHL.RBV:42.4"
            filename = tmp_dir / f"ob_{i:03}.tiff"
            img.save(filename, exif=exif)
            cls.ob_paths.append(filename)

        # create 2 dark tiffs with values 4 and 6 and metadata
        for i in range(2):
            data = np.full((32, 32), 4 + 2 * i, dtype=np.float32)
            img = Image.fromarray(data)
            exif = img.getexif()
            exif[65027] = "ExposureTime:30.000000"
            exif[65022] = f"RunNo:{1000 + i}"
            exif[65025] = "ManufacturerStr:DW936_BV"
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
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                dark_paths=[self.dark_paths],
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
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i - 5) / (100 - 5), rtol=1e-5)
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
                # Check metadata (nested per-run paths stored as round-trippable JSON)
                assert "metadata/sample_paths" in hf
                assert json.loads(hf["metadata/sample_paths"].asstr()[()]) == [[str(p) for p in self.sample_paths]]
                assert "metadata/ob_paths" in hf
                assert json.loads(hf["metadata/ob_paths"].asstr()[()]) == [[str(p) for p in self.ob_paths]]
                assert "metadata/dark_paths" in hf
                assert json.loads(hf["metadata/dark_paths"].asstr()[()]) == [[str(p) for p in self.dark_paths]]
                assert "metadata/dark_correction_applied" in hf
                np.testing.assert_equal(hf["metadata/dark_correction_applied"][()], True)
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
                assert "ManufacturerStr" in hf
                np.testing.assert_equal(hf["ManufacturerStr"].asstr()[()], "DW936_BV")

    def test_mars_ccd_pipeline_roi_and_tiff(self):
        """
        Test the MARS CCD pipeline with ROI applied and TIFF output, then verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
            output_path = Path(f.name)

            run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                dark_paths=[self.dark_paths],
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
            np.testing.assert_allclose(image.values[i], (81 + i - 5) / (100 - 5), rtol=1e-5)
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
                sample_paths=[self.sample_paths_bad_pixels],
                ob_paths=[self.ob_paths],
                dark_paths=[self.dark_paths],
                output_path=output_path,
                roi=(5, 5, 25, 25),
            )

            assert output_path.exists()

        assert transmission.shape == (5, 20, 20)
        for i in range(5):
            expected_value = np.full((20, 20), (81 + i - 5) / (100 - 5))
            expected_value[22 - 5, 8 - 5] = 0  # dead pixel should be zero after dark subtraction and normalization
            # gamma spike should be replaced with the same values from that image
            np.testing.assert_allclose(transmission.values[i], expected_value, rtol=1e-5)

        # check that the variances are higher for the gamma spike pixel and near zero for the dead pixel
        approximate_variances = np.full((5, 20, 20), 0.011)
        approximate_variances[:, 22 - 5, 8 - 5] = 0  # dead pixel should have almost zero variance
        approximate_variances[:, 7 - 5, 19 - 5] = 0.114  # gamma spike pixel should have higher variance
        np.testing.assert_allclose(transmission.variances, approximate_variances, atol=0.002)

        # The mask should only have the dead pixel masked
        expected_dead_pixel_mask = np.zeros((20, 20), dtype=bool)
        expected_dead_pixel_mask[22 - 5, 8 - 5] = True
        np.testing.assert_array_equal(transmission.masks["dead_pixels"].values, expected_dead_pixel_mask)

    def test_mars_ccd_pipeline_combine_runs(self):
        """Test that multiple runs are combined correctly.
        Include the same sample paths 3 times, OB twice and dark twice to check that the combination is working.
        Data should double along with selected metadata.

        Just check the returned DataArray instead of looking at the output file."""

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_mars_ccd_pipeline(
                sample_paths=[
                    self.sample_paths,
                    self.sample_paths,
                    self.sample_paths,
                ],  # include the same sample paths three times to test combination
                ob_paths=[self.ob_paths, self.ob_paths],  # include the same OB paths twice to test combination
                dark_paths=[self.dark_paths, self.dark_paths],  # include the same dark paths twice to test combination
                output_path=output_path,
                roi=(5, 5, 25, 25),
            )

            assert output_path.exists()

        assert transmission.shape == (5, 20, 20)
        # values should be the same as the original sample values since we are combining three runs but also normalizing
        # by the number of runs, so it should not change the final transmission values, just reduce the variance.
        for i in range(5):
            expected_value = np.full((20, 20), ((81 + i) - 5) / (100 - 5))
            np.testing.assert_allclose(transmission.values[i], expected_value, rtol=1e-5)

        # check that the variances exist and are reasonable
        np.testing.assert_allclose(transmission.variances, 0.004, atol=0.002)

        # The mask should only have the dead pixel masked
        expected_dead_pixel_mask = np.zeros((20, 20), dtype=bool)
        np.testing.assert_array_equal(transmission.masks["dead_pixels"].values, expected_dead_pixel_mask)

        # check that the metadata keys that should be summed are unchanged
        assert "ExposureTime" in transmission.coords
        np.testing.assert_equal(
            transmission.coords["ExposureTime"].values, [30] * 5
        )  # should be unchanged since we are normalizing by the number of runs
        assert "ManufacturerStr" in transmission.coords
        np.testing.assert_equal(
            transmission.coords["ManufacturerStr"].values, "DW936_BV"
        )  # should be unchanged since it's the same for both runs
        assert "MotSlitVB.RBV" in transmission.coords
        np.testing.assert_equal(
            transmission.coords["MotSlitVB.RBV"].values, 42.3
        )  # should be unchanged since it's the same for both runs
        assert "MotSlitVT.RBV" in transmission.coords
        np.testing.assert_equal(
            transmission.coords["MotSlitVT.RBV"].values, 42.8
        )  # should be unchanged since it's the same for both runs
        assert "MotSlitHR.RBV" in transmission.coords
        np.testing.assert_equal(
            transmission.coords["MotSlitHR.RBV"].values, 41.4
        )  # should be unchanged since it's the same for both runs
        assert "MotSlitHL.RBV" in transmission.coords
        np.testing.assert_equal(
            transmission.coords["MotSlitHL.RBV"].values, 42.4
        )  # should be unchanged since it's the same for both runs

    def test_mars_ccd_pipeline_no_dark(self):
        """Dark current is optional: omitting dark_paths skips dark correction.

        Without dark subtraction the transmission is T = S / OB = (81 + i) / 100
        for these fixtures (vs (81 + i - 5) / (100 - 5) with dark). MARS does not
        apply a proton-charge correction.
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=output_path,
            )

            assert output_path.exists()
            assert transmission.shape == (5, 32, 32)

            with h5py.File(output_path, "r") as hf:
                assert hf["transmission"].shape == (5, 32, 32)
                # No dark subtraction: T = S / OB = (81 + i) / 100
                for i in range(5):
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i) / 100, rtol=1e-5)
                # Uncertainty is present, positive and finite
                assert "uncertainty" in hf
                assert np.all(np.isfinite(hf["uncertainty"][:]))
                assert np.all(hf["uncertainty"][:] > 0)
                # Provenance: dark correction not applied, dark_paths omitted entirely
                assert "metadata/dark_correction_applied" in hf
                np.testing.assert_equal(hf["metadata/dark_correction_applied"][()], False)
                assert "metadata/dark_paths" not in hf

    def test_mars_ccd_pipeline_empty_dark_paths(self):
        """An empty dark_paths list is treated the same as omitting dark."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            transmission = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                dark_paths=[],
                output_path=Path(f.name),
            )
        assert transmission.shape == (5, 32, 32)
        for i in range(5):
            np.testing.assert_allclose(transmission.values[i], (81 + i) / 100, rtol=1e-5)

    def test_mars_ccd_pipeline_requires_output_path(self):
        """output_path is required even though it carries a default for signature compatibility."""
        with pytest.raises(ValueError, match="output_path is required"):
            run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
            )

    def test_mars_ccd_pipeline_no_dark_uncertainty(self):
        """No-dark UQ equals the dark-free propagation, with no dark-frame variance term.

        Two independent checks:
        1. The pipeline's no-dark output (values AND variances) matches a direct
           composition of the same library functions with dark subtraction omitted,
           proving no spurious variance term is added or dropped.
        2. Removing the dark removes its variance contribution, so for these uniform
           fixtures the no-dark uncertainty is smaller than the with-dark uncertainty
           everywhere. (Fixture-specific, not a universal law — dark subtraction also
           shrinks the numerator/denominator; the pinned oracle above is what guards
           the general Var(T) property.)
        """
        match_keys = [
            "ExposureTime",
            "ManufacturerStr",
            "MotSlitVB.RBV",
            "MotSlitVT.RBV",
            "MotSlitHR.RBV",
            "MotSlitHL.RBV",
        ]
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            no_dark = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=Path(f.name),
                gamma_filter=False,
            )

        # Independent dark-free propagation using the same composable functions the pipeline uses.
        sample = combine_runs(
            [load_stack(self.sample_paths)],
            metadata_keys_to_sum=("ExposureTime",),
            metadata_check_match=match_keys,
            normalize_by_runs=True,
        )
        ob = combine_runs(
            [load_stack(self.ob_paths)],
            metadata_keys_to_sum=("ExposureTime",),
            metadata_check_match=match_keys,
            normalize_by_runs=True,
        )
        ob = prepare_reference(ob, dim="N_image")
        expected = normalize_transmission(sample, ob)
        # The pipeline produces float32 normalized data end-to-end. MARS
        # has no proton-charge division, so loaders being float32 keeps the whole path
        # float32 and this structural check stays bit-tight against the pipeline output.
        assert no_dark.dtype == sc.DType.float32
        np.testing.assert_allclose(no_dark.values, expected.values)
        np.testing.assert_allclose(no_dark.variances, expected.variances)

        # Independent (non-circular) oracle: pin the no-dark variance to literal
        # values, NOT recomposed from the same helpers. A regression inside the
        # shared variance propagation would shift `expected` in lock-step but not
        # these constants, so this is what actually guards the Var(T) property.
        expected_var_per_image = np.array([0.010287, 0.010441, 0.010596, 0.010752, 0.010908])
        expected_var = np.broadcast_to(expected_var_per_image[:, None, None], no_dark.variances.shape)
        np.testing.assert_allclose(no_dark.variances, expected_var, rtol=1e-3)

        # With dark, the dark-frame variance is folded into both numerator and
        # denominator, raising the uncertainty for these fixtures.
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            with_dark = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                dark_paths=[self.dark_paths],
                output_path=Path(f.name),
                gamma_filter=False,
            )
        assert with_dark.dtype == sc.DType.float32
        assert np.all(no_dark.variances < with_dark.variances)


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
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                dark_paths=[self.dark_paths],
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
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i - 5) / (100 - 5), rtol=1e-5)
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
                # Check metadata (nested per-run paths stored as round-trippable JSON)
                assert "metadata/sample_paths" in hf
                assert json.loads(hf["metadata/sample_paths"].asstr()[()]) == [[str(p) for p in self.sample_paths]]
                assert "metadata/ob_paths" in hf
                assert json.loads(hf["metadata/ob_paths"].asstr()[()]) == [[str(p) for p in self.ob_paths]]
                assert "metadata/dark_paths" in hf
                assert json.loads(hf["metadata/dark_paths"].asstr()[()]) == [[str(p) for p in self.dark_paths]]
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

    def test_mars_ccd_pipeline_background_roi(self):
        """background_roi flux normalization runs end-to-end and changes the result."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            t_default = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths], ob_paths=[self.ob_paths], output_path=output_path, gamma_filter=False
            )
            t_bg = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=output_path,
                gamma_filter=False,
                background_roi=(0, 0, 8, 8),
            )
        assert str(t_bg.unit) == "dimensionless"
        assert t_bg.shape == (5, 32, 32)
        assert t_bg.dtype == np.float32
        # spatially-uniform images -> background-ROI normalization cancels to T = 1 everywhere
        np.testing.assert_allclose(t_bg.values, 1.0, rtol=1e-5)
        # and it differs from the plain S/OB normalization
        assert not np.allclose(t_bg.values, t_default.values)

    def test_mars_ccd_pipeline_background_roi_with_dark(self):
        """background_roi + a dark frame routes through normalize_with_dark."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            t = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                dark_paths=[self.dark_paths],
                output_path=output_path,
                gamma_filter=False,
                background_roi=(0, 0, 8, 8),
            )
        assert str(t.unit) == "dimensionless"
        assert t.shape == (5, 32, 32)
        # uniform images -> background-ROI normalization cancels to T = 1 even after dark subtraction
        np.testing.assert_allclose(t.values, 1.0, rtol=1e-5)

    def test_mars_ccd_pipeline_background_roi_accepts_roi_object(self):
        """A background_roi ROI yields the same transmission as the equivalent tuple."""
        from neunorm.data_models.roi import ROI

        with (
            tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f1,
            tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f2,
        ):
            t_tuple = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=Path(f1.name),
                gamma_filter=False,
                background_roi=(0, 0, 8, 8),
            )
            # width/height form, resolved to the same (0, 0, 8, 8) bounds
            t_roi = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=Path(f2.name),
                gamma_filter=False,
                background_roi=ROI(x0=0, y0=0, width=8, height=8),
            )
        np.testing.assert_array_equal(t_tuple.values, t_roi.values)

    def test_mars_ccd_pipeline_crop_roi_accepts_roi_object(self):
        """A crop roi=ROI(...) crops correctly AND is coerced to a tuple in the written provenance.

        Guards the pipeline-level coercion specifically: ``apply_roi`` coerces internally (so the
        shape is right regardless), and the HDF5 writer would NOT crash on a raw ROI — it would
        str()-coerce it via the JSON backstop (``encoding="json"``). So we round-trip
        ``roi_applied`` and assert it is the native int array (the coerced tuple), which fails if a
        raw ROI ever reaches provenance.
        """
        from neunorm.data_models.roi import ROI

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            t = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=output_path,
                gamma_filter=False,
                roi=ROI(x0=5, y0=5, x1=25, y1=25),
            )
            assert t.shape == (5, 20, 20)
            with h5py.File(output_path, "r") as hf:
                ds = hf["metadata/roi_applied"]
                # stored as a native int array (the coerced tuple), NOT the JSON str(ROI) fallback
                assert ds.attrs.get("encoding") != "json"
                np.testing.assert_array_equal(ds[()], [5, 5, 25, 25])

    def test_mars_ccd_pipeline_pooled_background_rois(self):
        """background_roi accepts a pooled SEQUENCE of ROIs end-to-end (#172); provenance stores the list."""
        from neunorm.data_models.roi import ROI

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            t = run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=output_path,
                gamma_filter=False,
                background_roi=[ROI(x0=0, y0=0, width=8, height=8), ROI(x0=10, y0=10, width=8, height=8)],
            )
            assert t.shape == (5, 32, 32)
            # uniform images -> the pooled flux proxy still cancels to T = 1
            np.testing.assert_allclose(t.values, 1.0, rtol=1e-5)
            with h5py.File(output_path, "r") as hf:
                # a pooled (multi-ROI) list is stored as a JSON string — round-trippable AND safe on
                # the flattening TIFF writer (unlike a raw nested list).
                assert json.loads(hf["metadata/background_roi"].asstr()[()]) == [[0, 0, 8, 8], [10, 10, 18, 18]]

    def test_mars_ccd_pipeline_single_background_roi_provenance_is_flat(self):
        """A single background_roi keeps the pre-#172 flat int-array provenance (backward compatible)."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)
            run_mars_ccd_pipeline(
                sample_paths=[self.sample_paths],
                ob_paths=[self.ob_paths],
                output_path=output_path,
                gamma_filter=False,
                background_roi=(0, 0, 8, 8),
            )
            with h5py.File(output_path, "r") as hf:
                ds = hf["metadata/background_roi"]
                assert ds.attrs.get("encoding") != "json"  # native int array, not the JSON backstop
                np.testing.assert_array_equal(ds[()], [0, 0, 8, 8])
