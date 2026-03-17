import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipp as sc
from scitiff.io import load_scitiff

from neunorm import __version__
from neunorm.pipelines.mars_tpx3 import run_mars_tpx3_pipeline


class TestMarsTPX3Pipeline:
    """Tests for the MARS TPX3 pipeline."""

    @classmethod
    def setup_class(cls):
        """Create TPX3-style HDF5 event files for testing once for all tests in this class."""
        cls.sample_paths = []
        cls.sample_paths_bad_pixels = []
        cls.ob_paths = []

        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_dir = Path(cls._tmpdir.name)

        # for our test data we have 32x32 detector
        x_values = np.tile(np.arange(32), (32, 1)).flatten()
        y_values = np.tile(np.arange(32), (32, 1)).T.flatten()
        tofs = np.zeros(32 * 32, dtype=np.int64)  # all events at same TOF for simplicity

        # create 5 sample TPX3-style HDF5 with values 81-85 and metadata.
        for i in range(5):
            temp_path = tmp_dir / f"sample_{i:03}.hdf5"
            with h5py.File(temp_path, "w") as hf:
                # TPX3 format: tof, x, y arrays
                # repeat the same values to increase counts in each
                repeats = 81 + i
                hf.create_dataset("tof", data=np.repeat(tofs, repeats))
                hf.create_dataset("x", data=np.tile(x_values, repeats), dtype=np.int32)
                hf.create_dataset("y", data=np.tile(y_values, repeats), dtype=np.int32)
            cls.sample_paths.append(temp_path)

        # create 5 sample TPX3-style HDF5 with values 81-85 and metadata. These have a dead pixel and a hot pixel.
        for i in range(5):
            # add dead pixel at (22, 8)
            # add pixel for hot pixel at (7, 19)
            temp_path = tmp_dir / f"sample_bad_{i:03}.hdf5"
            with h5py.File(temp_path, "w") as hf:
                repeats = 81 + i
                t = []
                x = []
                y = []
                for tof, x_val, y_val in zip(tofs, x_values, y_values):
                    if (x_val, y_val) == (8, 22):  # dead pixel
                        continue  # skip events for dead pixel
                    elif (x_val, y_val) == (19, 7):  # hot pixel
                        t.extend([tof] * 1000)  # add many events for hot pixel
                        x.extend([x_val] * 1000)
                        y.extend([y_val] * 1000)
                    else:
                        t.extend([tof] * repeats)
                        x.extend([x_val] * repeats)
                        y.extend([y_val] * repeats)
                hf.create_dataset("tof", data=t)
                hf.create_dataset("x", data=x, dtype=np.int32)
                hf.create_dataset("y", data=y, dtype=np.int32)
            cls.sample_paths_bad_pixels.append(temp_path)

        # create 3 OB TPX3-style HDF5 with values 99, 100, 101 and metadata
        for i in range(3):
            temp_path = tmp_dir / f"ob_{i:03}.hdf5"
            with h5py.File(temp_path, "w") as hf:
                # repeat the same values to increase counts in each
                repeats = 99 + i
                hf.create_dataset("tof", data=np.repeat(tofs, repeats))
                hf.create_dataset("x", data=np.tile(x_values, repeats), dtype=np.int32)
                hf.create_dataset("y", data=np.tile(y_values, repeats), dtype=np.int32)
            cls.ob_paths.append(temp_path)

    @classmethod
    def teardown_class(cls):
        """Remove all temp test files after all tests in this class have run."""
        cls._tmpdir.cleanup()

    def test_mars_tpx3_pipeline_hdf5(self):
        """
        Test the MARS TPX3 pipeline end-to-end with HDF5 output and verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            run_mars_tpx3_pipeline(
                sample_paths=self.sample_paths,
                ob_paths=self.ob_paths,
                output_path=output_path,
                detector_shape=(32, 32),  # match our test data
            )

            assert output_path.exists()

            # Read back the file and check contents
            with h5py.File(output_path, "r") as hf:
                # Check transmission data
                assert "transmission" in hf
                assert hf["transmission"].shape == (5, 32, 32)
                # check that transmission values are correct based on the formula
                # T = S / OB
                for i in range(5):
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i) / (100))
                assert hf["transmission"].attrs["units"] == "dimensionless"
                assert hf["transmission"].dtype == np.float32
                # Check uncertainty data exists and is reasonable
                assert "uncertainty" in hf
                assert hf["uncertainty"].dtype == np.float32
                np.testing.assert_allclose(hf["uncertainty"], 0.1, rtol=0.15)
                # Check coordinates, bin edges should be 0-32 for both x and y
                assert "x" in hf
                np.testing.assert_equal(hf["x"], np.arange(33))
                assert "y" in hf
                np.testing.assert_equal(hf["y"], np.arange(33))
                # Check masks
                assert "masks/dead" in hf
                np.testing.assert_equal(hf["masks/dead"], np.zeros((32, 32), dtype=bool))
                assert "masks/hot" in hf
                np.testing.assert_equal(hf["masks/hot"], np.zeros((32, 32), dtype=bool))
                # Check metadata
                assert "metadata/sample_paths" in hf
                np.testing.assert_equal(hf["metadata/sample_paths"].asstr()[:], [str(p) for p in self.sample_paths])
                assert "metadata/ob_paths" in hf
                np.testing.assert_equal(hf["metadata/ob_paths"].asstr()[:], [str(p) for p in self.ob_paths])
                assert "metadata/gamma_filter_applied" in hf
                np.testing.assert_equal(hf["metadata/gamma_filter_applied"][()], True)
                assert "metadata/processing_timestamp" in hf
                assert "metadata/version" in hf
                np.testing.assert_equal(hf["metadata/version"].asstr()[()], __version__)
                assert "metadata/roi_applied" not in hf  # ROI not applied in this test

    def test_mars_tpx3_pipeline_roi_and_tiff(self):
        """
        Test the MARS TPX3 pipeline with ROI applied and TIFF output, then verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
            output_path = Path(f.name)

            run_mars_tpx3_pipeline(
                sample_paths=self.sample_paths,
                ob_paths=self.ob_paths,
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
            np.testing.assert_allclose(image.values[i], (81 + i) / 100)
        # Check uncertainty data exists and is reasonable
        np.testing.assert_allclose(image.variances, 0.011, rtol=0.15)
        assert "scitiff-mask" in image.masks
        assert image.masks["scitiff-mask"].shape == (5, 20, 20)
        np.testing.assert_array_equal(image.masks["scitiff-mask"].values, False)

        # Check DAQ metadata
        daq = dg["daq"]  # this is type scitiff.DAQMetadata
        assert daq.facility == "HFIR"
        assert daq.instrument == "MARS"
        assert daq.detector_type == "TPX3"
        assert daq.source_type == "neutron"

        # Check extra metadata
        extra = dg["extra"]
        assert len(extra["sample_paths"].values) == 5
        assert len(extra["ob_paths"].values) == 3
        assert extra["gamma_filter_applied"]

        assert "processing_timestamp" in extra
        np.testing.assert_equal(extra["roi_applied"].value, (5, 5, 25, 25))
        assert extra["version"] == __version__

    def test_mars_tpx3_pipeline_dark_and_hot_pixels(self):
        """Test that the dark and gamma spike pixels are correctly handled.

        For this test just check the returned DataArray instead of looking at the output file.
        """

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_mars_tpx3_pipeline(
                sample_paths=self.sample_paths_bad_pixels,
                ob_paths=self.ob_paths,
                output_path=output_path,
                roi=(5, 5, 25, 25),
                gamma_filter=False,  # disable gamma filter to test that hot pixel is not removed
            )

            assert output_path.exists()

        assert transmission.shape == (5, 20, 20)
        for i in range(5):
            expected_value = np.full((20, 20), (81 + i) / 100)
            expected_value[8 - 5, 22 - 5] = 0  # dead pixel should be zero after dark subtraction and normalization
            expected_value[19 - 5, 7 - 5] = (
                1000 / 100
            )  # hot pixel should have very high value since gamma filter is disabled
            np.testing.assert_allclose(transmission.values[i], expected_value)

        # check that the variances are higher for the hot pixel and near zero for the dead pixel
        approximate_variances = np.full((5, 20, 20), 0.011)
        approximate_variances[:, 8 - 5, 22 - 5] = 0  # dead pixel should have zero variance
        approximate_variances[:, 19 - 5, 7 - 5] = 0.433  # hot pixel should have higher variance
        np.testing.assert_allclose(transmission.variances, approximate_variances, atol=0.002)

        # Check the dead pixel masked
        expected_dead_pixel_mask = np.zeros((20, 20), dtype=bool)
        expected_dead_pixel_mask[8 - 5, 22 - 5] = True
        np.testing.assert_array_equal(transmission.masks["dead_pixels"].values, expected_dead_pixel_mask)
        # Check hot pixel mask
        expected_hot_pixel_mask = np.zeros((20, 20), dtype=bool)
        expected_hot_pixel_mask[19 - 5, 7 - 5] = True
        np.testing.assert_array_equal(transmission.masks["hot_pixels"].values, expected_hot_pixel_mask)
