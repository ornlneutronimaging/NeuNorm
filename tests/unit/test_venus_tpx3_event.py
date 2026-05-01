import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipp as sc
from scitiff.io import load_scitiff

from neunorm import __version__
from neunorm.data_models.tof import BinningConfig
from neunorm.pipelines.venus_tpx3_event import run_venus_tpx3_event_pipeline


class TestVenusTPX3EventPipeline:
    """Tests for the VENUS TPX3 event pipeline."""

    @classmethod
    def setup_class(cls):
        """Create HDF5 event files for testing once for all tests in this class."""
        cls.binning = BinningConfig(bins=5, bin_space="tof", tof_range=(100000, 125000), use_log_bin=False)  # ns

        cls._tmpdir = tempfile.TemporaryDirectory(delete=False)
        tmp_dir = Path(cls._tmpdir.name)

        # for our test data we have 32x32 detector
        x_values = np.tile(np.arange(32), (32, 1)).flatten()
        y_values = np.tile(np.arange(32), (32, 1)).T.flatten()
        event_ids = x_values + y_values * 32

        # create sample HDF5 with metadata.
        cls.sample = tmp_dir / "sample.hdf5"
        with h5py.File(cls.sample, "w") as hf:
            # HDF5 format:
            # /entry/bank1_events/event_id
            # /entry/bank1_events/event_time_offset
            entry = hf.create_group("entry")
            entry.create_dataset("proton_charge", data=[12345])
            entry.create_dataset("duration", data=[60.0])
            bank = entry.create_group("bank100_events")
            # repeat the same values to increase counts in each
            file_event_ids = []
            file_tofs = []
            for i in range(5):
                tofs = np.full(32 * 32, 100 + i * 5, dtype=np.int64)  # us
                repeats = 5 + i
                file_event_ids.extend(np.tile(event_ids, repeats))
                file_tofs.extend(np.tile(tofs, repeats))
            bank.create_dataset("event_time_offset", data=file_tofs)
            bank.create_dataset("event_id", data=file_event_ids, dtype=np.int32)
            daslogs = entry.create_group("DASlogs")
            time_offset_path = daslogs.create_group("BL10:Det:TH:DSPT1:TIDelay")
            time_offset_path.create_dataset("average_value", data=[5000])
            detector_path = daslogs.create_group("BL10:Exp:Det")
            detector_path.create_dataset("value_strings", data=[[b"MCP TPX3"]])

        # create OB HDF5 with values 99, 100, 101 and metadata
        cls.ob = tmp_dir / "ob.hdf5"
        with h5py.File(cls.ob, "w") as hf:
            # repeat the same values to increase counts in each
            entry = hf.create_group("entry")
            entry.create_dataset("proton_charge", data=[12345 * 2])
            entry.create_dataset("duration", data=[60.0])
            bank = entry.create_group("bank100_events")
            # repeat the same values to increase counts in each
            file_event_ids = []
            file_tofs = []
            for i in range(5):
                tofs = np.full(32 * 32, 100 + i * 5, dtype=np.int64)
                repeats = 10 + i
                file_event_ids.extend(np.tile(event_ids, repeats))
                file_tofs.extend(np.tile(tofs, repeats))
            bank.create_dataset("event_time_offset", data=file_tofs)
            bank.create_dataset("event_id", data=file_event_ids, dtype=np.int32)
            daslogs = entry.create_group("DASlogs")
            time_offset_path = daslogs.create_group("BL10:Det:TH:DSPT1:TIDelay")
            time_offset_path.create_dataset("average_value", data=[5000])
            detector_path = daslogs.create_group("BL10:Exp:Det")
            detector_path.create_dataset("value_strings", data=[[b"MCP TPX3"]])

        # create OB HDF5 with metadata. These have a dead pixel and a hot pixel.
        # add dead pixel at (22, 8)
        # add pixel for hot pixel at (7, 19)
        cls.ob_bad_pixels = tmp_dir / "ob_bad_pixels.hdf5"
        with h5py.File(cls.ob_bad_pixels, "w") as hf:
            t = []
            ids = []
            for i in range(5):
                tofs = np.full(32 * 32, 100 + i * 5, dtype=np.int64)  # us
                for tof, event_id in zip(tofs, event_ids):
                    if event_id == 8 + 22 * 32:  # dead pixel (8, 22)
                        continue  # skip events for dead pixel
                    elif event_id == 19 + 7 * 32:  # hot pixel (19, 7)
                        t.extend([tof] * 1000)  # add many events for hot pixel
                        ids.extend([event_id] * 1000)
                    else:
                        t.extend([tof] * (10 + i))
                        ids.extend([event_id] * (10 + i))
            entry = hf.create_group("entry")
            entry.create_dataset("proton_charge", data=[12345 * 2])
            entry.create_dataset("duration", data=[60.0])
            bank = entry.create_group("bank100_events")
            bank.create_dataset("event_time_offset", data=t)
            bank.create_dataset("event_id", data=ids, dtype=np.int32)
            daslogs = entry.create_group("DASlogs")
            time_offset_path = daslogs.create_group("BL10:Det:TH:DSPT1:TIDelay")
            time_offset_path.create_dataset("average_value", data=[5000])
            detector_path = daslogs.create_group("BL10:Exp:Det")
            detector_path.create_dataset("value_strings", data=[[b"MCP TPX3"]])

    @classmethod
    def teardown_class(cls):
        """Remove all temp test files after all tests in this class have run."""
        cls._tmpdir.cleanup()

    def test_venus_tpx3_event_pipeline_hdf5(self):
        """
        Test the VENUS TPX3 event pipeline end-to-end with HDF5 output and verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            run_venus_tpx3_event_pipeline(
                sample_paths=[self.sample],
                ob_paths=[self.ob],
                binning=self.binning,
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
                # T = S / OB * (proton_charge_ob / proton_charge_sample)
                for i in range(5):
                    np.testing.assert_allclose(hf["transmission"][i], (5 + i) / (10 + i) * 2)
                assert hf["transmission"].attrs["units"] == "dimensionless"
                assert hf["transmission"].dtype == np.float32
                # Check uncertainty data exists and is reasonable
                assert "uncertainty" in hf
                assert hf["uncertainty"].dtype == np.float32
                np.testing.assert_allclose(hf["uncertainty"], 0.5477, rtol=0.1)
                # Check coordinates, bin edges should be 0-32 for both x and y
                assert "x" in hf
                np.testing.assert_equal(hf["x"], np.arange(33))
                assert "y" in hf
                np.testing.assert_equal(hf["y"], np.arange(33))
                assert "tof" in hf
                np.testing.assert_equal(hf["tof"], np.arange(100000, 130000, 5000))  # ns
                # Check masks
                assert "masks/dead" in hf
                np.testing.assert_equal(hf["masks/dead"], np.zeros((32, 32), dtype=bool))
                assert "masks/hot" in hf
                np.testing.assert_equal(hf["masks/hot"], np.zeros((32, 32), dtype=bool))
                # Check metadata
                assert "metadata/sample_paths" in hf
                np.testing.assert_equal(hf["metadata/sample_paths"].asstr()[:], [str(self.sample)])
                assert "metadata/ob_paths" in hf
                np.testing.assert_equal(hf["metadata/ob_paths"].asstr()[:], [str(self.ob)])
                assert "metadata/processing_timestamp" in hf
                assert "metadata/version" in hf
                np.testing.assert_equal(hf["metadata/version"].asstr()[()], __version__)
                assert "metadata/roi_applied" not in hf  # ROI not applied in this test

    def test_venus_tpx3_event_pipeline_roi_and_tiff(self):
        """
        Test the VENUS TPX3 event pipeline with ROI applied and TIFF output, then verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
            output_path = Path(f.name)

            run_venus_tpx3_event_pipeline(
                sample_paths=[self.sample],
                ob_paths=[self.ob],
                binning=self.binning,
                output_path=output_path,
                roi=(5, 5, 25, 25),
                detector_shape=(32, 32),
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
        assert image.dims == ("t", "y", "x")
        assert image.values.shape == (5, 20, 20)
        # check that transmission values are correct based on the formula
        # T = S / OB * (proton_charge_ob / proton_charge_sample)
        for i in range(5):
            np.testing.assert_allclose(image.values[i], (5 + i) / (10 + i) * 2)
        # Check uncertainty data exists and is reasonable
        np.testing.assert_allclose(image.variances, 0.3, rtol=0.1)
        assert "scitiff-mask" in image.masks
        assert image.masks["scitiff-mask"].shape == (5, 20, 20)
        np.testing.assert_array_equal(image.masks["scitiff-mask"].values, False)

        # Check DAQ metadata
        daq = dg["daq"]  # this is type scitiff.DAQMetadata
        assert daq.facility == "SNS"
        assert daq.instrument == "VENUS"
        assert daq.detector_type == "TPX3"
        assert daq.source_type == "neutron"

        # Check extra metadata
        extra = dg["extra"]
        assert len(extra["sample_paths"].values) == 1
        assert len(extra["ob_paths"].values) == 1

        assert "processing_timestamp" in extra
        np.testing.assert_equal(extra["roi_applied"].value, (5, 5, 25, 25))
        assert extra["version"] == __version__

    def test_venus_tpx3_event_pipeline_dark_and_hot_pixels(self):
        """Test that the dark and hot pixels are correctly handled.

        For this test just check the returned DataArray instead of looking at the output file.
        """

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx3_event_pipeline(
                sample_paths=[self.sample],
                ob_paths=[self.ob_bad_pixels],
                binning=self.binning,
                output_path=output_path,
                roi=(5, 5, 25, 25),
                detector_shape=(32, 32),
            )

            assert output_path.exists()

        assert transmission.shape == (5, 20, 20)
        for i in range(5):
            expected_value = np.full((20, 20), (5 + i) / (10 + i) * 2)
            expected_value[8 - 5, 22 - 5] = np.inf  # dead pixel should be zero after dark subtraction and normalization
            expected_value[19 - 5, 7 - 5] = (5 + i) / (1000) * 2  # hot pixel
            np.testing.assert_allclose(transmission.values[i], expected_value)

        # check that the variances are different for the hot pixel and the dead pixel
        approximate_variances = np.full((5, 20, 20), 0.3)
        approximate_variances[:, 8 - 5, 22 - 5] = np.nan  # dead pixel
        approximate_variances[:, 19 - 5, 7 - 5] = 0  # hot pixel
        np.testing.assert_allclose(transmission.variances, approximate_variances, atol=1e-1)

        # Check the dead pixel masked
        expected_dead_pixel_mask = np.zeros((20, 20), dtype=bool)
        expected_dead_pixel_mask[8 - 5, 22 - 5] = True
        np.testing.assert_array_equal(transmission.masks["dead_pixels"].values, expected_dead_pixel_mask)
        # Check hot pixel mask
        expected_hot_pixel_mask = np.zeros((20, 20), dtype=bool)
        expected_hot_pixel_mask[19 - 5, 7 - 5] = True
        np.testing.assert_array_equal(transmission.masks["hot_pixels"].values, expected_hot_pixel_mask)

    def test_venus_tpx3_event_pipeline_combining_runs(self):
        """Test combining multiple runs by providing multiple sample paths and multiple ob paths."""

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx3_event_pipeline(
                sample_paths=[self.sample, self.sample, self.sample],
                ob_paths=[self.ob, self.ob],
                binning=self.binning,
                output_path=output_path,
                detector_shape=(32, 32),
            )

            assert output_path.exists()

        assert transmission.shape == (5, 32, 32)
        # values should be the same as original sample values since we are normalizing by number of runs
        for i in range(5):
            expected_value = np.full((32, 32), (5 + i) / (10 + i) * 2)
            np.testing.assert_allclose(transmission.values[i], expected_value)

        # The variances should be less than the variance from a single run due to combining multiple runs
        np.testing.assert_allclose(transmission.variances, 0.117, rtol=0.1)
