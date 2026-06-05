import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
from PIL import Image
from scitiff.io import load_scitiff

from neunorm import __version__
from neunorm.pipelines.venus_tpx1 import run_venus_tpx1_pipeline


class TestVenusTPX1Pipeline:
    """Tests for the VENUS TPX1 pipeline."""

    @classmethod
    def setup_class(cls):
        """Create tiff files for testing once for all tests in this class."""
        cls.sample_tiff_paths = []
        cls.ob_tiff_paths = []

        cls._tmpdir = tempfile.TemporaryDirectory(delete=False)
        tmp_dir = Path(cls._tmpdir.name)

        # create 5 sample tiffs with values 81-85 and metadata.
        for i in range(5):
            data = np.full((32, 32), 81 + i, dtype=np.float32)
            img = Image.fromarray(data)
            filename = tmp_dir / f"sample_{i:03}.tiff"
            img.save(filename)
            cls.sample_tiff_paths.append(filename)

        # create 5 OB tiffs with values 99, 100, 101, 102, 103 and metadata
        for i in range(5):
            data = np.full((32, 32), 99 + i, dtype=np.float32)
            img = Image.fromarray(data)
            filename = tmp_dir / f"ob_{i:03}.tiff"
            img.save(filename)
            cls.ob_tiff_paths.append(filename)

        # Create temporary HDF5 file with minimal metadata
        cls.sample_nexus_path = tmp_dir / "nexus" / "test_sample_metadata.nxs.h5"
        cls.sample_nexus_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(cls.sample_nexus_path, "w") as f:
            entry = f.create_group("entry")
            entry.create_dataset("proton_charge", data=[12345])
            entry.create_dataset("duration", data=[60.0])
            daslogs = entry.create_group("DASlogs")
            image_file_path = daslogs.create_group("BL10:Exp:IM:ImageFilePath")
            image_file_path.create_dataset("value", data=[[b"images/sample"]])
            time_offset_path = daslogs.create_group("BL10:Det:TH:DSPT1:TIDelay")
            time_offset_path.create_dataset("average_value", data=[5000])
            detector_path = daslogs.create_group("BL10:Exp:Det")
            detector_path.create_dataset("value_strings", data=[[b"MCP TPX1"]])

        # Also create a directory for the image file and a spectra TOF file
        image_dir = tmp_dir / "images" / "sample"
        image_dir.mkdir(exist_ok=True, parents=True)
        spectra_tof_file = image_dir / "test_Spectra.txt"
        with open(spectra_tof_file, "w") as f:
            for i in range(6):
                f.write(f"{i * 0.1 + 0.1} 0\n")

        cls.ob_nexus_path = tmp_dir / "nexus" / "test_ob_metadata.nxs.h5"
        cls.ob_nexus_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(cls.ob_nexus_path, "w") as f:
            entry = f.create_group("entry")
            entry.create_dataset("proton_charge", data=[12345 * 2])
            entry.create_dataset("duration", data=[60.0])
            daslogs = entry.create_group("DASlogs")
            image_file_path = daslogs.create_group("BL10:Exp:IM:ImageFilePath")
            image_file_path.create_dataset("value", data=[[b"images/ob"]])
            time_offset_path = daslogs.create_group("BL10:Det:TH:DSPT1:TIDelay")
            time_offset_path.create_dataset("average_value", data=[5000])
            detector_path = daslogs.create_group("BL10:Exp:Det")
            detector_path.create_dataset("value_strings", data=[[b"MCP TPX1"]])

        # Also create a directory for the image file and a spectra TOF file
        image_dir = tmp_dir / "images" / "ob"
        image_dir.mkdir(exist_ok=True, parents=True)
        spectra_tof_file = image_dir / "test_Spectra.txt"
        with open(spectra_tof_file, "w") as f:
            for i in range(6):
                f.write(f"{i * 0.1 + 0.1} 0\n")

    @classmethod
    def teardown_class(cls):
        """Remove all temp test files after all tests in this class have run."""
        cls._tmpdir.cleanup()

    def test_venus_tpx1_pipeline_hdf5(self):
        """
        Test the VENUS TPX1 pipeline end-to-end with HDF5 output and verify contents.
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
            )

            assert output_path.exists()

            # Read back the file and check contents
            with h5py.File(output_path, "r") as hf:
                # Check transmission data
                assert "transmission" in hf
                assert hf["transmission"].shape == (5, 32, 32)
                # check that transmission values are correct based on the formula
                # T = S / AVERAGE(OB) * (proton_charge_ob / proton_charge_sample)
                # but will be two times because of the difference proton charge between sample and OB
                for i in range(5):
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i) / (99 + i) * 2)
                assert hf["transmission"].attrs["units"] == "dimensionless"
                assert hf["transmission"].dtype == np.float32
                # Check uncertainty data exists and is reasonable
                assert "uncertainty" in hf
                assert hf["uncertainty"].dtype == np.float32
                np.testing.assert_allclose(hf["uncertainty"], 0.245, rtol=0.1)
                # Check coordinates
                assert "x" in hf
                np.testing.assert_equal(hf["x"], np.arange(32))
                assert "y" in hf
                np.testing.assert_equal(hf["y"], np.arange(32))
                assert "tof" in hf
                np.testing.assert_allclose(hf["tof"], np.arange(0.1, 0.7, 0.1))
                assert "wavelength" in hf
                np.testing.assert_allclose(hf["wavelength"], np.array([16.6, 32.4, 48.3, 64.1, 79.9, 95.7]), atol=0.1)
                assert "energy" in hf
                np.testing.assert_allclose(
                    hf["energy"], np.array([0.296, 0.078, 0.035, 0.020, 0.013, 0.008]), atol=0.001
                )
                # Check masks
                assert "masks/dead" in hf
                np.testing.assert_equal(hf["masks/dead"], np.zeros((32, 32), dtype=bool))
                # Check metadata
                assert "metadata/sample_tiff_paths" in hf
                assert "metadata/ob_tiff_paths" in hf
                assert "metadata/sample_hdf5_paths" in hf
                assert "metadata/ob_hdf5_paths" in hf
                assert "metadata/processing_timestamp" in hf
                assert "metadata/version" in hf
                np.testing.assert_equal(hf["metadata/version"].asstr()[()], __version__)
                assert "metadata/roi_applied" not in hf  # ROI not applied in this test
                assert "proton_charge" in hf
                np.testing.assert_equal(hf["proton_charge"][()], 12345)
                assert "detector" in hf
                np.testing.assert_equal(hf["detector"].asstr()[()], "MCP TPX1")

    def test_venus_tpx1_pipeline_tiff(self):
        """
        Test the VENUS TPX1 pipeline end-to-end with TIFF output and ROI.
        """
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as f:
            output_path = Path(f.name)

            run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
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
        assert image.dims == ("t", "y", "x")
        assert image.values.shape == (5, 20, 20)
        # check that transmission values are correct based on the formula
        # T = (S) / AVERAGE(OB) * (proton_charge_ob / proton_charge_sample)
        for i in range(5):
            np.testing.assert_allclose(image.values[i], (81 + i) / (99 + i) * 2)
        # Check uncertainty data exists and is reasonable
        np.testing.assert_allclose(image.variances, 0.06, rtol=0.1)
        assert "scitiff-mask" in image.masks
        assert image.masks["scitiff-mask"].shape == (5, 20, 20)
        np.testing.assert_array_equal(image.masks["scitiff-mask"].values, False)

        # Check DAQ metadata
        daq = dg["daq"]  # this is type scitiff.DAQMetadata
        assert daq.facility == "SNS"
        assert daq.instrument == "VENUS"
        assert daq.detector_type == "MCP TPX1"
        assert daq.source_type == "neutron"

        # Check extra metadata
        extra = dg["extra"]
        assert len(extra["sample_tiff_paths"].values) == 5
        assert len(extra["ob_tiff_paths"].values) == 5
        assert len(extra["sample_hdf5_paths"].values) == 1
        assert len(extra["ob_hdf5_paths"].values) == 1

        assert "processing_timestamp" in extra
        np.testing.assert_equal(extra["roi_applied"].value, (5, 5, 25, 25))
        assert extra["version"] == __version__

    def test_venus_tpx1_pipeline_spatial_rebin(self):
        """
        Test the rebin_by_spatial function. Just look at the return DataArray and not the output file
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_spatial=8,
            )

            assert output_path.exists()

        assert transmission.shape == (5, 4, 4)  # original was (5, 32, 32) so 8x8 rebin should give (5, 4, 4)

        # values should be the same but variances should be reduced because of the rebinning
        for i in range(5):
            np.testing.assert_allclose(transmission.values[i], (81 + i) / (99 + i) * 2)

        np.testing.assert_allclose(transmission.variances, 0.001, rtol=0.1)

    def test_venus_tpx1_pipeline_tof_rebin(self):
        """
        Test the rebin_by_tof function. Just look at the return DataArray and not the output file
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_tof=2,
            )

            assert output_path.exists()

        assert transmission.shape == (
            3,
            32,
            32,
        )  # original was (5, 32, 32) so rebin by factor of 2 should give (3, 32, 32)

        np.testing.assert_allclose(transmission.values[0], (81 + 82) / (99 + 100) * 2)
        np.testing.assert_allclose(transmission.values[1], (83 + 84) / (101 + 102) * 2)
        np.testing.assert_allclose(transmission.values[2], (85) / 103 * 2)

        np.testing.assert_allclose(transmission.variances[0], 0.03, rtol=0.1)
        np.testing.assert_allclose(transmission.variances[1], 0.03, rtol=0.1)
        np.testing.assert_allclose(transmission.variances[2], 0.06, rtol=0.1)

        np.testing.assert_allclose(transmission.coords["tof"].values, [0.1, 0.3, 0.5, 0.6])
        np.testing.assert_allclose(transmission.coords["wavelength"].values, [16.6, 48.3, 79.9, 95.7], atol=0.1)
        np.testing.assert_allclose(transmission.coords["energy"].values, [0.296, 0.035, 0.013, 0.008], atol=0.001)

    def test_venus_tpx1_pipeline_tof_rebin_auto(self):
        """
        Test the rebin_by_tof function with analyze statistics to get recommended rebinning factor.
        Just look at the return DataArray and not the output file
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_tof=True,
            )

            assert output_path.exists()

        assert transmission.shape == (
            5,
            32,
            32,
        )  # should be unchanged because the recommended rebinning factor based on the test TOF data is 1 (no rebinning)

        # values and variances should be the same
        for i in range(5):
            np.testing.assert_allclose(transmission.values[i], (81 + i) / (99 + i) * 2)

        np.testing.assert_allclose(transmission.variances, 0.06, rtol=0.1)

        np.testing.assert_allclose(transmission.coords["tof"].values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_venus_tpx1_pipeline_air_region_correction(self):
        """
        Test the air region correction function. Just look at the return DataArray and not the output file
        """
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[self.sample_nexus_path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                air_roi=(0, 0, 10, 10),
            )

            assert output_path.exists()

        assert transmission.shape == (5, 32, 32)
        # Since all the data are the same for a single tof the air correction should just normalize 1.
        np.testing.assert_allclose(transmission.values, 1)
        np.testing.assert_allclose(transmission.variances, 0.0227, rtol=0.1)

    def test_venus_tpx1_pipeline_invalid_paths(self):
        """Check error when the length of tiff and hdf5 paths do not match."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            with pytest.raises(
                ValueError,
                match=r"Number of sample HDF5 paths \(0\) does not match number of sample TIFF path groups \(1\).",
            ):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[self.sample_tiff_paths],
                    ob_tiff_paths=[self.ob_tiff_paths],
                    sample_hdf5_paths=[],  # empty list should trigger error
                    ob_hdf5_paths=[self.ob_nexus_path],
                    output_path=output_path,
                )

            with pytest.raises(
                ValueError, match=r"Number of OB HDF5 paths \(0\) does not match number of OB TIFF path groups \(1\)."
            ):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[self.sample_tiff_paths],
                    ob_tiff_paths=[self.ob_tiff_paths],
                    sample_hdf5_paths=[self.sample_nexus_path],
                    ob_hdf5_paths=[],  # empty list should trigger error
                    output_path=output_path,
                )

    def test_venus_tpx1_pipeline_invalid_rebin_by_tof(self):
        """Check error for invalid rebin_by_tof values."""
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            with pytest.raises(ValueError, match=r"Invalid value for rebin_by_tof: invalid. Must be bool or int."):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[self.sample_tiff_paths],
                    ob_tiff_paths=[self.ob_tiff_paths],
                    sample_hdf5_paths=[self.sample_nexus_path],
                    ob_hdf5_paths=[self.ob_nexus_path],
                    output_path=output_path,
                    rebin_by_tof="invalid",  # invalid value should trigger error
                )

    def test_venus_tpx1_pipeline_invalid_output_format(self):
        """Check error for unsupported output file format."""
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=True) as f:
            output_path = Path(f.name)

            with pytest.raises(ValueError, match=r"Unsupported output file format: .bmp"):
                run_venus_tpx1_pipeline(
                    sample_tiff_paths=[self.sample_tiff_paths],
                    ob_tiff_paths=[self.ob_tiff_paths],
                    sample_hdf5_paths=[self.sample_nexus_path],
                    ob_hdf5_paths=[self.ob_nexus_path],
                    output_path=output_path,
                )

    def test_venus_tpx1_pipeline_missing_detector_time_offset(self):
        """Check warning when detector_time_offset is missing."""

        path = Path(self._tmpdir.name) / "nexus" / "test_missing_detector_time_offset.nxs.h5"
        with h5py.File(path, "w") as f:
            entry = f.create_group("entry")
            entry.create_dataset("proton_charge", data=[12345])
            entry.create_dataset("duration", data=[60.0])
            daslogs = entry.create_group("DASlogs")
            image_file_path = daslogs.create_group("BL10:Exp:IM:ImageFilePath")
            image_file_path.create_dataset("value", data=[[b"images/sample"]])
            detector_path = daslogs.create_group("BL10:Exp:Det")
            detector_path.create_dataset("value_strings", data=[[b"MCP TPX1"]])

        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
            output_path = Path(f.name)

            transmission = run_venus_tpx1_pipeline(
                sample_tiff_paths=[self.sample_tiff_paths],
                ob_tiff_paths=[self.ob_tiff_paths],
                sample_hdf5_paths=[path],
                ob_hdf5_paths=[self.ob_nexus_path],
                output_path=output_path,
                rebin_by_tof=2,
            )

            assert output_path.exists()

        assert "tof" in transmission.coords
        assert "wavelength" not in transmission.coords
        assert "energy" not in transmission.coords
