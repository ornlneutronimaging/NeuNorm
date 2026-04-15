import tempfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

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

        # create 3 OB tiffs with values 99, 100, 101 and metadata
        for i in range(3):
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

        # Also create a directory for the image file and a spectra TOF file
        image_dir = tmp_dir / "images" / "ob"
        image_dir.mkdir(exist_ok=True, parents=True)
        spectra_tof_file = image_dir / "test_Spectra.txt"
        with open(spectra_tof_file, "w") as f:
            for i in range(4):
                f.write(f"{i * 0.1 + 0.1} 0\n")

    @classmethod
    def teardown_class(cls):
        """Remove all temp test files after all tests in this class have run."""
        # cls._tmpdir.cleanup()

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
                    np.testing.assert_allclose(hf["transmission"][i], (81 + i) / (100) * 2)
                assert hf["transmission"].attrs["units"] == "dimensionless"
                assert hf["transmission"].dtype == np.float32
                # Check uncertainty data exists and is reasonable
                assert "uncertainty" in hf
                assert hf["uncertainty"].dtype == np.float32
                np.testing.assert_allclose(hf["uncertainty"], 0.2, rtol=0.1)
                # Check coordinates
                assert "x" in hf
                np.testing.assert_equal(hf["x"], np.arange(32))
                assert "y" in hf
                np.testing.assert_equal(hf["y"], np.arange(32))
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
