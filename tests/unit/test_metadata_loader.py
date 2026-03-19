import tempfile
from pathlib import Path

import h5py

from neunorm.loaders.metadata_loader import load_metadata


class TestMetadataLoader:
    """Tests for the loading metadata from VENUS NeXus files."""

    @classmethod
    def setup_class(cls):
        """Create temporary HDF5 file with minimal metadata for testing."""
        cls._tmpdir = tempfile.TemporaryDirectory(delete=False)
        cls.nexus_path = Path(cls._tmpdir.name) / "nexus" / "test_metadata.nxs.h5"
        cls.nexus_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(cls.nexus_path, "w") as f:
            entry = f.create_group("entry")
            entry.create_dataset("proton_charge", data=[1.23e-6])
            entry.create_dataset("duration", data=[60.0])
            daslogs = entry.create_group("DASlogs")
            image_file_path = daslogs.create_group("BL10:Exp:IM:ImageFilePath")
            image_file_path.create_dataset("value", data=[[b"images"]])

        # Also create a directory for the image file and a shutter count file
        image_dir = Path(cls._tmpdir.name) / "images"
        image_dir.mkdir(exist_ok=True)
        shutter_count_file = image_dir / "test_ShutterCount.txt"
        with open(shutter_count_file, "w") as f:
            for i in range(5):
                f.write(f"{i}\t{i * 1000 + 1000}\n")
            for i in range(5, 10):
                f.write(f"{i}\t0\n")

    @classmethod
    def teardown_class(cls):
        """Remove all temp test files after all tests in this class have run."""
        cls._tmpdir.cleanup()

    def test_load_metadata(self):
        """
        Test that load_metadata correctly loads metadata from the NeXus file
        """
        metadata = load_metadata(self.nexus_path)

        assert "proton_charge" in metadata
        assert "duration" in metadata
        assert "image_file_path" in metadata
        assert "shutter_counts" not in metadata

        assert metadata["proton_charge"] == 1.23e-6
        assert metadata["duration"] == 60.0
        assert metadata["image_file_path"] == "images"

    def test_load_metadata_with_shutter_counts(self):
        """
        Test that load_metadata correctly loads metadata from the NeXus file, including shutter counts.
        """
        metadata = load_metadata(self.nexus_path, read_shutter_counts=True)

        assert "proton_charge" in metadata
        assert "duration" in metadata
        assert "image_file_path" in metadata
        assert "shutter_counts" in metadata

        assert metadata["proton_charge"] == 1.23e-6
        assert metadata["duration"] == 60.0
        assert metadata["image_file_path"] == "images"
        assert metadata["shutter_counts"] == [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
