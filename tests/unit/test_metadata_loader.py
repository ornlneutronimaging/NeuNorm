import tempfile
from pathlib import Path

import h5py
import scipp as sc

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

        assert isinstance(metadata["proton_charge"], sc.Variable)
        assert isinstance(metadata["duration"], sc.Variable)
        assert isinstance(metadata["image_file_path"], sc.Variable)
        assert sc.identical(metadata["proton_charge"], sc.scalar(1.23e-6, unit="pC"))
        assert sc.identical(metadata["duration"], sc.scalar(60.0, unit="s"))
        assert sc.identical(metadata["image_file_path"], sc.scalar("images"))

    def test_load_metadata_with_shutter_counts(self):
        """
        Test that load_metadata correctly loads metadata from the NeXus file, including shutter counts.
        """
        metadata = load_metadata(self.nexus_path, read_shutter_counts=True)

        assert "proton_charge" in metadata
        assert "duration" in metadata
        assert "image_file_path" in metadata
        assert "shutter_counts" in metadata

        assert isinstance(metadata["shutter_counts"], sc.Variable)
        assert sc.identical(metadata["proton_charge"], sc.scalar(1.23e-6, unit="pC"))
        assert sc.identical(metadata["duration"], sc.scalar(60.0, unit="s"))
        assert sc.identical(metadata["image_file_path"], sc.scalar("images"))
        assert sc.identical(
            metadata["shutter_counts"], sc.array(dims=["N_image"], values=[1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        )

    def test_load_metadata_image_dir_overrides_daslog_for_spectra(self):
        """#187: when image_dir is given, spectra TOF is read from THERE, not the DAS-log raw path.

        A decoy ``*_Spectra.txt`` with different values sits at the DAS-log-resolved directory; the
        loader must ignore it and read the file co-located with the caller's images instead.
        """
        tmp = Path(self._tmpdir.name)
        # decoy spectra at the DAS-log-resolved dir (tmp/images) — must NOT be read
        with open(tmp / "images" / "decoy_Spectra.txt", "w") as f:
            for i in range(3):
                f.write(f"{9.0 + i} 0\n")
        # the real image directory the caller passes, with the correct spectra
        real_dir = tmp / "autoreduce_spectra"
        real_dir.mkdir(exist_ok=True)
        with open(real_dir / "real_Spectra.txt", "w") as f:
            for v in (0.1, 0.2, 0.3, 0.4):
                f.write(f"{v} 0\n")

        metadata = load_metadata(self.nexus_path, read_spectra_tof=True, image_dir=real_dir)

        assert "spectra_tof" in metadata
        assert sc.identical(metadata["spectra_tof"], sc.array(dims=["N_image"], values=[0.1, 0.2, 0.3, 0.4], unit="s"))

    def test_load_metadata_image_dir_overrides_daslog_for_shutter_counts(self):
        """#187: shutter counts are read from image_dir too (same resolution path as spectra)."""
        tmp = Path(self._tmpdir.name)
        real_dir = tmp / "autoreduce_shutter"
        real_dir.mkdir(exist_ok=True)
        with open(real_dir / "real_ShutterCount.txt", "w") as f:
            f.write("0\t111\n1\t222\n2\t0\n")  # loader stops at the first 0 count

        metadata = load_metadata(self.nexus_path, read_shutter_counts=True, image_dir=real_dir)

        # values come from real_dir, not the tmp/images shutter file used by the other tests
        assert sc.identical(metadata["shutter_counts"], sc.array(dims=["N_image"], values=[111.0, 222.0]))
