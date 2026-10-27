import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipp as sc

from neunorm.loaders.metadata_loader import (
    find_nexus_file,
    load_metadata,
    load_proton_charge_coulombs,
    run_number_in_name,
)


class TestMetadataLoader:
    """Tests for the loading metadata from VENUS NeXus files."""

    @classmethod
    def setup_class(cls):
        """Create temporary HDF5 file with minimal metadata for testing."""
        cls._tmpdir = tempfile.TemporaryDirectory()
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
        # image_file_path now reflects the real read directory (resolved), not the DAS-log value (#187)
        assert sc.identical(metadata["image_file_path"], sc.scalar(str(real_dir.resolve())))

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


class TestNexusLookup:
    """Locating the NeXus file of a VENUS image from the run number in its name."""

    def test_run_number_in_name(self):
        assert run_number_in_name("20260928_Run_31567_reptVert_PFV231_300_000s_Ang_000_000_1.tiff") == 31567
        assert run_number_in_name("20260814_Run_27075_Moon_0.tiff") == 27075
        assert run_number_in_name("x_Run_12.tiff") == 12
        assert run_number_in_name("x_Run_12") == 12
        assert run_number_in_name("x_Run_abc_Run_7_y.tiff") == 7
        assert run_number_in_name("dark_current_temporary.tiff") is None
        assert run_number_in_name("x_Run_12a_y.tiff") is None

    def test_find_nexus_file(self, tmp_path):
        ipts = tmp_path / "SNS" / "VENUS" / "IPTS-38722"
        image = ipts / "images" / "qhy411" / "raw" / "radiography" / "20260814_Run_27075_Moon_0.tiff"
        image.parent.mkdir(parents=True)
        image.touch()
        nexus = ipts / "nexus" / "VENUS_27075.nxs.h5"
        # No nexus/ folder yet.
        assert find_nexus_file(image) is None
        nexus.parent.mkdir()
        assert find_nexus_file(image) is None
        nexus.touch()
        assert find_nexus_file(image) == nexus
        # Another run of the same experiment has no file.
        other = image.with_name("20260814_Run_27076_Moon_0.tiff")
        other.touch()
        assert find_nexus_file(other) is None
        # No run number in the name.
        assert find_nexus_file(image.with_name("dark_current_temporary.tiff")) is None
        # Outside an IPTS tree: nothing without nexus_dir, found with it.
        copy = tmp_path / "copies" / image.name
        copy.parent.mkdir()
        copy.touch()
        assert find_nexus_file(copy) is None
        assert find_nexus_file(copy, nexus_dir=nexus.parent) == nexus
        assert find_nexus_file(copy, nexus_dir=tmp_path / "nowhere") is None

    def test_load_proton_charge_coulombs(self, tmp_path):
        path = tmp_path / "VENUS_1.nxs.h5"
        with h5py.File(path, "w") as f:
            entry = f.create_group("entry")
            entry.create_dataset("proton_charge", data=[4.41537944e11])  # pC
            entry.create_dataset("duration", data=[338.0])
        np.testing.assert_allclose(load_proton_charge_coulombs(path), 0.441537944)
