import tempfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def generate_test_data():
    """Create tiff files for testing"""
    sample_paths = []
    ob_paths = []
    dark_paths = []

    # create 5 sample tiffs with values 81-85 and metadata
    sample_temp_name = tempfile.NamedTemporaryFile(prefix="sample_", suffix="_{:03}.tiff")
    for i in range(5):
        data = np.full((32, 32), 81 + i, dtype=np.float32)
        img = Image.fromarray(data)
        exif = img.getexif()
        exif[65027] = "ExposureTime:30.000000"
        exif[65022] = f"RunNo:{1000 + i}"
        filename = sample_temp_name.name.format(i)
        img.save(filename, exif=exif)
        sample_paths.append(filename)

    # create 3 OB tiffs with values 99,100,101 and metadata
    ob_temp_name = tempfile.NamedTemporaryFile(prefix="ob_", suffix="_{:03}.tiff")
    for i in range(3):
        data = np.full((32, 32), 99 + i, dtype=np.float32)
        img = Image.fromarray(data)
        filename = ob_temp_name.name.format(i)
        img.save(filename, exif=exif)
        ob_paths.append(filename)

    # create 2 dark tiffs with values 4 and 6 and metadata
    dark_temp_name = tempfile.NamedTemporaryFile(prefix="dark_", suffix="_{:03}.tiff")
    for i in range(2):
        data = np.full((32, 32), 4 + 2 * i, dtype=np.float32)
        img = Image.fromarray(data)
        filename = dark_temp_name.name.format(i)
        img.save(filename, exif=exif)
        dark_paths.append(filename)

    return sample_paths, ob_paths, dark_paths


def test_mars_ccd_pipeline():
    from neunorm import __version__
    from neunorm.pipelines.mars_ccd import run_mars_ccd_pipeline

    sample_paths, ob_paths, dark_paths = generate_test_data()

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        run_mars_ccd_pipeline(
            sample_paths=sample_paths, ob_paths=ob_paths, dark_paths=dark_paths, output_path=Path(f.name)
        )

        assert Path(f.name).exists()

        # Read back the file and check contents
        with h5py.File(f.name, "r") as f:
            # Check transmission data
            assert "transmission" in f
            assert f["transmission"].shape == (5, 32, 32)
            # check that transmission values are correct based on the formula
            # T = (S - AVERAGE(D)) / (AVERAGE(OB) - AVERAGE(D))
            for i in range(5):
                np.testing.assert_allclose(f["transmission"][i], (81 + i - 5) / (100 - 5))
            assert f["transmission"].attrs["units"] == "dimensionless"
            assert f["transmission"].dtype == np.float32
            # Check uncertainty data exists and is reasonable
            assert "uncertainty" in f
            assert f["uncertainty"].dtype == np.float32
            np.testing.assert_allclose(f["uncertainty"], 0.1, rtol=0.15)
            # Check coordinates
            assert "x" in f
            np.testing.assert_equal(f["x"], np.arange(32))
            assert "y" in f
            np.testing.assert_equal(f["y"], np.arange(32))
            # Check masks
            assert "masks/dead" in f
            np.testing.assert_equal(f["masks/dead"], np.zeros((32, 32), dtype=bool))
            # Check metadata
            assert "metadata/sample_paths" in f
            np.testing.assert_equal(f["metadata/sample_paths"].asstr()[:], sample_paths)
            assert "metadata/ob_paths" in f
            np.testing.assert_equal(f["metadata/ob_paths"].asstr()[:], ob_paths)
            assert "metadata/dark_paths" in f
            np.testing.assert_equal(f["metadata/dark_paths"].asstr()[:], dark_paths)
            assert "metadata/gamma_filter_applied" in f
            np.testing.assert_equal(f["metadata/gamma_filter_applied"][()], True)
            assert "metadata/processing_timestamp" in f
            assert "metadata/version" in f
            np.testing.assert_equal(f["metadata/version"].asstr()[()], __version__)
            assert "metadata/roi_applied" not in f  # ROI not applied in this test
            # check metadata from files, RunNo and ExposureTime
            assert "RunNo" in f
            np.testing.assert_equal(f["RunNo"][:], [1000, 1001, 1002, 1003, 1004])
            assert "ExposureTime" in f
            np.testing.assert_equal(f["ExposureTime"][:], [30] * 5)

    # cleanup temp test files
    for path in sample_paths + ob_paths + dark_paths:
        Path(path).unlink()
