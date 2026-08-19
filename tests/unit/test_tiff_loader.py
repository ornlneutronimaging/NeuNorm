"""
Unit tests for the TIFF data loader.

These tests verify loading TIFF image stacks, including variants with time-of-flight (TOF) binning.
"""

from pathlib import Path

import numpy as np


def test_load_tiff_stack():
    """Test loading TIFF stack"""
    from neunorm.loaders.tiff_loader import load_tiff_stack

    # Load TIFF stack
    tiff_dir = Path(__file__).parent.parent / "data" / "tif" / "sample"
    paths = sorted(tiff_dir.glob("*.tif"))

    da = load_tiff_stack(paths)

    # Check dimensions and coordinates
    assert da.dims == ("N_image", "y", "x")
    assert "y" in da.coords
    assert "x" in da.coords
    assert da.coords["y"].values.shape == (5,)
    assert da.coords["x"].values.shape == (5,)

    assert da.data.shape == (3, 5, 5)
    assert da.values.max() == 5
    assert da.variances.shape == (3, 5, 5)
    assert da.variances.max() == 5

    # float32 is sufficient for neutron imaging; loading in float32 halves memory
    assert da.values.dtype == np.float32
    assert da.variances.dtype == np.float32

    assert len(da.coords) == 15

    assert "SampleFormat" in da.coords
    assert len(da.coords["SampleFormat"].values) == 1
    assert da.coords["SampleFormat"].values[0] == 3

    assert "InteropIndex" in da.coords
    assert len(da.coords["InteropIndex"]) == 3
    assert da.coords["InteropIndex"].values[0] == "this is metadata of image001.tif"


def test_load_tiff_stack_tof_edges():
    """Test loading TIFF stack with TOF edges"""
    from neunorm.loaders.tiff_loader import load_tiff_stack

    # Load TIFF stack
    tiff_dir = Path(__file__).parent.parent / "data" / "tif" / "sample"
    paths = sorted(tiff_dir.glob("*.tif"))

    da = load_tiff_stack(paths, tof_edges=np.linspace(1000, 2500, num=4))

    # Check dimensions and coordinates
    assert da.dims == ("TOF", "y", "x")
    assert "y" in da.coords
    assert "x" in da.coords
    assert "TOF" in da.coords
    assert da.coords["y"].values.shape == (5,)
    assert da.coords["x"].values.shape == (5,)
    assert da.coords.is_edges("TOF")
    assert da.coords["TOF"].values.shape == (4,)
    np.testing.assert_equal(da.coords["TOF"].values, (1000, 1500, 2000, 2500))

    assert da.data.shape == (3, 5, 5)
    assert da.values.max() == 5
    assert da.variances.shape == (3, 5, 5)
    assert da.variances.max() == 5


def test_load_tiff_stack_tof_centers():
    """Test loading TIFF stack with TOF centers"""
    from neunorm.loaders.tiff_loader import load_tiff_stack

    tiff_dir = Path(__file__).parent.parent / "data" / "tif" / "sample"
    paths = sorted(tiff_dir.glob("*.tif"))

    da = load_tiff_stack(paths, tof_edges=np.array([1000, 1500, 2000]))

    assert da.dims == ("TOF", "y", "x")
    assert "TOF" in da.coords
    assert not da.coords.is_edges("TOF")
    assert da.coords["TOF"].values.shape == (3,)
    np.testing.assert_equal(da.coords["TOF"].values, (1000, 1500, 2000))


def test_load_tiff_stack_negative_pixels_zeroed(tmp_path):
    """Detector-glitch negative pixels are zeroed with a warning, not fatal.

    Seen on VENUS run 28787 (IPTS-38504): a glitching Timepix chip wrote
    wrapped values around ±32k into a handful of autoreduced float32 frames.
    """
    import io

    from loguru import logger
    from PIL import Image

    from neunorm.loaders.tiff_loader import load_tiff_stack

    clean = np.full((5, 5), 7.0, dtype=np.float32)
    glitched = clean.copy()
    glitched[1, 2] = -32565.0
    glitched[3, 4] = -50474.6
    for i, frame in enumerate([clean, glitched, clean]):
        Image.fromarray(frame).save(tmp_path / f"image{i:03d}.tif")

    captured = io.StringIO()
    sink_id = logger.add(captured, level="WARNING", format="{message}")
    try:
        da = load_tiff_stack(sorted(tmp_path.glob("*.tif")))
    finally:
        logger.remove(sink_id)

    warning_text = captured.getvalue()
    assert "2 negative pixel" in warning_text
    assert "across 1 of 3 frame" in warning_text
    assert "most negative value: -50474.6" in warning_text

    assert da.values.min() == 0.0
    assert da.values[1, 1, 2] == 0.0
    assert da.values[1, 3, 4] == 0.0
    # variance = counts stays valid and untouched pixels are preserved
    assert da.variances.min() == 0.0
    assert da.values[0].min() == 7.0
    assert da.values[1, 0, 0] == 7.0
