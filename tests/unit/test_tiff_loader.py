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

    assert len(da.coords) == 15
    assert "1" in da.coords
    assert len(da.coords["1"]) == 3
    assert da.coords["1"].values[0] == "this is metadata of image001.tif"


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
