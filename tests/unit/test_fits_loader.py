"""
Unit tests for the FITS data loader.

These tests verify loading FITS image stacks and single FITS files,
including variants with time-of-flight (TOF) binning.
"""

from pathlib import Path

import numpy as np


def test_load_fits_stack():
    """Test loading FITS stack"""
    from neunorm.loaders.fits_loader import load_fits_stack

    # Load FITS stack
    fits_dir = Path(__file__).parent.parent / "data" / "fits" / "sample"
    paths = sorted(fits_dir.glob("image00*.fits"))

    da = load_fits_stack(paths)

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

    assert len(da.coords) == 8
    assert "SIMPLE" in da.coords


def test_load_fits_stack_tof_edges():
    """Test loading FITS stack with TOF edges"""
    from neunorm.loaders.fits_loader import load_fits_stack

    # Load FITS stack
    fits_dir = Path(__file__).parent.parent / "data" / "fits" / "sample"
    paths = sorted(fits_dir.glob("image00*.fits"))

    da = load_fits_stack(paths, tof_edges=np.linspace(1000, 2500, num=4))

    # Check dimensions and coordinates
    assert da.dims == ("TOF", "y", "x")
    assert "y" in da.coords
    assert "x" in da.coords
    assert "TOF" in da.coords
    assert da.coords["y"].values.shape == (5,)
    assert da.coords["x"].values.shape == (5,)
    assert da.coords["TOF"].values.shape == (4,)
    np.testing.assert_equal(da.coords["TOF"].values, (1000, 1500, 2000, 2500))

    assert da.data.shape == (3, 5, 5)
    assert da.values.max() == 5
    assert da.variances.shape == (3, 5, 5)
    assert da.variances.max() == 5


def test_load_fits_stack_tof_centers():
    """Test loading FITS stack with TOF centers"""
    from neunorm.loaders.fits_loader import load_fits_stack

    fits_dir = Path(__file__).parent.parent / "data" / "fits" / "sample"
    paths = sorted(fits_dir.glob("image00*.fits"))

    da = load_fits_stack(paths, tof_edges=np.array([1000, 1500, 2000]))

    assert da.dims == ("TOF", "y", "x")
    assert "TOF" in da.coords
    assert da.coords["TOF"].values.shape == (3,)
    np.testing.assert_equal(da.coords["TOF"].values, (1000, 1500, 2000))


def test_load_single_fit():
    """Test loading a single FITS file"""
    from neunorm.loaders.fits_loader import load_fits_stack

    # Load FITS stack. Test with str instead of Path to check both types are accepted.
    path = str(Path(__file__).parent.parent / "data" / "fits" / "sample" / "image001.fits")

    da = load_fits_stack([path])

    # Check dimensions and coordinates
    assert da.dims == ("N_image", "y", "x")
    assert "y" in da.coords
    assert "x" in da.coords
    assert da.coords["y"].values.shape == (5,)
    assert da.coords["x"].values.shape == (5,)

    assert da.data.shape == (1, 5, 5)
    assert da.values.max() == 5
    assert da.variances.shape == (1, 5, 5)
    assert da.variances.max() == 5
