"""
Unit tests for the FITS data loader.

These tests verify loading FITS image stacks and single FITS files,
including variants with time-of-flight (TOF) binning.
"""

from pathlib import Path

import numpy as np
import pytest


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

    # float32 is sufficient for neutron imaging; loading in float32 halves memory
    assert da.values.dtype == np.float32
    assert da.variances.dtype == np.float32

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
    assert da.coords.is_edges("TOF")
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
    assert not da.coords.is_edges("TOF")
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


def test_load_fits_stack_casts_integer_files_to_float32(tmp_path):
    """An integer FITS still loads as float32.

    The checked-in fixtures are float, so they cannot catch a lost cast. A CCD writes integer
    counts, and FITS is big-endian on disk: the cast is what makes the result a native-order
    float32 rather than a byte-swapped view of the file.
    """
    from astropy.io import fits

    from neunorm.loaders.fits_loader import load_fits_stack

    for i in range(2):
        fits.PrimaryHDU(data=np.full((4, 6), 7 + i, dtype=">i2")).writeto(tmp_path / f"int{i:03d}.fits")

    da = load_fits_stack(sorted(tmp_path.glob("*.fits")))

    assert da.values.dtype == np.float32
    assert da.variances.dtype == np.float32
    np.testing.assert_allclose(da.values[0], 7.0)
    np.testing.assert_allclose(da.values[1], 8.0)


def _write_ramp(directory, n=17, ny=3, nx=4):
    """n frames whose every pixel equals the frame's own input index."""
    from astropy.io import fits

    paths = []
    for i in range(n):
        p = directory / f"frame{i:03d}.fits"
        fits.PrimaryHDU(data=np.full((ny, nx), float(i), dtype=np.float32)).writeto(p)
        paths.append(p)
    return paths


def test_parallel_decode_preserves_frame_order(tmp_path):
    """Frame i must land at index i, whatever order the decodes finish in.

    The stack's first dimension becomes the TOF/N_image axis and the tof coordinate is matched to
    it positionally, so a frame at the wrong index mislabels the time axis — the result still looks
    like a plausible spectrum. Each frame is filled with its own index so a permutation is visible
    rather than merely suspected.
    """
    from neunorm.loaders.fits_loader import load_fits_stack

    paths = _write_ramp(tmp_path)

    da = load_fits_stack(paths, max_workers=8)

    np.testing.assert_allclose(da.values[:, 0, 0], np.arange(len(paths), dtype=np.float32))
    for i in range(len(paths)):
        np.testing.assert_allclose(da.values[i], float(i))


def test_parallel_decode_keeps_each_header_with_its_own_frame(tmp_path):
    """A per-frame header value must stay aligned with the pixels it came from.

    The headers are collected off the same futures as the pixels, so a header stored by completion
    order rather than by input index would attach the wrong exposure time to every frame — and the
    values would still all be present, so a test that only checked the set would pass. Frame i
    carries both pixel value i and ``FRAMEIDX`` i, which pins the two to each other.
    """
    from astropy.io import fits

    from neunorm.loaders.fits_loader import load_fits_stack

    paths = []
    for i in range(17):
        hdu = fits.PrimaryHDU(data=np.full((3, 4), float(i), dtype=np.float32))
        hdu.header["FRAMEIDX"] = i
        p = tmp_path / f"frame{i:03d}.fits"
        hdu.writeto(p)
        paths.append(p)

    da = load_fits_stack(paths, max_workers=8)

    np.testing.assert_allclose(da.coords["FRAMEIDX"].values, np.arange(len(paths)))
    np.testing.assert_allclose(da.values[:, 0, 0], np.arange(len(paths), dtype=np.float32))


def test_parallel_and_serial_decode_agree(tmp_path):
    """max_workers=1 and a real pool must produce identical arrays, variances and coordinates.

    This guards against divergence that appears only with a pool — a race, a dropped frame, a
    worker-only code path. It does **not** establish that the order is right: both calls run the
    same placement code, so a bug that misplaces every frame identically leaves the two agreeing.
    The ordering guarantee lives in test_parallel_decode_preserves_frame_order.
    """
    from neunorm.loaders.fits_loader import load_fits_stack

    paths = _write_ramp(tmp_path)

    import scipp as sc

    serial = load_fits_stack(paths, max_workers=1)
    parallel = load_fits_stack(paths, max_workers=8)

    # sc.identical rather than a coordinate-name comparison: names matching says nothing about
    # coordinate values, dims or alignment, and those are published output too.
    assert sc.identical(serial, parallel)


def test_parallel_decode_raises_on_shape_mismatch(tmp_path):
    """An inconsistent frame still raises, and names the offending file."""
    from astropy.io import fits

    from neunorm.loaders.fits_loader import load_fits_stack

    paths = _write_ramp(tmp_path, n=6)
    odd = tmp_path / "frame003.fits"
    odd.unlink()
    fits.PrimaryHDU(data=np.zeros((5, 9), dtype=np.float32)).writeto(odd)

    with pytest.raises(ValueError, match="Shape mismatch"):
        load_fits_stack(paths, max_workers=4)


def test_parallel_decode_propagates_a_failed_read(tmp_path):
    """A missing file raises rather than hanging the pool or yielding a partial stack."""
    from neunorm.loaders.fits_loader import load_fits_stack

    paths = _write_ramp(tmp_path, n=6)
    paths[4] = tmp_path / "does-not-exist.fits"

    with pytest.raises(Exception):  # noqa: B017 - astropy's own error type is not part of the contract
        load_fits_stack(paths, max_workers=4)


def test_max_workers_rejects_values_that_are_not_a_worker_count(tmp_path):
    """0, a negative, a float, a bool or a string are mistakes, not settings.

    A float is the one that matters: ThreadPoolExecutor compares its live thread count against the
    value rather than truncating it, so ``max_workers=1.5`` builds two threads and ``2.9`` three --
    silently exceeding the cap, which is the only thing this parameter does. ``True`` is an int
    subclass and would otherwise pass as 1.

    numpy integers are accepted, because a caller sizing the pool from an array shape produces one.
    Same rule and same error shapes as ``_check_advance`` in utils/progress.py.
    """
    from neunorm.loaders.fits_loader import load_fits_stack

    paths = _write_ramp(tmp_path, n=3)

    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            load_fits_stack(paths, max_workers=bad)

    for bad in (1.5, 2.9, 1.0, True, "4"):
        with pytest.raises(TypeError, match="max_workers must be an int"):
            load_fits_stack(paths, max_workers=bad)

    # accepted, and equivalent to the plain int
    import scipp as sc

    assert sc.identical(load_fits_stack(paths, max_workers=np.int64(2)), load_fits_stack(paths, max_workers=2))


def test_a_set_of_paths_still_loads(tmp_path):
    """A sized-but-unindexable collection must still work: the decode addresses frames by index.

    The pre-parallel loader only iterated `paths`, so a set worked. `_decode_stack` subscripts it,
    and a guard testing only for `__len__` let a set through to a TypeError.
    """
    from neunorm.loaders.fits_loader import load_fits_stack

    paths = _write_ramp(tmp_path, n=4)

    da = load_fits_stack(set(paths), max_workers=2)

    assert da.data.shape == (4, 3, 4)
    # A set has no order, so only the multiset of frame values is defined here.
    assert sorted(da.values[:, 0, 0]) == [0.0, 1.0, 2.0, 3.0]
