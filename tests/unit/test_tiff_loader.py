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


def test_load_tiff_stack_casts_integer_files_to_float32(tmp_path):
    """An integer TIFF still loads as float32.

    The checked-in fixtures are all float32 (SampleFormat 3), which is what the VENUS TPX1
    auto-reduction writes, so they cannot catch a lost cast. A CCD writes integer counts, and
    tifffile returns the file's own dtype rather than Pillow's requested one.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    for i in range(2):
        tifffile.imwrite(tmp_path / f"int{i:03d}.tif", np.full((4, 6), 7 + i, dtype=np.uint16))

    da = load_tiff_stack(sorted(tmp_path.glob("*.tif")))

    assert da.values.dtype == np.float32
    assert da.variances.dtype == np.float32
    np.testing.assert_allclose(da.values[0], 7.0)
    np.testing.assert_allclose(da.values[1], 8.0)


def test_load_tiff_stack_reads_only_the_first_page(tmp_path):
    """A multi-page TIFF contributes one 2-D frame, not its whole page stack.

    ``PIL.Image.open`` yields page 0; ``tifffile.imread`` would return every page stacked and
    silently turn each 2-D frame into 3-D, changing the shape of the result.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    stack = np.stack([np.full((4, 6), v, dtype=np.float32) for v in (1.0, 2.0, 3.0)])
    # photometric="minisblack" is required: without it tifffile stores a (3, y, x) float32
    # array as one RGB page with three component planes rather than three pages, so the
    # fixture would not exercise the multi-page path at all.
    tifffile.imwrite(tmp_path / "multipage.tif", stack, photometric="minisblack")

    da = load_tiff_stack([tmp_path / "multipage.tif"])

    assert da.data.shape == (1, 4, 6)
    np.testing.assert_allclose(da.values[0], 1.0)


def test_load_tiff_stack_sample_format_stays_a_scalar_coordinate():
    """``SampleFormat`` is one value for the stack, not one per frame.

    Pillow reports it as the tuple ``(3,)``, which fails ``float()`` and lands in the scalar
    branch of the metadata block. tifffile reports the ``IntEnum`` ``SAMPLEFORMAT.IEEEFP``,
    which converts under ``float()`` and would become a per-frame array instead — a silent
    change to a published coordinate. This pins the Pillow shape.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    tiff_dir = Path(__file__).parent.parent / "data" / "tif" / "sample"
    da = load_tiff_stack(sorted(tiff_dir.glob("*.tif")))

    assert da.coords["SampleFormat"].dims == ()
    assert len(da.coords["SampleFormat"].values) == 1
    assert da.coords["SampleFormat"].values[0] == 3
