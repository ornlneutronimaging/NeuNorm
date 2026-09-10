"""
Unit tests for the TIFF data loader.

These tests verify loading TIFF image stacks, including variants with time-of-flight (TOF) binning.
"""

from pathlib import Path

import numpy as np
import pytest


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


def _write_ramp(directory, n=17, ny=3, nx=4):
    """n frames whose every pixel equals the frame's own input index."""
    import tifffile

    paths = []
    for i in range(n):
        p = directory / f"frame{i:03d}.tif"
        tifffile.imwrite(p, np.full((ny, nx), float(i), dtype=np.float32))
        paths.append(p)
    return paths


def test_parallel_decode_preserves_frame_order(tmp_path):
    """Frame i must land at index i, whatever order the decodes finish in.

    The stack's first dimension becomes the TOF/N_image axis and the tof coordinate is matched to
    it positionally, so a frame at the wrong index mislabels the time axis — the result still looks
    like a plausible spectrum. Each frame is filled with its own index so a permutation is visible
    rather than merely suspected.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    paths = _write_ramp(tmp_path)

    da = load_tiff_stack(paths, max_workers=8)

    np.testing.assert_allclose(da.values[:, 0, 0], np.arange(len(paths), dtype=np.float32))
    for i in range(len(paths)):
        np.testing.assert_allclose(da.values[i], float(i))


def test_parallel_decode_keeps_each_tag_with_its_own_frame(tmp_path):
    """A per-frame TIFF tag must stay aligned with the pixels it came from.

    The tags are collected off the same futures as the pixels, so tags stored by completion order
    rather than by input index would attach the wrong exposure time to every frame — and every
    value would still be present, so a test comparing sets or names would pass. This is not
    hypothetical: mutating ``tags[index]`` to ``tags[n - index]`` left all ten of this file's other
    tests passing while ``InteropIndex`` came out permuted.

    It matters downstream, not just here: ``run_mars_ccd_pipeline`` passes
    ``metadata_keys_to_sum=("ExposureTime",)``, so a mis-paired tag is summed into the HDF5 product.

    Frame i carries both pixel value i and ``ImageDescription`` "frame i", which pins the two to
    each other.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    paths = []
    for i in range(17):
        p = tmp_path / f"frame{i:03d}.tif"
        # metadata=None or tifffile writes its own JSON blob into ImageDescription, which is
        # identical across frames and so would land in the scalar branch, testing nothing.
        tifffile.imwrite(p, np.full((3, 4), float(i), dtype=np.float32), description=f"frame {i}", metadata=None)
        paths.append(p)

    da = load_tiff_stack(paths, max_workers=8)

    assert list(da.coords["ImageDescription"].values) == [f"frame {i}" for i in range(len(paths))]
    np.testing.assert_allclose(da.values[:, 0, 0], np.arange(len(paths), dtype=np.float32))


def test_parallel_and_serial_decode_agree(tmp_path):
    """max_workers=1 and a real pool must produce identical arrays, variances and coordinates.

    This guards against divergence that appears only with a pool — a race, a dropped frame, a
    worker-only code path. It does **not** establish that the order is right: both calls run the
    same placement code, so a bug that misplaces every frame identically leaves the two agreeing.
    Verified by mutation: reversing the write index leaves this test passing and is caught only by
    test_parallel_decode_preserves_frame_order, which is where the ordering guarantee lives.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    paths = _write_ramp(tmp_path)

    import scipp as sc

    serial = load_tiff_stack(paths, max_workers=1)
    parallel = load_tiff_stack(paths, max_workers=8)

    # sc.identical rather than a coordinate-name comparison: names matching says nothing about
    # coordinate values, dims or alignment, and those are published output too.
    assert sc.identical(serial, parallel)


def test_parallel_decode_raises_on_shape_mismatch(tmp_path):
    """An inconsistent frame still raises, and names the offending file."""
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    paths = _write_ramp(tmp_path, n=6)
    odd = tmp_path / "frame003.tif"
    tifffile.imwrite(odd, np.zeros((5, 9), dtype=np.float32))

    with pytest.raises(ValueError, match="Shape mismatch"):
        load_tiff_stack(paths, max_workers=4)


def test_parallel_decode_propagates_a_failed_read(tmp_path):
    """A missing file raises rather than hanging the pool or yielding a partial stack."""
    from neunorm.loaders.tiff_loader import load_tiff_stack

    paths = _write_ramp(tmp_path, n=6)
    paths[4] = tmp_path / "does-not-exist.tif"

    with pytest.raises(Exception):  # noqa: B017 - tifffile's own error type is not part of the contract
        load_tiff_stack(paths, max_workers=4)


# --------------------------------------------------------------------------------------
# what the decoder swap must NOT change
#
# tifffile replaced Pillow for the pixels, and these are the file kinds where the two disagree.
# Each case is compared against a value written into the file, not against another run of this
# loader: a test that runs the new implementation twice cannot see a change relative to the old
# one, which is the whole risk of a decoder swap.
# --------------------------------------------------------------------------------------


def test_lzw_compressed_files_still_load(tmp_path):
    """LZW is decoded by Pillow, because tifffile delegates it to a package NeuNorm does not have.

    tifffile hands LZW, JPEG and CCITT to the optional ``imagecodecs``; without it, ``asarray()``
    raises ``ValueError: <COMPRESSION.LZW: 5> requires the 'imagecodecs' package``. LZW is what
    ImageJ/Fiji, MATLAB and Pillow write, so a re-saved stack would otherwise stop loading
    outright. Written with Pillow because tifffile cannot even encode LZW here.
    """
    from PIL import Image

    from neunorm.loaders.tiff_loader import load_tiff_stack

    paths = []
    for i in range(3):
        p = tmp_path / f"lzw{i}.tif"
        Image.fromarray(np.full((4, 6), 100 + i, dtype=np.uint16)).save(p, compression="tiff_lzw")
        paths.append(p)

    da = load_tiff_stack(paths, max_workers=4)

    assert da.data.shape == (3, 4, 6)
    np.testing.assert_allclose(da.values[:, 0, 0], [100.0, 101.0, 102.0])


def test_packbits_and_deflate_still_load(tmp_path):
    """The control for the LZW case: these two codecs tifffile does carry, so they take its path.

    Without this, the LZW test alone would be consistent with compression being broken in general.
    """
    from PIL import Image

    from neunorm.loaders.tiff_loader import load_tiff_stack

    for name, compression in (("packbits", "packbits"), ("deflate", "tiff_adobe_deflate")):
        p = tmp_path / f"{name}.tif"
        Image.fromarray(np.full((4, 6), 42, dtype=np.uint16)).save(p, compression=compression)
        da = load_tiff_stack([p])
        np.testing.assert_allclose(da.values[0], 42.0), name


def test_whiteiszero_8bit_keeps_its_inverted_values(tmp_path):
    """An 8-bit WhiteIsZero file must load inverted, as Pillow loaded it.

    Pillow inverts the samples for photometric 0 at 8 bits and below — a stored 0 loads as 255 —
    and tifffile returns them raw. Counts become the Poisson variances, so taking tifffile's values
    here would silently change both the data and its stated uncertainty. Asserted against the
    stored value, so it fails if the frame is ever decoded raw.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "miniswhite8.tif"
    tifffile.imwrite(p, np.full((4, 6), 10, dtype=np.uint8), photometric="miniswhite")

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0], 245.0)


def test_whiteiszero_16bit_is_not_inverted(tmp_path):
    """At 16 bits neither library inverts, so neither may this loader.

    This is what stops anyone "simplifying" the branch into an unconditional inversion on
    photometric 0, which would corrupt every 16-bit WhiteIsZero frame.

    It does **not** pin the `BitsPerSample <= 8` half of the routing condition. Verified by
    mutation: dropping that condition sends 16-bit WhiteIsZero frames down the Pillow path too, and
    Pillow returns the same values at that depth, so this test still passes. That mutant is
    equivalent on values — it only costs tifffile's faster decode — which is why no test forbids it.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "miniswhite16.tif"
    tifffile.imwrite(p, np.full((4, 6), 10, dtype=np.uint16), photometric="miniswhite")

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0], 10.0)


def test_orientation_tagged_file_loads_the_stored_raster(tmp_path):
    """An Orientation tag no longer reorders pixels, and is published instead. Deliberate.

    Pillow does not reorient such a file correctly — it swaps width and height from the tag before
    decoding, so it reads the strips at the wrong width. On this 4x6 ramp with Orientation 6 it
    returns a 4x6 array whose first row is [20, 16, 12, 8, 4, 0]; any valid reorientation of a 4x6
    raster is 6x4, so that output is not a reorientation of anything, it is a mis-strided read.
    This test asserts the stored raster comes back untouched, which is both correct and what this
    project's orientation rule wants: nothing implicit inside the pipeline, applied once at the end
    for display. The tag is published so that display step has it.

    Written to fail loudly if the mis-striding is ever reintroduced: it pins the exact stored
    values, not merely the shape.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "orient.tif"
    stored = np.arange(24, dtype=np.uint16).reshape(4, 6)
    tifffile.imwrite(p, stored, photometric="minisblack", extratags=[(274, "H", 1, 6, True)])

    da = load_tiff_stack([p])

    assert da.data.shape == (1, 4, 6)
    np.testing.assert_allclose(da.values[0], stored.astype(np.float32))
    assert da.coords["Orientation"].values == 6


def test_an_oriented_file_needing_the_pillow_decoder_is_rejected(tmp_path):
    """The one combination with no good answer fails loudly rather than returning scrambled data.

    An LZW frame can only be decoded by Pillow here, and Pillow mis-strides an oriented file, so
    there is no way to produce the stored raster. Silently returning Pillow's interleaved output
    would be the worst outcome of the three.
    """
    from PIL import Image

    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "lzw_oriented.tif"
    Image.fromarray(np.arange(24, dtype=np.uint16).reshape(4, 6)).save(p, compression="tiff_lzw", tiffinfo={274: 6})

    with pytest.raises(ValueError, match="Orientation"):
        load_tiff_stack([p])


def test_max_workers_below_one_is_rejected(tmp_path):
    """0 and -1 are mistakes, not settings, and must not be silently reinterpreted.

    `max_workers or _DEFAULT` read 0 as "unset" and gave 8 threads; the `max(1, ...)` clamp turned
    a negative into a serial read. Both accepted a wrong value without a word.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    paths = _write_ramp(tmp_path, n=3)

    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            load_tiff_stack(paths, max_workers=bad)


def test_a_set_of_paths_still_loads(tmp_path):
    """A sized-but-unindexable collection must still work: the decode addresses frames by index.

    The pre-parallel loaders only iterated `paths`, so a set worked. `_decode_stack` subscripts it,
    and a guard testing only for `__len__` let a set through to a TypeError.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    paths = _write_ramp(tmp_path, n=4)

    da = load_tiff_stack(set(paths), max_workers=2)

    assert da.data.shape == (4, 3, 4)
    # A set has no order, so only the multiset of frame values is defined here.
    assert sorted(da.values[:, 0, 0]) == [0.0, 1.0, 2.0, 3.0]
