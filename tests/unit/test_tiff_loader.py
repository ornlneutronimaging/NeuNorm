"""
Unit tests for the TIFF data loader.

These tests verify loading TIFF image stacks, including variants with time-of-flight (TOF) binning.
"""

import re
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
        np.testing.assert_allclose(da.values[0], 42.0, err_msg=name)


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


@pytest.mark.parametrize(
    "shape",
    [(4, 6), (6, 6)],
    ids=["non-square", "square"],
)
@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_orientation_tagged_file_loads_the_stored_raster(tmp_path, shape, orientation):
    """An Orientation tag no longer reorders pixels, and is published instead. Deliberate.

    **The square case is the one that matters and it is not a bug fix.** Measured against
    ``np.rot90``, Pillow's decode of an oriented file is a *correct* rotation whenever the raster
    is square — which is every real detector frame — and also at orientation 3 for any shape. It
    is a mis-strided read only at orientation 6 and 8 on a non-square raster, where it swaps width
    and height from the tag before decoding. So for square frames this change removes a rotation
    that was right, and for non-square 6/8 it un-scrambles one that was not.

    Both are wanted for the same reason: this project applies orientation once at the end for
    display and never implicitly inside the pipeline, and the tag is published so that display step
    still has it. But the square case means a stack can now load un-rotated relative to 2.4.0,
    which is a visible change and not merely a repair.

    Parametrised over both shapes and all three interesting orientation values so the guarantee is
    "the stored raster, always" rather than an accident of one fixture — the original version of
    this test used a 4x6 frame at orientation 6, the single combination that flattered the
    mis-striding explanation.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / f"orient{orientation}.tif"
    stored = np.arange(shape[0] * shape[1], dtype=np.uint16).reshape(shape)
    tifffile.imwrite(p, stored, photometric="minisblack", extratags=[(274, "H", 1, orientation, True)])

    da = load_tiff_stack([p])

    assert da.data.shape == (1, *shape)
    np.testing.assert_allclose(da.values[0], stored.astype(np.float32))
    assert da.coords["Orientation"].values == orientation


def test_an_oriented_file_needing_the_pillow_decoder_is_rejected(tmp_path):
    """The one combination with no good answer fails loudly rather than guessing.

    An LZW frame can only be decoded by Pillow here, and Pillow applies the Orientation tag as it
    decodes. On this path it applies it *correctly* — measured, its libtiff decoder returns an exact
    ``np.rot90`` at every orientation and shape — but it cannot be asked for the un-rotated raster,
    which is what this loader publishes for every other file. Inverting the rotation afterwards
    would mean knowing which of Pillow's decoders ran, since its raw decoder mis-strides the same
    tag instead of rotating, and that guess is what refusing avoids.
    """
    from PIL import Image

    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "lzw_oriented.tif"
    Image.fromarray(np.arange(24, dtype=np.uint16).reshape(4, 6)).save(p, compression="tiff_lzw", tiffinfo={274: 6})

    with pytest.raises(ValueError, match="Orientation"):
        load_tiff_stack([p])


def _write_raw_tiff(path, width, height, bits, payload, *, photometric=1, sample_format=1, extras=(), omit=()):
    """A minimal little-endian single-strip TIFF, for depths no library here can encode.

    tifffile refuses to write 2-, 4- and 12-bit samples without ``imagecodecs``, and Pillow will
    not write them either, so a fixture for those depths has to be assembled by hand. ``omit``
    leaves out tags by code, which is the only way to build a file *missing* a tag both libraries
    always write.
    """
    import struct

    tags = sorted(
        tag
        for tag in [
            (256, 3, 1, width),
            (257, 3, 1, height),
            (258, 3, 1, bits),
            (259, 3, 1, 1),
            (262, 3, 1, photometric),
            (273, 4, 1, 0),
            (277, 3, 1, 1),
            (278, 3, 1, height),
            (279, 4, 1, len(payload)),
            (339, 3, 1, sample_format),
        ]
        + list(extras)
        if tag[0] not in omit
    )
    header = b"II" + struct.pack("<HI", 42, 8)
    data_offset = len(header) + 2 + 12 * len(tags) + 4
    out = bytearray(header) + struct.pack("<H", len(tags))
    for code, typ, count, value in tags:
        if code == 273:
            value = data_offset
        out += struct.pack("<HHI", code, typ, count)
        out += struct.pack("<I", value) if typ == 4 else struct.pack("<HH", value, 0)
    out += struct.pack("<I", 0) + payload
    Path(path).write_bytes(bytes(out))


def test_twelve_bit_packed_samples_load_exactly(tmp_path):
    """A 12-bit packed frame must load, with its stored counts, through the Pillow fallback.

    12 bits is an ordinary CCD/CMOS depth. tifffile cannot unpack non-byte-aligned samples without
    `imagecodecs` — it raises NotImplementedError, which is not even a ValueError — and Pillow
    decodes them exactly. This is the case that made predicting tifffile's failures the wrong
    design: it is not a compression code, so no list of codecs would have caught it.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    values = [0, 137, 274, 411, 2000, 4095]
    payload = bytearray()
    for i in range(0, len(values), 2):
        a, b = values[i], values[i + 1]
        payload += bytes(((a >> 4) & 0xFF, ((a & 0xF) << 4) | ((b >> 8) & 0xF), b & 0xFF))

    p = tmp_path / "b12.tif"
    _write_raw_tiff(p, len(values), 1, 12, bytes(payload))

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0, 0], np.array(values, dtype=np.float32))


@pytest.mark.parametrize("bits", [2, 4])
def test_two_and_four_bit_samples_never_load_altered(tmp_path, bits):
    """A 2- or 4-bit frame must either load its stored counts or be refused — never be rescaled.

    Measured on hand-written files: Pillow rescales these to full range, so a 4-bit 0,1,2,3,4,5
    comes back as 0,17,34,51,68,85 and a 2-bit ramp as all zeros, and the previous loader returned
    those silently, straight into the Poisson variances.

    **Which of the two acceptable outcomes happens depends on the environment, and this test says
    so rather than quietly assuming one.** tifffile unpacks non-byte-aligned samples only with the
    optional ``imagecodecs`` package, which is not a NeuNorm dependency. Without it tifffile
    raises, the Pillow fallback is reached, and the loader refuses. With it installed tifffile
    returns the stored counts and the refusal never fires — the better outcome, not a hole.
    Asserting only the refusal would have made this a statement about which packages happen to be
    installed, which is what it was before.
    """
    import importlib.util

    from neunorm.loaders.tiff_loader import load_tiff_stack

    per_byte = 8 // bits
    stored = list(range(per_byte))
    payload = bytes((sum(v << (8 - bits * (i + 1)) for i, v in enumerate(stored)),))
    p = tmp_path / f"b{bits}.tif"
    _write_raw_tiff(p, per_byte, 1, bits, payload)

    if importlib.util.find_spec("imagecodecs") is None:
        with pytest.raises(ValueError, match=f"{bits}-bit samples cannot be decoded"):
            load_tiff_stack([p])
    else:
        da = load_tiff_stack([p])
        np.testing.assert_allclose(da.values[0, 0], np.array(stored, dtype=np.float32))


def test_an_absent_photometric_tag_is_treated_as_whiteiszero(tmp_path):
    """A file with no PhotometricInterpretation tag must load inverted, as Pillow loaded it.

    Pillow defaults that tag to 0 — WhiteIsZero — and inverts at 8 bits and below; tifffile treats
    absent as MinIsBlack and does not. The routing compared the raw tag against 0, which is False
    for an absent tag, so such a frame took the tifffile path and came back un-inverted: a stored
    0, 1, 2, 3 loaded as 255, 254, 253, 252 before and as 0, 1, 2, 3 after.

    Pillow's own source attributes that default to real writers omitting a required tag, so this is
    not a theoretical input. No existing fixture could reach it: `tifffile.imwrite` always writes
    tag 262, which is why this one is hand-written with the tag omitted.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    stored = [0, 1, 2, 3, 10, 20, 200, 255]
    p = tmp_path / "nophoto.tif"
    _write_raw_tiff(p, len(stored), 1, 8, bytes(stored), omit=(262,))

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0, 0], np.array([255 - v for v in stored], dtype=np.float32))


def test_an_absent_photometric_tag_is_not_inverted_at_sixteen_bits(tmp_path):
    """The control: Pillow only inverts WhiteIsZero at 8 bits and below, so 16-bit must not flip.

    This forbids turning the routing into an unconditional inversion on photometric 0, which would
    corrupt every 16-bit frame whose photometric tag is absent or 0.

    It does **not** pin the ``BitsPerSample <= 8`` half of the routing condition. Verified by
    mutation: dropping that half sends these frames down the Pillow path as well, and Pillow
    returns the same values at 16 bits, so this test still passes. Equivalent on values — it only
    costs tifffile's faster decode — which is why no test forbids it. Same finding as
    test_whiteiszero_16bit_is_not_inverted.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    stored = np.array([0, 1, 2, 3, 1000, 40000], dtype=np.uint16)
    p = tmp_path / "nophoto16.tif"
    _write_raw_tiff(p, len(stored), 1, 16, stored.tobytes(), omit=(262,))

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0, 0], stored.astype(np.float32))


def test_a_tag_missing_from_some_frames_is_dropped_not_raised(tmp_path):
    """A stack mixing frames with and without a tag must load, in either order.

    The metadata block indexes every frame with frame 0's keys, so a tag frame 0 carries and a
    later frame lacks escaped as a bare ``KeyError: 274`` from a public function. Orientation was
    unreachable there until this loader stopped reading tags through Pillow's pixel load, which
    deletes tag 274 as a side effect — so the failure is new, and it depended on file order:
    oriented-first raised, plain-first silently dropped the coordinate. Both now drop it.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    stored = np.arange(24, dtype=np.uint16).reshape(4, 6)
    oriented = tmp_path / "a_oriented.tif"
    plain = tmp_path / "b_plain.tif"
    tifffile.imwrite(oriented, stored, photometric="minisblack", extratags=[(274, "H", 1, 1, True)])
    tifffile.imwrite(plain, stored, photometric="minisblack")

    for paths in ([oriented, plain], [plain, oriented]):
        da = load_tiff_stack(paths)
        assert da.data.shape == (2, 4, 6)
        assert "Orientation" not in da.coords, "a tag missing from one frame must not be published"


def test_signed_eight_bit_negative_counts_are_rejected_not_reinterpreted(tmp_path):
    """A signed 8-bit frame with negative samples must raise, not load as large positive counts.

    Pillow maps SampleFormat 2 at 8 bits to unsigned, so a stored -12 loaded as 244 and sailed
    past the non-negative-counts guard as a plausible count. tifffile reads -12, the guard fires,
    and that is the correct outcome: silently reinterpreting corrupt or mis-declared data as valid
    counts is exactly what that guard exists to prevent. Deliberately not routed to Pillow.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "i8.tif"
    _write_raw_tiff(p, 6, 1, 8, np.array([-12, -1, 0, 1, 100, 127], dtype=np.int8).tobytes(), sample_format=2)

    with pytest.raises(ValueError, match="negative counts"):
        load_tiff_stack([p])


def test_the_routing_covers_every_pillow_decode_that_transforms_values():
    """The routing is complete against Pillow's own decision table, not just against a sweep.

    Three review rounds each found another file kind where Pillow and tifffile disagree, because
    each fix was aimed at the instance found rather than the class. This closes it from the other
    end. Pillow decides how to decode a TIFF by looking
    ``(byteorder, photometric, sampleformat, planarconfig, bitspersample, extrasamples)`` up in
    ``TiffImagePlugin.OPEN_INFO`` and taking the *rawmode* it finds. A rawmode whose suffix carries
    ``I`` inverts the samples; one naming a sub-byte width (``L;2``, ``L;4``) expands them to full
    range. Those are the only two ways Pillow's decode differs in value from returning the stored
    samples, so enumerating that table enumerates the whole risk.

    Two properties are asserted, and between them they are what the loader relies on:

    1. **No entry transforms above 8 bits per sample.** This is what licenses the
       ``BitsPerSample <= 8`` half of the routing condition — and therefore what licenses sending
       every 16- and 32-bit frame, which is all real detector data, down tifffile's faster path.
    2. **Every inverting entry is routed to Pillow** by `_needs_pillow_pixels`, or refused outright
       for a depth where Pillow rescales.

    If a future Pillow adds an inverting mode at 16 bits, or an inverting combination the predicate
    does not match, this fails instead of a stack quietly loading 65535-x.
    """
    from PIL import TiffImagePlugin

    from neunorm.loaders.tiff_loader import _UNLOADABLE_BIT_DEPTHS, _needs_pillow_pixels

    def transforms(rawmode: str) -> bool:
        """Whether this rawmode's output differs in value from the stored samples.

        A rawmode is ``mode`` optionally followed by ``;<width><flags>``. Only two flags change
        values: ``I`` inverts, and a declared width of 2 or 4 expands sub-byte samples to full
        range. ``F``/``B``/``N``/``R`` are float, byte order and fill order — how the samples are
        stored, not a transform of them. The width must be parsed as a number, not searched for as
        a substring: ``F;32F`` and ``I;12`` both contain a "2".
        """
        match = re.fullmatch(r"(?P<mode>[^;]+)(?:;(?P<bits>\d*)(?P<flags>[A-Z]*))?", rawmode)
        assert match is not None, f"unparsed rawmode {rawmode!r}; the assertions below would be vacuous"
        flags = match.group("flags") or ""
        bits = match.group("bits") or ""
        return "I" in flags or bits in ("2", "4")

    transforming = {key: value for key, value in TiffImagePlugin.OPEN_INFO.items() if transforms(value[1])}
    assert transforming, "OPEN_INFO yielded no transforming rawmodes; the parsing above is wrong"

    above_eight = {k: v for k, v in transforming.items() if max(k[4]) > 8}
    assert not above_eight, (
        "Pillow now transforms values above 8 bits per sample, so the loader's depth guard no "
        f"longer bounds the risk: {sorted(above_eight.items())[:3]}"
    )

    unrouted = []
    for (_byteorder, photometric, sample_format, _planar, bits, _extra), (_mode, rawmode) in transforming.items():
        tags = {262: photometric, 258: bits, 339: sample_format}
        if _needs_pillow_pixels(tags) or bits[0] in _UNLOADABLE_BIT_DEPTHS:
            continue
        unrouted.append((photometric, sample_format, bits, rawmode))
    assert not unrouted, f"Pillow transforms these but the loader sends them to tifffile: {sorted(set(unrouted))}"


def test_a_planar_multisample_file_is_rejected(tmp_path):
    """A multi-sample TIFF is not a detector frame and must fail loudly, not load something.

    This pins the rejection, not the routing. The two readers do disagree on such a file — Pillow
    returns ``(y, x, sample)`` and tifffile ``(sample, y, x)`` — but the disagreement is invisible
    from here: either way the frame is 3-D, the stack is 4-D, and the unpack raises the same error.
    Verified by mutation: adding a routing clause for these files, then removing it again, leaves
    this test passing both times, which is why the loader has no such clause.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "planar.tif"
    tifffile.imwrite(
        p,
        np.arange(3 * 4 * 6, dtype=np.uint8).reshape(3, 4, 6),
        planarconfig="separate",
        photometric="rgb",
    )

    with pytest.raises(ValueError, match="too many values to unpack"):
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
