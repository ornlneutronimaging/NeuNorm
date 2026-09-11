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
# This loader replaced a Pillow-only read. Every file that loaded before must still load, with
# the same values, dtype, shape and published coordinates. Where Pillow and tifffile disagree,
# Pillow wins -- including where Pillow is, on its own merits, wrong: it rescales sub-byte depths,
# inverts WhiteIsZero, reads signed 8-bit as unsigned, and applies an Orientation tag. Reproducing
# those is the point. Whether any of them should change is a separate decision.
#
# Each expected value below was measured by loading the same file through the pre-change loader,
# so these are an independent oracle rather than another run of the code under test.
# --------------------------------------------------------------------------------------


def test_lzw_compressed_files_still_load(tmp_path):
    """LZW must keep loading: tifffile cannot decode it here, so Pillow does.

    tifffile hands LZW, JPEG and CCITT to the optional ``imagecodecs``; without it ``asarray()``
    raises and the loader falls back. LZW is a compression ImageJ/Fiji, MATLAB and Pillow can all
    write, so a re-saved stack would otherwise have stopped loading outright.
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
    """An 8-bit WhiteIsZero file loads inverted, as Pillow loaded it.

    Pillow inverts the samples for photometric 0 at 8 bits and below -- a stored 10 becomes 245 --
    and tifffile returns them raw. Those counts become the Poisson variances, so taking tifffile's
    values here would change the data and its stated uncertainty together.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "miniswhite8.tif"
    tifffile.imwrite(p, np.full((4, 6), 10, dtype=np.uint8), photometric="miniswhite")

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0], 245.0)


def test_whiteiszero_16bit_is_not_inverted(tmp_path):
    """At 16 bits neither library inverts, so neither may this loader.

    This forbids turning the routing into an unconditional inversion on photometric 0, which would
    corrupt every 16-bit WhiteIsZero frame.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "miniswhite16.tif"
    tifffile.imwrite(p, np.full((4, 6), 10, dtype=np.uint16), photometric="miniswhite")

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0], 10.0)


def _write_raw_tiff(path, width, height, bits, payload, *, photometric=1, sample_format=1, extras=(), omit=()):
    """A minimal little-endian single-strip TIFF, for variants no library here can encode.

    tifffile refuses to write 2-, 4- and 12-bit samples without ``imagecodecs``, Pillow will not
    write them or int8 at all, and both always write a photometric tag -- so fixtures for those
    have to be assembled by hand. ``omit`` leaves out tags by code.
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
    """A 12-bit packed frame keeps loading, with its stored counts.

    12 bits is an ordinary CCD/CMOS depth. tifffile cannot unpack non-byte-aligned samples without
    `imagecodecs` -- it raises NotImplementedError, which is not even a ValueError -- and Pillow
    decodes them exactly.
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


@pytest.mark.parametrize(("bits", "scale"), [(2, 85), (4, 17)])
def test_sub_byte_depths_keep_pillow_rescaling(tmp_path, bits, scale):
    """2- and 4-bit frames keep loading rescaled, exactly as before.

    Pillow expands sub-byte samples to the full 8-bit range, so 2-bit 0,1,2,3 loads as
    0,85,170,255 and 4-bit is x17. Those counts feed the Poisson variances, so the rescaling is
    arguably wrong -- but it is what these files have always loaded as, and a change whose purpose
    is decoding speed does not get to alter it. Briefly, this branch refused such files instead;
    that broke input that used to work, which is why the expected values here are Pillow's.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    per_byte = 8 // bits
    stored = list(range(per_byte))
    payload = bytes((sum(v << (8 - bits * (i + 1)) for i, v in enumerate(stored)),))
    p = tmp_path / f"b{bits}.tif"
    _write_raw_tiff(p, per_byte, 1, bits, payload)

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0, 0], np.array(stored, dtype=np.float32) * scale)


def test_one_bit_frames_are_not_rescaled(tmp_path):
    """The control for the sub-byte case: 1-bit is faithful in both readers, so it must stay 0/1."""
    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "b1.tif"
    _write_raw_tiff(p, 8, 1, 1, bytes((0b01101001,)))

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0, 0], [0, 1, 1, 0, 1, 0, 0, 1])


def test_signed_eight_bit_keeps_pillow_unsigned_reading(tmp_path):
    """A signed 8-bit frame keeps loading as unsigned, exactly as before.

    Pillow maps SampleFormat 2 at 8 bits to unsigned, so a stored -12 loads as 244. tifffile
    returns -12, which the caller's non-negative-counts guard would reject -- turning a file that
    used to load into an error. Routing it to Pillow keeps it loading. Whether 244 is a sensible
    count is a real question, and a separate one.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "i8.tif"
    _write_raw_tiff(p, 6, 1, 8, np.array([-12, -1, 0, 1, 100, 127], dtype=np.int8).tobytes(), sample_format=2)

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0, 0], [244.0, 255.0, 0.0, 1.0, 100.0, 127.0])


def test_signed_sixteen_bit_is_unaffected(tmp_path):
    """The control for the depth test: at 16 bits both readers agree, so nothing is rerouted."""
    from neunorm.loaders.tiff_loader import load_tiff_stack

    p = tmp_path / "i16.tif"
    _write_raw_tiff(p, 6, 1, 16, np.array([5, 10, 20, 30, 40, 50], dtype="<i2").tobytes(), sample_format=2)

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0, 0], [5.0, 10.0, 20.0, 30.0, 40.0, 50.0])


def test_an_absent_photometric_tag_is_treated_as_whiteiszero(tmp_path):
    """A file with no PhotometricInterpretation tag keeps loading inverted, as Pillow loaded it.

    Pillow defaults that tag to 0 -- WhiteIsZero -- and inverts at 8 bits and below; tifffile
    treats absent as MinIsBlack. Comparing the raw tag against 0 missed this, since an absent tag
    reads as None. Pillow's own source attributes the default to real writers omitting a required
    tag, so it is not a theoretical input, and no fixture written by tifffile can reach it.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    stored = [0, 1, 2, 3, 10, 20, 200, 255]
    p = tmp_path / "nophoto.tif"
    _write_raw_tiff(p, len(stored), 1, 8, bytes(stored), omit=(262,))

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0, 0], np.array([255 - v for v in stored], dtype=np.float32))


def test_an_absent_photometric_tag_is_not_inverted_at_sixteen_bits(tmp_path):
    """The control: Pillow only inverts WhiteIsZero at 8 bits and below, so 16-bit must not flip."""
    from neunorm.loaders.tiff_loader import load_tiff_stack

    stored = np.array([0, 1, 2, 3, 1000, 40000], dtype=np.uint16)
    p = tmp_path / "nophoto16.tif"
    _write_raw_tiff(p, len(stored), 1, 16, stored.tobytes(), omit=(262,))

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0, 0], stored.astype(np.float32))


@pytest.mark.parametrize("shape", [(4, 6), (6, 6)], ids=["non-square", "square"])
@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_orientation_tagged_files_load_exactly_as_before(tmp_path, shape, orientation):
    """An Orientation tag keeps being applied by Pillow, and keeps not becoming a coordinate.

    Pillow applies the tag while decoding and deletes it from ``tag_v2`` as a side effect, so the
    previous loader returned reoriented pixels and never published ``Orientation``. Both halves are
    reproduced: the pixels come from Pillow, and tag 274 is dropped from the published tags for
    every file, including the ones tifffile decodes.

    This branch briefly returned the stored raster instead and published the tag. That is arguably
    the better behaviour -- this project applies orientation once at the end for display -- but it
    silently changed the pixels of every orientation-tagged stack and added a coordinate to the
    HDF5 product, which is not something a speed change gets to do. The expected array here is
    whatever Pillow produces, exact rotation or mis-strided read alike.
    """
    import tifffile
    from PIL import Image

    from neunorm.loaders.tiff_loader import load_tiff_stack

    stored = np.arange(shape[0] * shape[1], dtype=np.uint16).reshape(shape)
    p = tmp_path / f"orient{orientation}.tif"
    tifffile.imwrite(p, stored, photometric="minisblack", extratags=[(274, "H", 1, orientation, True)])

    with Image.open(p) as img:
        expected = np.asanyarray(img, dtype=np.float32)

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0], expected)
    assert "Orientation" not in da.coords, "the previous loader never published this tag"


def test_an_oriented_lzw_file_still_loads(tmp_path):
    """The combination that needs Pillow twice over -- for the codec and for the tag -- still loads.

    An LZW frame can only be decoded by Pillow here, and it also carries an orientation. This
    branch briefly refused it; it loaded before, so it loads now.
    """
    from PIL import Image

    from neunorm.loaders.tiff_loader import load_tiff_stack

    stored = np.arange(24, dtype=np.uint16).reshape(4, 6)
    p = tmp_path / "lzw_oriented.tif"
    Image.fromarray(stored).save(p, compression="tiff_lzw", tiffinfo={274: 6})

    with Image.open(p) as img:
        expected = np.asanyarray(img, dtype=np.float32)

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0], expected)


def test_the_routing_covers_every_pillow_decode_that_transforms_values():
    """The routing is complete against Pillow's own decision table, not just against a sweep.

    Pillow decides how to decode a TIFF by looking
    ``(byteorder, photometric, sampleformat, planarconfig, bitspersample, extrasamples)`` up in
    ``TiffImagePlugin.OPEN_INFO`` and taking the *rawmode* it finds. A rawmode whose suffix carries
    ``I`` inverts the samples; one naming a sub-byte width expands them to full range. Those are
    the only two ways Pillow's decode differs in value from returning the stored samples, so
    enumerating that table enumerates the whole risk of the decoder swap.

    Asserted: no entry transforms above 8 bits per sample -- which is what licenses sending every
    16- and 32-bit frame, all real detector data, down tifffile's faster path -- and every
    transforming entry is routed to Pillow. If a future Pillow adds an inverting mode at 16 bits,
    this fails instead of a stack quietly loading 65535-x.

    **This covers Pillow's raw decoder only, and saying otherwise is what hid a real difference.**
    ``OPEN_INFO`` governs the uncompressed path. Every *compressed* file goes through Pillow's
    libtiff decoder instead, which does not consult this table and which ignores the file's byte
    order for modes ``I`` and ``F``. An earlier version of this docstring claimed enumerating the
    table "enumerates the whole risk of the decoder swap" and dismissed the byte-order flag as
    storage rather than a transform; both are true of the raw path and false of the libtiff one, so
    the big-endian compressed case in
    ``test_big_endian_compressed_frames_load_the_stored_samples`` went unnoticed until a
    pre-release audit found it.
    """
    from PIL import TiffImagePlugin

    from neunorm.loaders.tiff_loader import _needs_pillow_pixels

    def transforms(rawmode: str) -> bool:
        """Whether this rawmode's output differs in value from the stored samples.

        A rawmode is ``mode`` optionally followed by ``;<width><flags>``. Only two flags change
        values: ``I`` inverts, and a declared width of 2 or 4 expands sub-byte samples to full
        range. ``F``/``B``/``N``/``R`` are float, byte order and fill order -- how the samples are
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

    unrouted = [
        (photometric, sample_format, bits, rawmode)
        for (_byteorder, photometric, sample_format, _planar, bits, _extra), (_mode, rawmode) in transforming.items()
        if not _needs_pillow_pixels({262: photometric, 258: bits, 339: sample_format})
    ]
    assert not unrouted, f"Pillow transforms these but the loader sends them to tifffile: {sorted(set(unrouted))}"


@pytest.mark.parametrize("dtype", ["int16", "int32", "float32"])
# deflate and lzma only. Zstd belongs to the same class and was measured alongside them, but
# tifffile encodes it through `compression.zstd`, which is stdlib only from Python 3.12 -> 3.14;
# on the 3.12 CI runner writing the fixture raises ModuleNotFoundError. Two compressions
# demonstrate the class, and deflate is the one that matters in practice (ImageJ's "ZIP").
@pytest.mark.parametrize("compression", ["deflate", "lzma"])
def test_big_endian_compressed_frames_load_the_stored_samples(tmp_path, dtype, compression):
    """A big-endian compressed frame loads its stored values, which it did not before 2.5.0.

    This is the one case where the decoder swap *did* change what a file loads as, and it changed
    it in the right direction. Pillow routes every compressed file through its libtiff decoder,
    which ignores the file's byte order for modes ``I`` and ``F``, so a big-endian int16, int32 or
    float32 frame came back byte-swapped: a stored 0, 3, 6, 10 read as 0, 768, 1536, 2560 at int16.
    tifffile honours the byte order, so the same file now loads correctly.

    Only big-endian *compressed* frames are affected — uncompressed ones go through the raw
    decoder, which does honour byte order, and uint16 carries its order in the rawmode (``I;16B``).
    Measured across 64 dtype/byte-order/compression combinations: 9 differ, all of this shape, and
    none is a regression.

    The harness that checked compatibility for this release had no big-endian variant, which is why
    this needed an audit to find; see the note in
    ``test_the_routing_covers_every_pillow_decode_that_transforms_values`` for the reasoning error
    behind that gap.
    """
    import tifffile

    from neunorm.loaders.tiff_loader import load_tiff_stack

    stored = np.array([[0, 3, 6, 10]], dtype=dtype)
    p = tmp_path / f"be_{dtype}_{compression}.tif"
    tifffile.imwrite(
        p,
        stored.astype(">" + np.dtype(dtype).str[1:]),
        photometric="minisblack",
        compression=compression,
        byteorder=">",
    )

    da = load_tiff_stack([p])

    np.testing.assert_allclose(da.values[0, 0], stored[0].astype(np.float32))


def test_a_planar_multisample_file_is_rejected(tmp_path):
    """A multi-sample TIFF is not a detector frame and fails, as it did before.

    The two readers disagree on axis order for ``PlanarConfiguration`` 2, but the disagreement is
    invisible from here: either way the frame is 3-D, the stack is 4-D, and the unpack raises the
    same error. This is not a new refusal -- the pre-change loader raised too.
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


def test_a_tag_missing_from_some_frames_is_dropped_not_raised(tmp_path):
    """A stack mixing frames with and without a tag loads, in either order.

    The metadata block indexes every frame with frame 0's keys, so a tag frame 0 carries and a
    later frame lacks escaped as a bare ``KeyError``. That is strictly more permissive than before
    -- such a stack used to fail outright -- so it breaks nothing.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    stored = np.arange(24, dtype=np.uint16).reshape(4, 6).tobytes()
    tagged = tmp_path / "a_tagged.tif"
    plain = tmp_path / "b_plain.tif"
    # Hand-written, and each frame carries a tag the other lacks rather than one frame carrying an
    # extra: that keeps the two IFDs the same size, so the strip offsets match. An IFD of a
    # different size shifts StripOffsets between frames, which trips a separate pre-existing
    # failure in the metadata block and would test the wrong thing. Having a tag missing in each
    # direction also exercises both orders inside a single stack.
    _write_raw_tiff(tagged, 6, 4, 16, stored, extras=[(254, 4, 1, 0)])
    _write_raw_tiff(plain, 6, 4, 16, stored, extras=[(255, 4, 1, 1)])

    import io

    from loguru import logger

    warned = {}
    results = {}
    for label, paths in (("tagged first", [tagged, plain]), ("plain first", [plain, tagged])):
        captured = io.StringIO()
        sink_id = logger.add(captured, level="WARNING", format="{message}")
        try:
            results[label] = load_tiff_stack(paths)
        finally:
            logger.remove(sink_id)
        warned[label] = captured.getvalue()

    for label, da in results.items():
        assert da.data.shape == (2, 4, 6), label
    assert set(results["tagged first"].coords) == set(results["plain first"].coords), (
        "the published coordinates depended on file order"
    )
    for name in ("NewSubfileType", "SubfileType"):
        assert name not in results["tagged first"].coords, f"{name} is in only one frame and must not be published"

    # Both non-shared tags must be named whichever order the files arrive in. Warning from the
    # first frame's keys alone mentioned only the tag that frame happened to carry, so which tags a
    # user heard about depended on the order they passed their files -- and the tag carried solely
    # by a later frame was dropped in silence.
    for label, text in warned.items():
        assert "254" in text, f"the tag only the first file carries went unmentioned ({label})"
        assert "255" in text, f"the tag only the second file carries went unmentioned ({label})"


def test_max_workers_rejects_values_that_are_not_a_worker_count(tmp_path):
    """0, a negative, a float, a bool or a string are mistakes, not settings.

    A float is the one that matters: ThreadPoolExecutor compares its live thread count against the
    value rather than truncating it, so ``max_workers=1.5`` builds two threads and ``2.9`` three --
    silently exceeding the cap, which is the only thing this parameter does. ``True`` is an int
    subclass and would otherwise pass as 1.

    numpy integers are accepted, because a caller sizing the pool from an array shape produces one.
    Same rule and same error shapes as ``_check_advance`` in utils/progress.py.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    paths = _write_ramp(tmp_path, n=3)

    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            load_tiff_stack(paths, max_workers=bad)

    for bad in (1.5, 2.9, 1.0, True, "4"):
        with pytest.raises(TypeError, match="max_workers must be an int"):
            load_tiff_stack(paths, max_workers=bad)

    # accepted, and equivalent to the plain int
    import scipp as sc

    assert sc.identical(load_tiff_stack(paths, max_workers=np.int64(2)), load_tiff_stack(paths, max_workers=2))


def test_a_set_of_paths_loads_but_in_no_defined_order(tmp_path):
    """A set is accepted, as it was before, and its frame order is NOT meaningful.

    This pins compatibility, not a recommendation. The pre-change loader iterated ``paths`` without
    subscripting, so a set was accepted and its frames came out in hash order; ``_decode_stack``
    addresses frames by index, so the guard materialises one to keep that working. What neither
    version does is *order* it: frame order is the spectral axis, and a set's iteration order varies
    between processes, so the same set yields a different stack each run and pairs frames with the
    wrong TOF.

    The assertion is therefore on the multiset, which is all that is defined here. ``load_stack``
    rejects a set outright and is deliberately left that way -- see the comment there. Callers
    should pass a sorted sequence, as every fixture and pipeline in this repository does.
    """
    from neunorm.loaders.tiff_loader import load_tiff_stack

    paths = _write_ramp(tmp_path, n=4)

    da = load_tiff_stack(set(paths), max_workers=2)

    assert da.data.shape == (4, 3, 4)
    assert sorted(da.values[:, 0, 0]) == [0.0, 1.0, 2.0, 3.0]
