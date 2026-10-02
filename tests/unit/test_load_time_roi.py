"""Cropping image stacks to an ROI as they load.

``load_tiff_stack``, ``load_fits_stack`` and ``load_stack`` take ``roi=``. A cropped load must be
identical to cropping the full load with ``apply_roi``; the shape and sign checks still cover every
whole frame; a malformed ROI fails before any file is read and an ROI that does not fit fails once
frame 0 is decoded; and only the region is stored.

The frames are non-square (48 x 64) and every pixel value depends on its position and frame index
(``base + 3y + 7x + 11 * frame``), so a shifted, transposed or wrong-frame crop changes values as
well as the shape.
"""

import inspect
import re
import threading
import time
import tracemalloc

import numpy as np
import pytest
import scipp as sc
import tifffile
from astropy.io import fits
from loguru import logger

import neunorm.loaders._frame_stack as frame_stack
import neunorm.loaders.fits_loader as fits_loader
import neunorm.loaders.stack_loader as stack_loader
import neunorm.loaders.tiff_loader as tiff_loader
from neunorm.data_models.roi import ROI, MaskROI
from neunorm.loaders.fits_loader import load_fits_stack
from neunorm.loaders.stack_loader import load_stack
from neunorm.loaders.tiff_loader import load_tiff_stack
from neunorm.processing.roi_clipper import apply_roi

NY, NX = 48, 64
N_FRAMES = 5
INTERIOR = (5, 9, 41, 31)

#: Stored form of each fixture format: file suffix and the dtype the frame is written in. astropy
#: writes uint16 as int16 with ``BZERO = 32768``.
FORMATS = {
    "tiff": (".tif", np.uint16),
    "fits": (".fits", np.uint16),
    "fits-f64": (".fits", ">f8"),
    "tiff-f32": (".tif", np.float32),
    "fits-f32": (".fits", ">f4"),
}

LOADERS = {".tif": load_tiff_stack, ".fits": load_fits_stack}

NEGATIVE_COUNTS = (
    "Loaded {} data contains negative counts; cannot attach Poisson variances (variance = counts) to negative data."
)

IDENTITY_ROIS = {
    "interior": INTERIOR,
    "square-offset": (3, 11, 23, 31),
    "top-left": (0, 0, 10, 7),
    "bottom-right": (50, 40, NX, NY),
    "full-frame": (0, 0, NX, NY),
    "bools": (False, False, True, True),
    "ROI-model": ROI(x0=5, y0=9, x1=41, y1=31),
    "ROI-inclusive": ROI(x0=2, y0=4, width=30, height=20, inclusive=True),
}


def _frame(index, shape=(NY, NX), base=100):
    y, x = np.indices(shape)
    return base + 3 * y + 7 * x + 11 * index


def _write(path, values):
    if path.suffix == ".tif":
        tifffile.imwrite(path, values)
    else:
        fits.PrimaryHDU(values).writeto(path)


def _write_stack(tmp_path, fmt, n=N_FRAMES, *, edit=None, shapes=None, base=100):
    """Write ``n`` frames of ``fmt``; ``edit(index, values)`` may change a frame before it is written."""
    suffix, dtype = FORMATS[fmt]
    paths = []
    for i in range(n):
        values = _frame(i, shape=(shapes or {}).get(i, (NY, NX)), base=base).astype(dtype)
        if edit is not None:
            edit(i, values)
        path = tmp_path / f"{fmt}_{i:03d}{suffix}"
        _write(path, values)
        paths.append(path)
    if fmt == "fits":
        header = fits.getheader(paths[0])
        assert (header["BITPIX"], header["BZERO"]) == (16, 32768)
    return paths


def _loader(paths):
    return LOADERS[paths[0].suffix]


def _assert_same_dataarray(got, expected):
    assert sc.identical(got, expected)
    assert list(got.coords) == list(expected.coords)
    assert {k: c.aligned for k, c in got.coords.items()} == {k: c.aligned for k, c in expected.coords.items()}
    assert got.values.dtype == expected.values.dtype == np.float32
    assert got.variances.dtype == np.float32


@pytest.fixture
def reads(monkeypatch):
    """Count calls to the TIFF and FITS frame readers, from any thread."""
    calls = []
    for module, name in ((tiff_loader, "_read_tiff_frame"), (fits_loader, "_read_fits_frame")):
        original = getattr(module, name)

        def counting(path, _original=original):
            calls.append(path)
            return _original(path)

        monkeypatch.setattr(module, name, counting)
    return calls


@pytest.fixture
def error_log():
    """Messages loguru emits at ERROR or above while the test runs."""
    messages = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level="ERROR")
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def allocations(monkeypatch):
    """Record every ``allocate_stack`` call made while decoding."""
    calls = []
    original = frame_stack.allocate_stack

    def spy(dims, shape):
        calls.append((list(dims), list(shape)))
        return original(dims, shape)

    monkeypatch.setattr(frame_stack, "allocate_stack", spy)
    return calls


# --------------------------------------------------------------------------------------
# identity with apply_roi on the full load
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("with_tof", [False, True], ids=["no-tof", "tof-edges"])
@pytest.mark.parametrize("max_workers", [1, 8])
@pytest.mark.parametrize("roi", list(IDENTITY_ROIS.values()), ids=list(IDENTITY_ROIS))
@pytest.mark.parametrize("fmt", ["tiff", "fits", "fits-f64"])
def test_cropped_load_is_identical_to_cropping_the_full_load(tmp_path, fmt, roi, max_workers, with_tof):
    paths = _write_stack(tmp_path, fmt)
    loader = _loader(paths)
    tof_edges = np.linspace(1000.0, 2000.0, N_FRAMES + 1) if with_tof else None

    got = loader(paths, tof_edges, max_workers=max_workers, roi=roi)
    expected = apply_roi(loader(paths, tof_edges, max_workers=max_workers), roi)

    _assert_same_dataarray(got, expected)


@pytest.mark.parametrize("max_workers", [1, 8])
@pytest.mark.parametrize("roi", [INTERIOR, (3, 11, 23, 31), (0, 0, NX, NY)], ids=["interior", "offset", "full"])
@pytest.mark.parametrize("variant", ["orientation-6", "whiteiszero-8bit"])
def test_pillow_decoded_frames_crop_identically(tmp_path, monkeypatch, variant, roi, max_workers):
    """Frames that Pillow decodes (an Orientation tag, 8-bit WhiteIsZero) crop like any other."""
    paths = []
    for i in range(N_FRAMES):
        path = tmp_path / f"{variant}_{i:03d}.tif"
        if variant == "orientation-6":
            values = _frame(i).astype(np.uint16)
            tifffile.imwrite(path, values, photometric="minisblack", extratags=[(274, "H", 1, 6, True)])
        else:
            values = (_frame(i) % 256).astype(np.uint8)
            tifffile.imwrite(path, values, photometric="miniswhite")
        paths.append(path)

    pillow_decodes = []
    original = tiff_loader._pillow_pixels

    def counting(img):
        pillow_decodes.append(1)
        return original(img)

    monkeypatch.setattr(tiff_loader, "_pillow_pixels", counting)

    got = load_tiff_stack(paths, max_workers=max_workers, roi=roi)
    expected = apply_roi(load_tiff_stack(paths, max_workers=max_workers), roi)

    assert len(pillow_decodes) == 2 * N_FRAMES
    assert got.sizes["y"] == roi[3] - roi[1] and got.sizes["x"] == roi[2] - roi[0]
    _assert_same_dataarray(got, expected)


# --------------------------------------------------------------------------------------
# load_stack forwarding
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("roi", [INTERIOR, ROI(x0=3, y0=11, x1=23, y1=31)], ids=["tuple", "ROI-model"])
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_load_stack_forwards_roi(tmp_path, fmt, roi):
    paths = _write_stack(tmp_path, fmt)

    got = load_stack(paths, roi=roi)

    _assert_same_dataarray(got, apply_roi(load_stack(paths), roi))
    _assert_same_dataarray(got, _loader(paths)(paths, roi=roi))


@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_load_stack_forwards_max_workers(tmp_path, reads, fmt):
    paths = _write_stack(tmp_path, fmt)

    with pytest.raises(ValueError, match="max_workers must be at least 1"):
        load_stack(paths, max_workers=0)
    assert reads == []


# --------------------------------------------------------------------------------------
# static ROI errors: before any file is read
# --------------------------------------------------------------------------------------

STATIC_BAD_ROIS = {
    "numpy-int64": tuple(np.int64(v) for v in INTERIOR),
    "floats": (5.0, 9.0, 41.0, 31.0),
    "x1-equals-x0": (10, 9, 10, 31),
    "x1-below-x0": (20, 9, 10, 31),
    "y1-equals-y0": (5, 9, 41, 9),
    "negative-x0": (-1, 9, 41, 31),
    "negative-y0": (5, -2, 41, 31),
    "three-values": (5, 9, 41),
    "five-values": (5, 9, 41, 31, 1),
}

ENTRY_POINTS = {
    "load_tiff_stack": ("tiff", load_tiff_stack),
    "load_fits_stack": ("fits", load_fits_stack),
    "load_stack-tiff": ("tiff", load_stack),
    "load_stack-fits": ("fits", load_stack),
}


@pytest.mark.parametrize("bad_roi", list(STATIC_BAD_ROIS.values()), ids=list(STATIC_BAD_ROIS))
@pytest.mark.parametrize("entry", list(ENTRY_POINTS), ids=list(ENTRY_POINTS))
def test_malformed_roi_raises_like_apply_roi_before_any_read(tmp_path, reads, error_log, entry, bad_roi):
    fmt, load = ENTRY_POINTS[entry]
    paths = _write_stack(tmp_path, fmt)
    with pytest.raises(ValueError) as from_apply_roi:
        apply_roi(load(paths), bad_roi)
    reads.clear()

    with pytest.raises(ValueError) as from_loader:
        load(paths, roi=bad_roi)

    assert type(from_loader.value) is type(from_apply_roi.value)
    assert str(from_loader.value) == str(from_apply_roi.value)
    assert reads == []
    assert error_log == []


@pytest.mark.parametrize("entry", list(ENTRY_POINTS), ids=list(ENTRY_POINTS))
def test_mask_roi_is_rejected_before_any_read(tmp_path, reads, entry):
    fmt, load = ENTRY_POINTS[entry]
    paths = _write_stack(tmp_path, fmt)
    selection = np.zeros((NY, NX), dtype=bool)
    selection[9:31, 5:41] = True
    mask = MaskROI(selection=selection)
    with pytest.raises(TypeError) as from_apply_roi:
        apply_roi(load(paths), mask)
    reads.clear()

    with pytest.raises(TypeError, match="does not accept a MaskROI") as from_loader:
        load(paths, roi=mask)

    message = str(from_loader.value)
    assert message.startswith("The roi argument crops to a rectangle and does not accept a MaskROI.")
    assert message.removeprefix("The roi argument") == str(from_apply_roi.value).removeprefix("apply_roi")
    assert reads == []


# --------------------------------------------------------------------------------------
# an ROI that does not fit the frames
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("max_workers", [1, None], ids=["serial", "default-pool"])
@pytest.mark.parametrize("roi", [(5, 9, NX + 1, 31), (5, 9, 41, NY + 1)], ids=["past-x", "past-y"])
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_roi_past_the_frame_raises_after_the_first_frame(tmp_path, reads, error_log, fmt, roi, max_workers):
    paths = _write_stack(tmp_path, fmt)
    loader = _loader(paths)
    with pytest.raises(ValueError) as from_apply_roi:
        apply_roi(loader(paths), roi)
    reads.clear()
    events = []

    with pytest.raises(ValueError, match="exceeds data size") as from_loader:
        loader(paths, max_workers=max_workers, roi=roi, progress=events.append)

    assert str(from_loader.value) == str(from_apply_roi.value)
    assert len(reads) == 1
    assert not [e for e in events if e.detail.startswith("attaching variances")]
    assert error_log == []


@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_roi_past_the_frame_allocates_nothing(tmp_path, allocations, fmt):
    paths = _write_stack(tmp_path, fmt)

    with pytest.raises(ValueError, match="exceeds data size"):
        _loader(paths)(paths, roi=(5, 9, NX + 1, 31))

    assert allocations == []


# --------------------------------------------------------------------------------------
# whole-frame checks under an ROI
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("with_nan", [False, True], ids=["negative", "negative-and-nan"])
@pytest.mark.parametrize("bad_frame", [0, 3])
@pytest.mark.parametrize("fmt", ["tiff-f32", "fits-f32"])
def test_negative_pixel_outside_the_roi_still_raises(tmp_path, error_log, fmt, bad_frame, with_nan):
    def edit(i, values):
        if i == bad_frame:
            values[0, 0] = -1.0
            if with_nan:
                values[0, 1] = np.nan

    paths = _write_stack(tmp_path, fmt, edit=edit)

    with pytest.raises(ValueError) as raised:
        _loader(paths)(paths, roi=INTERIOR)

    kind = "TIFF" if paths[0].suffix == ".tif" else "FITS"
    assert str(raised.value) == NEGATIVE_COUNTS.format(kind)
    assert error_log == []


@pytest.mark.parametrize("roi", [None, (0, 0, 10, 10), INTERIOR], ids=["no-roi", "roi-with-pixel", "roi-without-pixel"])
def test_tiny_float64_negative_loads_as_negative_zero(tmp_path, roi):
    """A float64 value too small for float32 is stored as -0.0 and is not a negative count."""

    def edit(i, values):
        if i == 1:
            values[2, 3] = -1e-300

    paths = _write_stack(tmp_path, "fits-f64", edit=edit)

    da = load_fits_stack(paths, roi=roi)

    if roi != INTERIOR:
        value, variance = da.values[1, 2, 3], da.variances[1, 2, 3]
        assert value == 0.0 and np.signbit(value)
        assert variance == 0.0 and np.signbit(variance)
    _assert_same_dataarray(da, load_fits_stack(paths) if roi is None else apply_roi(load_fits_stack(paths), roi))


@pytest.mark.parametrize("max_workers", [1, 8])
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_a_larger_later_frame_that_contains_the_roi_is_a_shape_mismatch(tmp_path, error_log, fmt, max_workers):
    paths = _write_stack(tmp_path, fmt, shapes={2: (50, 70)})

    with pytest.raises(frame_stack._ShapeMismatchError) as raised:
        _loader(paths)(paths, max_workers=max_workers, roi=INTERIOR)

    assert str(raised.value) == f"Shape mismatch in file {paths[2]}: expected ({NY}, {NX}), got (50, 70)"
    assert error_log == []


# --------------------------------------------------------------------------------------
# progress
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("roi", "note"),
    [
        (INTERIOR, "attaching variances to 5 frames of 36 x 22 px (15.5 KiB)"),
        (None, "attaching variances to 5 frames of 64 x 48 px (60.0 KiB)"),
    ],
    ids=["roi", "no-roi"],
)
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_progress_counts_every_file_then_one_note_on_the_calling_thread(tmp_path, fmt, roi, note):
    paths = _write_stack(tmp_path, fmt)
    events = []

    _loader(paths)(paths, max_workers=8, roi=roi, progress=lambda e: events.append((e, threading.get_ident())))

    assert {ident for _, ident in events} == {threading.get_ident()}
    per_file = [e for e, _ in events if not e.detail.startswith("attaching variances")]
    notes = [e for e, _ in events if e.detail.startswith("attaching variances")]
    assert [e.completed for e in per_file] == list(range(1, N_FRAMES + 1))
    assert sorted(e.detail for e in per_file) == sorted(p.name for p in paths)
    assert [e.detail for e in notes] == [note]
    assert events[-1][0] is notes[0]
    assert notes[0].completed == N_FRAMES


# --------------------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("roi", [None, (100, 200, 164, 232)], ids=["no-roi", "roi-64x32"])
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_traced_peak_stays_far_below_a_float32_stack(tmp_path, fmt, roi):
    """No numpy array the size of the stack is created while loading.

    The stack itself is scipp's buffer, which tracemalloc does not see; numpy arrays are traced. A
    serial load traces about one decoded frame, a small fraction of the float32 stack. A numpy
    float32 stack is 1.0 of it and a bool mask over the stack 0.25, so the 0.125 bound fails on
    either.
    """
    n, ny, nx = 32, 512, 512
    suffix = ".tif" if fmt == "tiff" else ".fits"
    paths = []
    for i in range(n):
        path = tmp_path / f"big_{i:03d}{suffix}"
        _write(path, np.full((ny, nx), 100 + i, dtype=np.uint16))
        paths.append(path)
    loader = LOADERS[suffix]
    # Imports and caches a first call creates are not part of a load's footprint.
    loader(paths[:2], max_workers=1, roi=roi)

    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        baseline, _ = tracemalloc.get_traced_memory()
        da = loader(paths, max_workers=1, roi=roi)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if started:
            tracemalloc.stop()

    expected_shape = (n, ny, nx) if roi is None else (n, roi[3] - roi[1], roi[2] - roi[0])
    assert da.shape == expected_shape
    float32_stack = n * ny * nx * 4
    assert peak - baseline < 0.125 * float32_stack


@pytest.mark.parametrize("max_workers", [1, 8])
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_stack_is_allocated_once_at_the_roi_size(tmp_path, allocations, fmt, max_workers):
    paths = _write_stack(tmp_path, fmt)
    x0, y0, x1, y1 = INTERIOR

    _loader(paths)(paths, max_workers=max_workers, roi=INTERIOR)

    assert allocations == [(["N_image", "y", "x"], [N_FRAMES, y1 - y0, x1 - x0])]


@pytest.mark.parametrize("roi", [None, INTERIOR], ids=["no-roi", "roi"])
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_returned_data_is_the_allocated_stack(tmp_path, monkeypatch, fmt, roi):
    """The loaded values and variances live in the buffers ``allocate_stack`` returned, not in a copy.

    scipp's buffers are invisible to tracemalloc, so a scipp-side copy of the stack is checked here.
    """
    allocated = []
    allocate_stack = frame_stack.allocate_stack

    def keeping(dims, shape):
        allocated.append(allocate_stack(dims, shape))
        return allocated[-1]

    monkeypatch.setattr(frame_stack, "allocate_stack", keeping)
    paths = _write_stack(tmp_path, fmt)

    da = _loader(paths)(paths, roi=roi)

    assert len(allocated) == 1
    assert np.shares_memory(da.values, allocated[0].values)
    assert np.shares_memory(da.variances, allocated[0].variances)


def test_tiff_reader_returns_the_file_dtype(tmp_path, monkeypatch):
    """A tifffile-decoded frame keeps its stored dtype; the float32 cast happens on copy into the stack."""
    path = tmp_path / "raw.tif"
    tifffile.imwrite(path, _frame(0).astype(np.uint16))
    monkeypatch.setattr(tiff_loader, "_pillow_pixels", lambda _img: pytest.fail("expected the tifffile path"))

    values, _ = tiff_loader._read_tiff_frame(path)

    assert values.dtype == np.uint16
    assert values.shape == (NY, NX)


# --------------------------------------------------------------------------------------
# variances
# --------------------------------------------------------------------------------------


def _unique_base(fmt, roi, offset):
    """A pixel base no other load in the module uses, so a reused, unfilled buffer cannot hold these counts."""
    return offset + 1000 * ["tiff", "fits"].index(fmt) + (500 if roi is not None else 0)


@pytest.mark.parametrize("roi", [None, INTERIOR], ids=["no-roi", "roi"])
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_variances_equal_the_counts_in_their_own_buffer(tmp_path, fmt, roi):
    paths = _write_stack(tmp_path, fmt, base=_unique_base(fmt, roi, 10000))

    da = _loader(paths)(paths, roi=roi)

    np.testing.assert_allclose(da.variances, da.values, rtol=0, atol=0)
    before = da.variances[0, 0, 0]
    da.values[0, 0, 0] = before + 1000.0
    assert da.values[0, 0, 0] == before + 1000.0
    assert da.variances[0, 0, 0] == before


@pytest.mark.parametrize("roi", [None, INTERIOR], ids=["no-roi", "roi"])
@pytest.mark.parametrize("fmt", ["tiff", "fits"])
def test_data_matches_an_explicit_values_and_variances_construction(tmp_path, fmt, roi):
    paths = _write_stack(tmp_path, fmt, base=_unique_base(fmt, roi, 20000))

    da = _loader(paths)(paths, roi=roi)

    rebuilt = sc.array(dims=list(da.dims), values=da.values, variances=da.values.copy(), unit="counts")
    assert sc.identical(da.data, rebuilt)


# --------------------------------------------------------------------------------------
# frames that are not 2-D
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("roi", [None, INTERIOR], ids=["no-roi", "roi"])
def test_rgb_first_frame_then_2d_frames_names_the_first_mismatching_frame(tmp_path, monkeypatch, error_log, roi):
    paths = [tmp_path / "frame_000.tif"]
    tifffile.imwrite(paths[0], np.zeros((NY, NX, 3), dtype=np.uint8), photometric="rgb")
    for i in range(1, 4):
        paths.append(tmp_path / f"frame_{i:03d}.tif")
        tifffile.imwrite(paths[-1], _frame(i).astype(np.uint16))
    read = tiff_loader._read_tiff_frame

    def frames_after_1_finish_late(path):
        if path in paths[2:]:
            time.sleep(0.2)
        return read(path)

    # Frames 2 and 3 mismatch too; delaying them makes frame 1 the first failure the pool reports.
    monkeypatch.setattr(tiff_loader, "_read_tiff_frame", frames_after_1_finish_late)
    events = []

    with pytest.raises(frame_stack._ShapeMismatchError) as raised:
        load_tiff_stack(paths, max_workers=1, roi=roi, progress=events.append)

    assert str(raised.value) == f"Shape mismatch in file {paths[1]}: expected ({NY}, {NX}, 3), got ({NY}, {NX})"
    assert [e.detail for e in events] == [paths[0].name]
    assert error_log == []


@pytest.mark.parametrize("roi", [None, INTERIOR], ids=["no-roi", "roi"])
def test_an_all_rgb_stack_fails_to_unpack(tmp_path, roi):
    paths = []
    for i in range(3):
        paths.append(tmp_path / f"rgb_{i:03d}.tif")
        tifffile.imwrite(paths[-1], np.full((NY, NX, 3), i, dtype=np.uint8), photometric="rgb")
    events = []

    with pytest.raises(ValueError, match=re.escape("too many values to unpack (expected 3")):
        load_tiff_stack(paths, max_workers=1, roi=roi, progress=events.append)

    assert [e.completed for e in events] == [1, 2, 3]


# --------------------------------------------------------------------------------------
# public functions and their private twins
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("public", "twin"),
    [
        (load_tiff_stack, tiff_loader._load_tiff_stack),
        (load_fits_stack, fits_loader._load_fits_stack),
        (load_stack, stack_loader._load_stack),
    ],
    ids=["tiff", "fits", "stack"],
)
def test_public_loader_and_its_twin_take_the_same_parameters(public, twin):
    public_params = inspect.signature(public).parameters
    assert public_params == inspect.signature(twin).parameters
    roi = list(public_params.values())[-1]
    assert (roi.name, roi.kind, roi.default) == ("roi", inspect.Parameter.KEYWORD_ONLY, None)
