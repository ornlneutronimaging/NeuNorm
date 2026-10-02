"""Helpers shared by the TIFF and FITS stack loaders."""

import numpy as np
import pytest
import scipp as sc

import neunorm.loaders._frame_stack as frame_stack
from neunorm.loaders._frame_stack import allocate_stack, format_bytes, has_negative, variances_note


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([-1.0], dtype=np.float32), True),
        (np.array([np.nan, -1.0], dtype=np.float32), True),
        (np.array([-0.0, np.nan], dtype=np.float32), False),
        (np.array([-1e-300], dtype=np.float64), False),
        (np.array([-1], dtype=np.int16), True),
        (np.array([np.iinfo(np.uint16).max], dtype=np.uint16), False),
    ],
    ids=["f32-negative", "f32-nan-then-negative", "f32-negative-zero-and-nan", "f64-below-f32", "int16", "uint16-max"],
)
def test_has_negative(values, expected):
    assert has_negative(values) is expected


@pytest.mark.parametrize(
    ("n_bytes", "expected"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.0 KiB"),
        (2**20, "1.0 MiB"),
        (3 * 2**29, "1.5 GiB"),
        (3 * 2**40, "3.0 TiB"),
    ],
)
def test_format_bytes(n_bytes, expected):
    assert format_bytes(n_bytes) == expected


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 2, 3), "attaching variances to 1 frame of 3 x 2 px (24 B)"),
        ((4, 2, 3), "attaching variances to 4 frames of 3 x 2 px (96 B)"),
    ],
    ids=["one-frame", "several-frames"],
)
def test_variances_note(shape, expected):
    data = sc.empty(dims=["N_image", "y", "x"], shape=list(shape), dtype="float32")
    assert variances_note(data) == expected


def test_allocate_stack_is_float32_counts_with_variances():
    data = allocate_stack(["TOF", "y", "x"], [2, 3, 4])

    assert data.dims == ("TOF", "y", "x")
    assert data.shape == (2, 3, 4)
    assert data.dtype == sc.DType.float32
    assert data.unit == sc.units.counts
    assert data.variances is not None


def test_allocate_stack_names_shape_and_size_when_out_of_memory(monkeypatch):
    original = MemoryError("std::bad_alloc")

    def fail(**_kwargs):
        raise original

    monkeypatch.setattr(frame_stack.sc, "empty", fail)

    with pytest.raises(MemoryError) as raised:
        allocate_stack(["N_image", "y", "x"], [500, 326, 1704])

    assert (
        str(raised.value) == "Unable to allocate 2.1 GiB for a float32 stack of shape (500, 326, 1704) with variances"
    )
    assert raised.value.__cause__ is original
