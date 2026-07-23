"""Unit tests for neunorm.tof.flexible_rebinner.reduce_tof_bins (flexible rebinning, Task 1)."""

import numpy as np
import pytest
import scipp as sc
from loguru import logger

from neunorm.tof.flexible_rebinner import reduce_tof_bins


def _stack(frame_values, variances=None, ny=2, nx=2, edges=None):
    """Build a (tof, y, x) stack where each frame is filled with one constant value.

    Constant-per-frame values make the reduction math easy to assert while still exercising
    the real (tof, y, x) shape, spatial coords, and — via the ``edges`` arg — a bin-edge tof
    coordinate of length N+1.
    """
    n = len(frame_values)
    vals = np.empty((n, ny, nx), dtype="float64")
    for i, v in enumerate(frame_values):
        vals[i] = v
    kwargs = {}
    if variances is not None:
        var = np.empty((n, ny, nx), dtype="float64")
        for i, v in enumerate(variances):
            var[i] = v
        kwargs["variances"] = var
    if edges is None:
        edges = np.arange(n + 1, dtype="float64") / 10.0  # e.g. [0.0, 0.1, ..., N/10]
    return sc.DataArray(
        sc.array(dims=["tof", "y", "x"], values=vals, unit="counts", **kwargs),
        coords={
            "y": sc.arange("y", ny),
            "x": sc.arange("x", nx),
            "tof": sc.array(dims=["tof"], values=np.asarray(edges, dtype="float64"), unit="s"),
        },
    )


def _capture_warnings():
    """Return (messages, remove) capturing loguru WARNING+ lines into ``messages``."""
    messages = []
    handler_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    return messages, lambda: logger.remove(handler_id)


# --------------------------------------------------------------------------------------------
# Reduction values + variance, per mode
# --------------------------------------------------------------------------------------------


def test_sum_values_and_variance():
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    out = reduce_tof_bins(data, [(0, 2), (2, 4)], reduction="sum")
    assert out.sizes == {"tof": 2, "y": 2, "x": 2}
    np.testing.assert_allclose(out.values[0], 30)  # 10 + 20
    np.testing.assert_allclose(out.values[1], 70)  # 30 + 40
    np.testing.assert_allclose(out.variances[0], 8)  # 4 + 4
    np.testing.assert_allclose(out.variances[1], 8)


def test_mean_values_and_variance():
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    out = reduce_tof_bins(data, [(0, 2), (2, 4)], reduction="mean")
    np.testing.assert_allclose(out.values[0], 15)  # mean(10, 20)
    np.testing.assert_allclose(out.values[1], 35)  # mean(30, 40)
    np.testing.assert_allclose(out.variances[0], 2)  # (4 + 4) / 2**2
    np.testing.assert_allclose(out.variances[1], 2)


def test_default_reduction_is_mean():
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    default = reduce_tof_bins(data, [(0, 2), (2, 4)])
    explicit = reduce_tof_bins(data, [(0, 2), (2, 4)], reduction="mean")
    np.testing.assert_allclose(default.values, explicit.values)
    np.testing.assert_allclose(default.variances, explicit.variances)


def test_median_values_and_approx_variance_with_warning():
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    messages, remove = _capture_warnings()
    try:
        out = reduce_tof_bins(data, [(0, 2), (2, 4)], reduction="median")
    finally:
        remove()
    # median of a 2-element bin is the mean of the two middle values
    np.testing.assert_allclose(out.values[0], 15)
    np.testing.assert_allclose(out.values[1], 35)
    # Var(median) ~= (pi/2) * Var(mean); Var(mean) = 8/4 = 2 -> (pi/2)*2
    np.testing.assert_allclose(out.variances[0], (np.pi / 2) * 2)
    np.testing.assert_allclose(out.variances[1], (np.pi / 2) * 2)
    assert any("median" in m.lower() and "approxim" in m.lower() for m in messages)


def test_median_single_frame_bin_variance_is_exact_no_warning():
    """A single-frame bin's median IS that frame, so its variance is exact (no pi/2, no warning)."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    messages, remove = _capture_warnings()
    try:
        out = reduce_tof_bins(data, [(0, 1), (1, 2), (2, 3), (3, 4)], reduction="median")
    finally:
        remove()
    np.testing.assert_allclose(out.values[:, 0, 0], [10, 20, 30, 40])
    np.testing.assert_allclose(out.variances, 4)  # exact Var, not (pi/2)*Var
    assert not any("median" in m.lower() and "approxim" in m.lower() for m in messages)


def test_variable_width_bins():
    """Bins of different widths reduce independently (1-frame bin + 3-frame bin)."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    out = reduce_tof_bins(data, [(0, 1), (1, 4)], reduction="mean")
    assert out.sizes["tof"] == 2
    np.testing.assert_allclose(out.values[0], 10)  # mean of one frame
    np.testing.assert_allclose(out.variances[0], 4)  # Var / 1**2
    np.testing.assert_allclose(out.values[1], 30)  # mean(20, 30, 40)
    np.testing.assert_allclose(out.variances[1], 12 / 9)  # (4+4+4) / 3**2


# --------------------------------------------------------------------------------------------
# Coordinate / mask preservation
# --------------------------------------------------------------------------------------------


def test_bin_edge_tof_coord_rebuilt():
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])  # edges [0, .1, .2, .3, .4]
    out = reduce_tof_bins(data, [(0, 2), (2, 4)], reduction="mean")
    # boundary edges of bins (0,2) and (2,4): indices 0, 2, 4 -> [0.0, 0.2, 0.4]
    assert out.coords["tof"].sizes["tof"] == out.sizes["tof"] + 1  # bin-edge axis (N+1)
    np.testing.assert_allclose(out.coords["tof"].values, [0.0, 0.2, 0.4])
    assert out.coords["tof"].unit == sc.Unit("s")


def test_spatial_coords_and_mask_preserved():
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    data.masks["dead"] = sc.array(dims=["y", "x"], values=[[False, True], [False, False]])
    out = reduce_tof_bins(data, [(0, 2), (2, 4)], reduction="sum")
    np.testing.assert_array_equal(out.coords["y"].values, [0, 1])
    np.testing.assert_array_equal(out.coords["x"].values, [0, 1])
    assert "dead" in out.masks
    np.testing.assert_array_equal(out.masks["dead"].values, [[False, True], [False, False]])


def test_works_without_variances():
    """Data without variances reduces fine; median emits no variance-approximation warning."""
    data = _stack([10, 20, 30, 40], variances=None)
    messages, remove = _capture_warnings()
    try:
        out = reduce_tof_bins(data, [(0, 2), (2, 4)], reduction="median")
    finally:
        remove()
    assert out.variances is None
    np.testing.assert_allclose(out.values[0], 15)
    assert not any("median" in m.lower() and "approxim" in m.lower() for m in messages)


# --------------------------------------------------------------------------------------------
# Guard / error cases
# --------------------------------------------------------------------------------------------


def test_invalid_reduction_raises():
    data = _stack([10, 20], variances=[1, 1])
    with pytest.raises(ValueError, match="reduction"):
        reduce_tof_bins(data, [(0, 2)], reduction="average")


def test_missing_tof_dim_raises():
    data = _stack([10, 20], variances=[1, 1])
    with pytest.raises(ValueError, match="not found"):
        reduce_tof_bins(data, [(0, 2)], tof_dim="spectrum")


def test_empty_bins_raises():
    data = _stack([10, 20], variances=[1, 1])
    with pytest.raises(ValueError, match="at least one"):
        reduce_tof_bins(data, [])


def test_out_of_bounds_range_raises():
    data = _stack([10, 20, 30, 40], variances=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="invalid"):
        reduce_tof_bins(data, [(0, 5)])
    with pytest.raises(ValueError, match="invalid"):
        reduce_tof_bins(data, [(2, 2)])  # non-increasing


def test_non_contiguous_bins_raise():
    data = _stack([10, 20, 30, 40], variances=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="contiguous"):
        reduce_tof_bins(data, [(0, 2), (3, 4)])  # hole at index 2


def test_non_bin_edge_coord_raises():
    """A point tof coord (length N) is rejected — a bin-edge (N+1) axis is required."""
    data = _stack([10, 20, 30, 40], variances=[1, 1, 1, 1], edges=[0.0, 0.1, 0.2, 0.3])  # len N, not N+1
    with pytest.raises(ValueError, match="bin-edge"):
        reduce_tof_bins(data, [(0, 2), (2, 4)])
