"""Unit tests for the flexible reduction path of neunorm.tof.histogram_rebinner.rebin_tof
(reduce_tof_bins, the list/int reduction dispatch, and the linear/log bin-list generators)."""

import inspect
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipp as sc
from loguru import logger

from neunorm.exporters.hdf5_writer import write_hdf5
from neunorm.tof.histogram_rebinner import (
    SPECTRA_TOF_COORD,
    linear_bin_list,
    log_bin_list,
    rebin_tof,
    reduce_tof_bins,
)


def _stack(frame_values, variances=None, ny=2, nx=2, edges=None, dtype="float64"):
    """Build a (tof, y, x) stack where each frame is filled with one constant value.

    Constant-per-frame values make the reduction math easy to assert while still exercising
    the real (tof, y, x) shape, spatial coords, and — via the ``edges`` arg — a bin-edge tof
    coordinate of length N+1. ``dtype`` allows integer fixtures (which carry no variances).
    """
    n = len(frame_values)
    vals = np.empty((n, ny, nx), dtype=dtype)
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


def test_median_two_frame_bin_variance_is_exact_no_warning():
    """A 2-frame bin's median EQUALS the mean, so its variance is exactly Var(mean) — no pi/2,
    no warning. (Independent oracle: median([10,20])=15; Var of the average of two var-4 frames
    is (4+4)/4 = 2, NOT the (pi/2)*2 the large-N formula would give.)"""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    messages, remove = _capture_warnings()
    try:
        out = reduce_tof_bins(data, [(0, 2), (2, 4)], reduction="median")
    finally:
        remove()
    np.testing.assert_allclose(out.values[0], 15)
    np.testing.assert_allclose(out.values[1], 35)
    np.testing.assert_allclose(out.variances[0], 2.0)  # exact Var(mean), NOT (pi/2)*2
    np.testing.assert_allclose(out.variances[1], 2.0)
    assert not any("median" in m.lower() and "approxim" in m.lower() for m in messages)


def test_median_large_bin_variance_uses_approximation_with_warning():
    """For n>=3 the median variance is NeuNorm's standard approximation, (pi/(2n))*mean(Var), with a
    warning that it is an estimate. The median VALUE is exact (middle of three sorted values).

    Independent oracle: n=3, all frame variances 4 -> mean(Var)=4 -> (pi/6)*4 = 2.0944, matching
    processing.reference_preparer.median_with_variance's rule for the same input."""
    data = _stack([10, 20, 30, 40, 50, 60], variances=[4, 4, 4, 4, 4, 4])
    messages, remove = _capture_warnings()
    try:
        out = reduce_tof_bins(data, [(0, 3), (3, 6)], reduction="median")
    finally:
        remove()
    np.testing.assert_allclose(out.values[0], 20)  # median(10, 20, 30) — exact
    np.testing.assert_allclose(out.values[1], 50)  # median(40, 50, 60) — exact
    np.testing.assert_allclose(out.variances, np.pi / 6 * 4)  # (pi/(2*3))*mean(Var)
    assert np.all(np.isfinite(out.variances))  # an estimate is reported, not NaN
    assert any("median" in m.lower() and "approximate" in m.lower() for m in messages)


def test_median_variance_matches_reference_preparer_convention():
    """The rebinner's n>=3 median variance must agree with NeuNorm's other median reduction
    (processing.reference_preparer.median_with_variance) for the same frames — one house rule."""
    from neunorm.processing.reference_preparer import median_with_variance

    data = _stack([10, 20, 30, 40], variances=[4, 9, 16, 25])  # heterogeneous variances
    out = reduce_tof_bins(data, [(0, 4)], reduction="median")
    expected = median_with_variance(data.rename_dims({"tof": "frame"}), "frame")
    np.testing.assert_allclose(out.values[0], expected.values)
    np.testing.assert_allclose(out.variances[0], expected.variances)


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


def test_non_contiguous_bins_accepted_dropping_uncovered_frames():
    """reduce_tof_bins accepts ordered, disjoint ranges that are NOT adjacent: the uncovered frame is
    simply never read (no gap bin — the CIS-requested silent drop)."""
    data = _stack([10, 20, 30, 40], variances=[1, 1, 1, 1])
    out = reduce_tof_bins(data, [(0, 2), (3, 4)])  # frame 2 uncovered -> dropped
    assert out.sizes["tof"] == 2
    assert not out.masks
    np.testing.assert_allclose(out.values[:, 0, 0], [15, 40])  # mean(10,20), then frame 3 alone
    np.testing.assert_allclose(out.coords["tof"].values, [0.0, 0.3, 0.4])  # left edges exact


def test_non_bin_edge_coord_raises():
    """A point tof coord (length N) is rejected — a bin-edge (N+1) axis is required."""
    data = _stack([10, 20, 30, 40], variances=[1, 1, 1, 1], edges=[0.0, 0.1, 0.2, 0.3])  # len N, not N+1
    with pytest.raises(ValueError, match="bin-edge"):
        reduce_tof_bins(data, [(0, 2), (2, 4)])


# --------------------------------------------------------------------------------------------
# rebin_tof — user-facing list input with gap-as-missing-data (Task 2)
# --------------------------------------------------------------------------------------------


def test_list_no_gap_equals_reduce_tof_bins():
    """Contiguous bins add no mask and match reduce_tof_bins on the same ranges."""
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    out = rebin_tof(data, [[0, 2], [2, 5]], reduction="mean")
    ref = reduce_tof_bins(data, [(0, 2), (2, 5)], reduction="mean")
    assert not out.masks
    np.testing.assert_allclose(out.values, ref.values)
    np.testing.assert_allclose(out.variances, ref.variances)
    np.testing.assert_allclose(out.coords["tof"].values, ref.coords["tof"].values)


def test_list_uncovered_frame_is_dropped_silently():
    """A frame covered by no range is DROPPED: one output image per requested range, no inserted
    bin, no NaN, no mask, nothing logged. (CIS/Jean's decision — an earlier revision returned an
    extra NaN 'missing data' bin, which meant 2 ranges gave 3 images.)"""
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])  # edges [0,.1,.2,.3,.4,.5]
    messages, remove = _capture_warnings()
    try:
        out = rebin_tof(data, [[0, 2], [3, 5]], reduction="mean")  # frame 2 uncovered -> dropped
    finally:
        remove()
    assert out.sizes["tof"] == 2  # one image per requested range
    assert not out.masks  # nothing flagged
    assert not np.isnan(out.values).any()  # no missing-data bin
    assert messages == []  # silent, by design
    np.testing.assert_allclose(out.values[0], 15)  # mean(10, 20)
    np.testing.assert_allclose(out.values[1], 45)  # mean(40, 50) — frame 2 (30) excluded
    np.testing.assert_allclose(out.variances, 2)
    # valid N+1 bin-edge axis; each bin's LEFT edge is exact (frame indices 0 and 3)
    assert out.coords["tof"].sizes["tof"] == out.sizes["tof"] + 1
    np.testing.assert_allclose(out.coords["tof"].values, [0.0, 0.3, 0.5])
    # the per-bin x-axis is what reveals the omission: bin 0 averages frames 0-1 only
    np.testing.assert_allclose(out.coords[SPECTRA_TOF_COORD].values, [0.05, 0.35])


def test_list_multiple_uncovered_frames_dropped_silently():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    out = rebin_tof(data, [[0, 1], [2, 3], [4, 5]], reduction="mean")  # frames 1 and 3 uncovered
    assert out.sizes["tof"] == 3  # 3 requested ranges -> 3 images
    assert not out.masks
    np.testing.assert_allclose(out.values[:, 0, 0], [10, 30, 50])
    assert not np.isnan(out.values).any()


def test_list_reduction_mode_flows_through():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    out = rebin_tof(data, [[0, 2], [3, 5]], reduction="sum")  # frame 2 uncovered -> dropped
    assert out.sizes["tof"] == 2
    np.testing.assert_allclose(out.values[0], 30)  # sum(10, 20)
    np.testing.assert_allclose(out.values[1], 90)  # sum(40, 50)
    np.testing.assert_allclose(out.variances[0], 8)  # 4 + 4


def test_list_leading_and_trailing_frames_excluded():
    """Frames before the first bin / after the last bin are dropped without a gap bin."""
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    out = rebin_tof(data, [[1, 3]], reduction="mean")  # frames 0, 3, 4 excluded
    assert out.sizes["tof"] == 1
    assert not out.masks
    np.testing.assert_allclose(out.values[0], 25)  # mean(20, 30)
    np.testing.assert_allclose(out.coords["tof"].values, [0.1, 0.3])


def test_list_overlap_rejected():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    with pytest.raises(ValueError, match="overlap"):
        rebin_tof(data, [[0, 3], [2, 5]])  # overlap on frame 2


def test_list_empty_raises():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    with pytest.raises(ValueError, match="at least one"):
        rebin_tof(data, [])


# --------------------------------------------------------------------------------------------
# rebin_tof — dedicated bin-list validation & clear errors (Task 3)
# --------------------------------------------------------------------------------------------


def test_validate_out_of_bounds():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    with pytest.raises(ValueError, match="out of bounds"):
        rebin_tof(data, [[0, 2], [3, 6]])  # stop 6 > 5 frames
    with pytest.raises(ValueError, match="out of bounds"):
        rebin_tof(data, [[-1, 2]])  # negative start


def test_validate_empty_or_reversed_bin():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    with pytest.raises(ValueError, match="empty or reversed"):
        rebin_tof(data, [[2, 2]])  # zero-width
    with pytest.raises(ValueError, match="empty or reversed"):
        rebin_tof(data, [[3, 1]])  # reversed


def test_validate_unordered_bins():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    with pytest.raises(ValueError, match="increasing order"):
        rebin_tof(data, [[3, 5], [0, 2]])  # out of order


def test_validate_malformed_structure():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    with pytest.raises(ValueError, match="pair"):
        rebin_tof(data, [[0, 2, 4]])  # three indices, not a pair
    with pytest.raises(ValueError, match="pair"):
        rebin_tof(data, [5])  # scalar entry, not a pair


def test_validate_non_integer_indices():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    with pytest.raises(ValueError, match="integers"):
        rebin_tof(data, [[0.0, 2.5]])  # float indices


def test_validate_gaps_still_allowed():
    """Validation must NOT reject a legitimate between-bin gap — the uncovered frame is simply
    dropped, yielding one image per requested range."""
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    out = rebin_tof(data, [[0, 2], [3, 5]])  # frame 2 uncovered -> dropped, no error
    assert out.sizes["tof"] == 2
    assert not out.masks


# --------------------------------------------------------------------------------------------
# Integer-dtype input paths (raw counts) — review findings F2 (median) and F1 (sum + gap)
# --------------------------------------------------------------------------------------------


def test_median_on_integer_data_promotes_no_crash():
    """median on an integer stack (raw counts, no variances) must not crash — sc.median promotes."""
    data = _stack([10, 20, 30, 40], dtype="int64")  # integer, carries no variances
    out = reduce_tof_bins(data, [(0, 2), (2, 4)], reduction="median")
    assert np.issubdtype(out.values.dtype, np.floating)
    assert out.variances is None
    np.testing.assert_allclose(out.values[0], 15)  # median(10, 20)
    np.testing.assert_allclose(out.values[1], 35)  # median(30, 40)


def test_sum_with_dropped_frame_on_integer_data_stays_integer():
    """An integer sum keeps its integer dtype even when frames are dropped: with no NaN gap bin to
    insert there is nothing forcing a float promotion."""
    data = _stack([10, 20, 30, 40, 50], dtype="int64")  # integer, no variances
    out = rebin_tof(data, [[0, 2], [3, 5]], reduction="sum")  # frame 2 uncovered -> dropped
    assert np.issubdtype(out.values.dtype, np.integer)
    assert not out.masks
    np.testing.assert_array_equal(out.values[:, 0, 0], [30, 90])  # sum(10,20), sum(40,50)


def test_sum_contiguous_on_integer_data_stays_integer():
    """Contiguous integer sum also keeps its integer dtype (no needless promote)."""
    data = _stack([10, 20, 30, 40], dtype="int64")
    out = rebin_tof(data, [[0, 2], [2, 4]], reduction="sum")  # contiguous
    assert np.issubdtype(out.values.dtype, np.integer)
    assert not out.masks
    np.testing.assert_array_equal(out.values[:, 0, 0], [30, 70])


def test_reduce_tof_bins_rejects_non_integer_indices():
    """The reduce_tof_bins primitive must also reject non-integer / bool indices (no silent int())."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    with pytest.raises(ValueError, match="integers"):
        reduce_tof_bins(data, [(0.5, 2.5)])
    with pytest.raises(ValueError, match="integers"):
        reduce_tof_bins(data, [(False, 2)])


def test_reduce_tof_bins_rejects_malformed_structure():
    """A malformed bin entry (not a [start, stop] pair) raises a clear error, not a bare unpack error."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    with pytest.raises(ValueError, match="pair"):
        reduce_tof_bins(data, [[0, 2, 4]])  # three indices
    with pytest.raises(ValueError, match="pair"):
        reduce_tof_bins(data, [5])  # scalar entry, not a pair
    with pytest.raises(ValueError, match="pair"):
        reduce_tof_bins(data, ["ab"])  # string entry: iterable of two chars, but not an index pair


# --------------------------------------------------------------------------------------------
# Task 4: per-bin mean-time (spectra_tof) coordinate + HDF5 export provenance
# --------------------------------------------------------------------------------------------


def test_spectra_tof_is_mean_of_member_left_edge_times():
    """reduce_tof_bins attaches a spectra_tof POINT coord = mean of member frames' left-edge times,
    alongside the unchanged bin-edge tof axis."""
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])  # tof edges [0,.1,.2,.3,.4,.5]
    out = reduce_tof_bins(data, [(0, 2), (2, 5)], reduction="mean")
    assert SPECTRA_TOF_COORD in out.coords
    # point coord: one value per bin (N), vs the bin-edge tof coord (N+1)
    assert out.coords[SPECTRA_TOF_COORD].sizes["tof"] == out.sizes["tof"]
    assert out.coords["tof"].sizes["tof"] == out.sizes["tof"] + 1
    # frames 0,1 left edges [0.0, 0.1] -> 0.05; frames 2,3,4 left edges [0.2, 0.3, 0.4] -> 0.3
    np.testing.assert_allclose(out.coords[SPECTRA_TOF_COORD].values, [0.05, 0.30])
    assert out.coords[SPECTRA_TOF_COORD].unit == sc.Unit("s")


def test_spectra_tof_variable_width_bins():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    out = reduce_tof_bins(data, [(0, 1), (1, 5)], reduction="mean")
    # frame 0 left edge [0.0] -> 0.0; frames 1..4 left edges [0.1,0.2,0.3,0.4] -> 0.25
    np.testing.assert_allclose(out.coords[SPECTRA_TOF_COORD].values, [0.0, 0.25])


def test_spectra_tof_reveals_dropped_frames_and_axis_monotonic():
    """With frames dropped, every spectra_tof entry stays a real time (no NaN) and the per-bin values
    are what tell the user frames were left out — this is the CIS's stated detection method. The
    bin-edge tof axis stays strictly increasing."""
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    out = rebin_tof(data, [[0, 2], [3, 5]], reduction="mean")  # frame 2 uncovered -> dropped
    st = out.coords[SPECTRA_TOF_COORD].values
    assert not np.isnan(st).any()
    np.testing.assert_allclose(st, [0.05, 0.35])  # mean left edges of frames 0-1 and 3-4
    # contrast: covering frame 2 instead shifts bin 0's time, so the axis distinguishes the cases
    covered = rebin_tof(data, [[0, 3], [3, 5]], reduction="mean")
    np.testing.assert_allclose(covered.coords[SPECTRA_TOF_COORD].values, [0.1, 0.35])
    tof = out.coords["tof"].values
    assert np.all(np.diff(tof) > 0)  # strictly increasing (monotonic)
    np.testing.assert_allclose(tof, [0.0, 0.3, 0.5])


def test_spectra_tof_round_trips_through_hdf5():
    """The updated spectra (spectra_tof) is written to and read back from HDF5 output (#192)."""
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    out = rebin_tof(data, [[0, 2], [3, 5]], reduction="mean")  # frame 2 uncovered -> dropped
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as f:
        output_path = Path(f.name)
        write_hdf5(output_path, out)
        with h5py.File(output_path, "r") as hf:
            assert "/spectra_tof" in hf
            st = hf["/spectra_tof"][()]
            assert st.shape == (2,)  # one entry per requested range
            np.testing.assert_allclose(st, [0.05, 0.35])
            assert hf["/spectra_tof"].attrs.get("units") == "s"
            np.testing.assert_allclose(hf["/tof"][()], [0.0, 0.3, 0.5])


# --------------------------------------------------------------------------------------------
# iBeatles parity: linear / log frame-index bin-list generators (Task 5)
# --------------------------------------------------------------------------------------------


def test_linear_bin_list_uniform():
    assert linear_bin_list(10, 2) == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]


def test_linear_bin_list_truncates_last_bin():
    assert linear_bin_list(10, 3) == [(0, 3), (3, 6), (6, 9), (9, 10)]


def test_linear_bin_list_step_one_is_per_frame():
    assert linear_bin_list(3, 1) == [(0, 1), (1, 2), (2, 3)]


def test_linear_bin_list_boundary_cases():
    assert linear_bin_list(1, 1) == [(0, 1)]  # single frame
    assert linear_bin_list(5, 10) == [(0, 5)]  # step > n_frames -> one truncated bin
    assert linear_bin_list(1, 3) == [(0, 1)]  # step > n_frames == 1


def test_linear_bin_list_validation():
    with pytest.raises(ValueError, match="n_frames"):
        linear_bin_list(0, 2)
    with pytest.raises(ValueError, match="step"):
        linear_bin_list(10, 0)
    with pytest.raises(ValueError, match="integer"):
        linear_bin_list(1.5, 2)  # non-integer n_frames
    with pytest.raises(ValueError, match="integer"):
        linear_bin_list(10, 1.5)  # non-integer step


def test_log_bin_list_zero_start_terminates_and_covers_frames():
    """The zero-start case must terminate (iBeatles infinite-loop guard) and give >=1 frame/bin.

    Independently derived edges for factor 0.5 with Python's round-half-to-even:
    0 ->1 ->2 ->3 ->4 ->6 ->9 ->10.
    """
    bins = log_bin_list(10, 0.5)
    assert bins == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 10)]


def test_log_bin_list_properties_various():
    """Across factors/sizes: contiguous, strictly increasing, >=1 frame/bin, covering [0, n)."""
    for n in (1, 5, 32, 100):
        for factor in (0.001, 0.01, 0.5, 2.0, 100.0):  # very small .. very large
            bins = log_bin_list(n, factor)
            assert bins[0][0] == 0
            assert bins[-1][1] == n
            assert all(stop > start for start, stop in bins)  # >= 1 frame each
            assert all(bins[i][1] == bins[i + 1][0] for i in range(len(bins) - 1))  # contiguous


def test_log_bin_list_validation():
    with pytest.raises(ValueError, match="n_frames"):
        log_bin_list(0, 0.5)
    with pytest.raises(ValueError, match="factor"):
        log_bin_list(10, 0)
    with pytest.raises(ValueError, match="integer"):
        log_bin_list(1.5, 0.5)  # non-integer n_frames
    with pytest.raises(ValueError, match="finite"):
        log_bin_list(10, float("nan"))  # non-finite factor
    with pytest.raises(ValueError, match="finite"):
        log_bin_list(10, float("inf"))  # non-finite factor


def test_generators_compose_with_rebin_tof():
    """linear/log generators feed rebin_tof end-to-end (mean) with a valid bin-edge axis."""
    data = _stack([10, 20, 30, 40, 50, 60], variances=[4, 4, 4, 4, 4, 4])

    lin = rebin_tof(data, linear_bin_list(6, 2), reduction="mean")
    assert lin.sizes["tof"] == 3
    np.testing.assert_allclose(lin.values[0], 15)  # mean(10, 20)
    assert lin.coords["tof"].sizes["tof"] == lin.sizes["tof"] + 1  # bin-edge axis (N+1)
    assert SPECTRA_TOF_COORD in lin.coords

    log_bins = log_bin_list(6, 0.5)
    log = rebin_tof(data, log_bins, reduction="mean")
    assert log.sizes["tof"] == len(log_bins)
    assert log.coords["tof"].sizes["tof"] == log.sizes["tof"] + 1
    assert SPECTRA_TOF_COORD in log.coords


# --------------------------------------------------------------------------------------------
# rebin_tof — unified entry point: int/list + reduction dispatch (Task 6, consolidated)
# --------------------------------------------------------------------------------------------


def test_rebin_tof_int_default_is_sum():
    """An integer factor with reduction=None sums (existing rebin_tof behavior; no spectra_tof)."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    out = rebin_tof(data, 2)
    assert out.sizes["tof"] == 2
    np.testing.assert_allclose(out.values[:, 0, 0], [30, 70])  # summed pairs
    assert SPECTRA_TOF_COORD not in out.coords  # sum path uses scipp.rebin, no spectra_tof


def test_rebin_tof_int_sum_explicit_matches_default():
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    np.testing.assert_allclose(rebin_tof(data, 2, reduction="sum").values, rebin_tof(data, 2).values)


def test_rebin_tof_int_mean_uses_linear_bins():
    """An integer factor with reduction='mean' averages via uniform (linear) frame-index bins."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    out = rebin_tof(data, 2, reduction="mean")
    assert out.sizes["tof"] == 2
    np.testing.assert_allclose(out.values[:, 0, 0], [15, 35])  # averaged pairs
    assert SPECTRA_TOF_COORD in out.coords


def test_rebin_tof_list_default_is_mean_with_gap():
    data = _stack([10, 20, 30, 40, 50], variances=[4, 4, 4, 4, 4])
    out = rebin_tof(data, [[0, 2], [3, 5]])  # frame 2 uncovered -> dropped
    assert out.sizes["tof"] == 2
    np.testing.assert_allclose(out.values[0, 0, 0], 15)  # mean(10, 20)
    assert not out.masks
    assert SPECTRA_TOF_COORD in out.coords


def test_rebin_tof_list_sum_reduction():
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    out = rebin_tof(data, [[0, 2], [2, 4]], reduction="sum")
    np.testing.assert_allclose(out.values[:, 0, 0], [30, 70])


def test_rebin_tof_mean_rejects_non_bins_unit_and_logarithmic():
    """mean/median reduce frame-index groups only (scipp.rebin — the time/wavelength/log sum
    machinery — cannot mean/median); those combinations must raise, not silently sum."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    with pytest.raises(ValueError, match="unit='bins'|bin list"):
        rebin_tof(data, 2, unit="time", reduction="mean")
    with pytest.raises(ValueError, match="log_bin_list|logarithmic"):
        rebin_tof(data, 2, reduction="median", logarithmic=True)


def test_rebin_tof_list_rejects_invalid_reduction():
    """A falsy-but-invalid reduction ("") must be rejected, not silently coerced to mean."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    with pytest.raises(ValueError, match="reduction"):
        rebin_tof(data, [[0, 2], [2, 4]], reduction="")


# --------------------------------------------------------------------------------------------
# Backward-compat guards for the consolidation refactor (folding flexible_rebinner into rebin_tof)
# --------------------------------------------------------------------------------------------


def test_rebin_tof_accepts_numpy_integer_factor():
    """A NumPy-integer factor must sum like a Python int, not be rejected by the unit='bins' check.

    Regression: the retired apply_tof_rebin coerced an int factor via int(spec) before calling
    rebin_tof; the consolidated pipelines pass the (np.integer-accepting) spec straight through, so
    rebin_tof itself must accept np.integer (np.int64 is not a Python int subclass)."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    np.testing.assert_allclose(rebin_tof(data, np.int64(2)).values[:, 0, 0], [30, 70])  # sum path
    np.testing.assert_allclose(rebin_tof(data, np.int64(2), reduction="mean").values[:, 0, 0], [15, 35])


def test_rebin_tof_preserves_legacy_positional_signature():
    """The released positional order (data, width, unit, logarithmic, tof_dim) must still bind after
    adding the keyword-only `reduction` parameter — i.e. `reduction` was NOT inserted before
    `logarithmic` (regression: it had been, so a bool `logarithmic` bound to `reduction` and raised)."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    # all SEVEN released positional parameters, in order:
    # data, width, unit, logarithmic, tof_dim, l_source_to_detector, detector_time_offset
    out = rebin_tof(data, 2, "bins", False, "tof", 25.0, 5000.0)
    np.testing.assert_allclose(out.values[:, 0, 0], [30, 70])  # adjacent-pair sum, unchanged
    # and `reduction` must be keyword-only AND last, so it cannot absorb a positional argument
    params = list(inspect.signature(rebin_tof).parameters.items())
    assert [n for n, p in params if p.kind is not p.KEYWORD_ONLY] == [
        "data",
        "width",
        "unit",
        "logarithmic",
        "tof_dim",
        "l_source_to_detector",
        "detector_time_offset",
    ]
    assert params[-1][0] == "reduction" and params[-1][1].kind is params[-1][1].KEYWORD_ONLY


def test_rebin_tof_accepts_tuple_bin_list():
    """A tuple-of-pairs bin list is accepted like a list (matching the Sequence-based reducers and
    the retired apply_tof_rebin, which took list|tuple)."""
    data = _stack([10, 20, 30, 40], variances=[4, 4, 4, 4])
    out = rebin_tof(data, ((0, 2), (2, 4)), reduction="sum")
    np.testing.assert_allclose(out.values[:, 0, 0], [30, 70])


def test_gapped_edges_documented_widening_vs_true_member_spans():
    """Pin the KNOWN, documented limitation of a single N+1 edge array over a non-contiguous bin list,
    against an INDEPENDENT oracle (each range's true member span), not the implementation's formula.

    N ranges over a non-contiguous frame set cannot be represented exactly by N+1 shared edges, so
    one boundary must absorb the omitted span. NeuNorm keeps every bin's LOWER edge exact (the VENUS
    spectra left-edge convention) and absorbs the omission into the preceding bin's closing edge.
    This test states both numbers explicitly so the discrepancy is visible rather than hidden behind
    an oracle that mirrors the code.
    """
    data = _stack([10, 20, 30, 40, 50, 60], variances=[1] * 6)  # frame left edges 0.0 .. 0.5, close 0.6
    ranges = [(0, 1), (2, 3), (5, 6)]  # frames 1 and 3-4 uncovered
    out = reduce_tof_bins(data, ranges, reduction="sum")

    edges = out.coords["tof"].values
    # every bin's LOWER edge equals its first member frame's left edge (exact, independent oracle)
    expected_lower = [round(0.1 * start, 10) for start, _ in ranges]
    np.testing.assert_allclose(edges[:-1], expected_lower)
    # the closing edge equals the LAST range's stop (exact)
    np.testing.assert_allclose(edges[-1], 0.1 * ranges[-1][1])

    # ... but the implied WIDTHS are wider than the true member spans wherever frames were dropped:
    implied_widths = np.diff(edges)
    true_widths = [0.1 * (stop - start) for start, stop in ranges]
    np.testing.assert_allclose(true_widths, [0.1, 0.1, 0.1])
    np.testing.assert_allclose(implied_widths, [0.2, 0.3, 0.1])  # bins 0 and 1 absorb the omissions
    assert implied_widths[0] > true_widths[0]  # the documented widening, made explicit
    assert implied_widths[2] == pytest.approx(true_widths[2])  # last bin unaffected

    # spectra_tof stays EXACT for every bin (mean of that bin's member left edges) — this is the
    # coordinate that reports what each image actually contains.
    np.testing.assert_allclose(out.coords[SPECTRA_TOF_COORD].values, [0.0, 0.2, 0.5])


def test_contiguous_edges_match_independent_frame_boundaries():
    """Backward-compat for the common (contiguous) case, checked against frame boundaries computed
    from the fixture rather than against the implementation's edge formula."""
    data = _stack([10, 20, 30, 40, 50, 60], variances=[1] * 6)  # left edges 0.0..0.5, close 0.6
    ranges = [(0, 2), (2, 5), (5, 6)]
    out = reduce_tof_bins(data, ranges, reduction="mean")
    # contiguous tiling -> every edge is a real frame boundary and widths equal member counts
    expected = [0.0, 0.2, 0.5, 0.6]
    np.testing.assert_allclose(out.coords["tof"].values, expected)
    np.testing.assert_allclose(np.diff(out.coords["tof"].values), [0.1 * (b - a) for a, b in ranges])
