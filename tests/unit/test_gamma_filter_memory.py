"""Peak memory, input isolation and bitwise output of `apply_gamma_filter`.

The output is compared bit for bit against `_reference_gamma_filter`, a whole-array implementation of
the same filter in which every intermediate is a separate full-size array.
"""

import tracemalloc

import numpy as np
import pytest
import scipp as sc
from scipy import ndimage as ndi

from neunorm.filters import gamma_filter
from neunorm.filters.gamma_filter import apply_gamma_filter


def _reference_gamma_filter(data, threshold_sigma=5.0, kernel_size=3, preserve_variance=True):
    """Gamma filter computed with one full-size array per intermediate and the output built by `sc.array`."""
    ndim = data.data.ndim
    size = [1] * ndim
    dims = data.dims
    size[dims.index("y")] = kernel_size
    size[dims.index("x")] = kernel_size
    footprint = np.full(size, True, dtype=bool)
    footprint[tuple((s // 2) for s in size)] = False

    values = data.data.values
    kernel = footprint.astype(float)
    count = kernel.sum()
    local_mean = ndi.convolve(values, kernel, mode="nearest") / count
    local_mean_sq = ndi.convolve(values**2, kernel, mode="nearest") / count
    local_var = np.clip(local_mean_sq - local_mean**2, 0, None)
    local_std = np.sqrt(local_var)
    local_median = ndi.median_filter(values, footprint=footprint, mode="nearest")
    local_threshold = local_median + threshold_sigma * local_std
    outlier_mask = values > local_threshold
    outlier_count = np.sum(outlier_mask)

    filtered_values = values.copy()
    filtered_values[outlier_mask] = local_median[outlier_mask]

    input_variances = data.data.variances
    filtered_variances = input_variances.copy() if input_variances is not None else None
    if not preserve_variance and input_variances is not None and outlier_count > 0:
        input_variances_padded = np.pad(input_variances, [(s // 2, s // 2) for s in size], mode="edge")
        for idx in np.ndindex(outlier_mask.shape):
            if outlier_mask[idx]:
                neighbor_indices = tuple(slice(i, i + s) for i, s in zip(idx, size))
                neighbor_variances = input_variances_padded[neighbor_indices][footprint]
                mean_variance = neighbor_variances.mean()
                filtered_variances[idx] = (np.pi / (2 * len(neighbor_variances))) * mean_variance

    out = data.copy(deep=False)
    out.data = sc.array(dims=dims, values=filtered_values, variances=filtered_variances, unit=data.unit)
    return out


def _bits(array):
    """The array's raw bit patterns, so that -0.0 differs from 0.0 and NaN payloads are compared."""
    array = np.ascontiguousarray(array)
    if array.dtype.kind == "f":
        return array.view(f"u{array.dtype.itemsize}")
    return array


def _stack(shape, dtype, seed, *, dims=None, variances=True, hostile=False, counts=3000):
    """Poisson counts with gamma spikes, coords and a dead-pixel mask.

    ``hostile`` adds a constant patch (zero local deviation) and negative values; for float data also
    NaN, ±inf, -0.0, values whose square overflows and subnormals.
    """
    rng = np.random.default_rng(seed)
    values = rng.poisson(counts, size=shape).astype(dtype)
    flat = values.reshape(-1)
    spikes = rng.choice(flat.size, max(1, flat.size // 500), replace=False)
    flat[spikes] += rng.uniform(5e3, 5e4, spikes.size).astype(dtype)
    if hostile:
        values[tuple(slice(1, 6) for _ in shape)] = 1234
        flat[rng.choice(flat.size, 20, replace=False)] *= -1
        if values.dtype.kind == "f":
            for value, n in ((np.nan, 7), (np.inf, 3), (-np.inf, 3), (-0.0, 20)):
                flat[rng.choice(flat.size, n, replace=False)] = value
            flat[rng.choice(flat.size, 5, replace=False)] = np.finfo(values.dtype).max / 4
            flat[rng.choice(flat.size, 5, replace=False)] = np.finfo(values.dtype).tiny / 8
    if dims is None:
        dims = {2: ["y", "x"], 3: ["N_image", "y", "x"], 4: ["N_image", "t", "y", "x"]}[len(shape)]
    data = sc.DataArray(
        sc.array(dims=dims, values=values, variances=np.abs(values) + 1 if variances else None, unit="counts"),
        coords={d: sc.arange(d, n, unit=None) for d, n in zip(dims, shape)},
    )
    spatial = [d for d in dims if d in ("y", "x")]
    data.masks["dead_pixels"] = sc.array(dims=spatial, values=rng.random([data.sizes[d] for d in spatial]) < 0.01)
    return data


def _transposed():
    return _stack((4, 70, 90), np.float32, 9, hostile=True).transpose(["x", "N_image", "y"])


def _sliced():
    return _stack((6, 80, 100), np.float32, 10, hostile=True)["N_image", 1:4]["x", 5:70]


#: The hostile fixtures' NaN, inf and overflowing squares make numpy emit RuntimeWarnings.
_IGNORE_FLOAT_WARNINGS = pytest.mark.filterwarnings("ignore::RuntimeWarning")

CASES = {
    "float32 stack": (lambda: _stack((4, 60, 90), np.float32, 1), {}),
    "nan inf negative constant patch": (lambda: _stack((3, 50, 70), np.float32, 2, hostile=True), {}),
    "kernel 5": (lambda: _stack((3, 60, 80), np.float32, 3, hostile=True), {"kernel_size": 5, "threshold_sigma": 2.5}),
    "kernel 7 sigma 0": (
        lambda: _stack((2, 50, 60), np.float32, 4, hostile=True),
        {"kernel_size": 7, "threshold_sigma": 0.0},
    ),
    "recomputed variances": (lambda: _stack((3, 40, 50), np.float32, 5, hostile=True), {"preserve_variance": False}),
    "float64 2-D image": (lambda: _stack((80, 110), np.float64, 6, hostile=True), {"preserve_variance": False}),
    "int64 without variances": (lambda: _stack((3, 40, 50), np.int64, 7, variances=False, hostile=True), {}),
    "4-D": (lambda: _stack((2, 3, 30, 40), np.float32, 8, hostile=True), {"threshold_sigma": 3}),
    "transposed view": (_transposed, {}),
    "sliced view": (_sliced, {"preserve_variance": False}),
    "more than one default block": (lambda: _stack((2, 600, 1000), np.float32, 11), {}),
    # A threshold one deviation above the median puts many pixels within rounding of it, so these two
    # cases resolve the local deviation to its last bits.
    "sigma 1 bright counts": (lambda: _stack((3, 80, 100), np.float32, 12, counts=30000), {"threshold_sigma": 1.0}),
    "sigma 0.5 recomputed variances": (
        lambda: _stack((2, 60, 70), np.float32, 13, counts=30000),
        {"threshold_sigma": 0.5, "preserve_variance": False},
    ),
}


@_IGNORE_FLOAT_WARNINGS
@pytest.mark.parametrize("block_size", [None, 4099], ids=["default block", "block 4099"])
@pytest.mark.parametrize("case", list(CASES))
def test_gamma_filter_output_is_bitwise_identical_to_whole_array_reference(monkeypatch, case, block_size):
    if block_size is not None:
        monkeypatch.setattr(gamma_filter, "_BLOCK_SIZE", block_size)
    make, kwargs = CASES[case]
    data = make()

    out = apply_gamma_filter(data, **kwargs)
    ref = _reference_gamma_filter(data, **kwargs)

    assert out.values.dtype == ref.values.dtype
    np.testing.assert_array_equal(_bits(out.values), _bits(ref.values))
    if ref.variances is None:
        assert out.variances is None
    else:
        np.testing.assert_array_equal(_bits(out.variances), _bits(ref.variances))
    assert sc.identical(out, ref, equal_nan=True)
    assert (_bits(out.values) != _bits(data.values)).any(), "fixture must contain outliers"


@_IGNORE_FLOAT_WARNINGS
@pytest.mark.parametrize("preserve_variance", [True, False])
def test_gamma_filter_leaves_input_unchanged_and_unshared(preserve_variance):
    data = _stack((3, 40, 60), np.float32, 21, hostile=True)
    values = _bits(data.values).copy()
    variances = _bits(data.variances).copy()

    out = apply_gamma_filter(data, preserve_variance=preserve_variance)

    np.testing.assert_array_equal(_bits(data.values), values)
    np.testing.assert_array_equal(_bits(data.variances), variances)
    assert not np.shares_memory(out.values, data.values)
    assert not np.shares_memory(out.variances, data.variances)
    replaced = _bits(out.values) != values
    assert replaced.any()
    if not preserve_variance:
        assert (_bits(out.variances) != variances)[replaced].all()


@pytest.mark.parametrize("preserve_variance", [True, False])
def test_gamma_filter_peak_stays_within_a_few_stacks(monkeypatch, preserve_variance):
    """Peak memory during the call relative to the values array.

    tracemalloc counts numpy's allocations but not scipp's. The output copy of the values and
    variances is made by ``sc.Variable.copy``, which is wrapped to read the traced memory as the copy
    starts; the copy's size is added to everything traced from then on. The block is set small so
    that one block of the mean is negligible next to the stack.
    """
    monkeypatch.setattr(gamma_filter, "_BLOCK_SIZE", 4096, raising=False)
    data = _stack((8, 96, 200), np.float32, 31)
    stack_bytes = data.values.nbytes
    copy_bytes = data.values.nbytes + data.variances.nbytes
    variable_copy = sc.Variable.copy
    at_copy = []

    def traced_copy(self, *args, **kwargs):
        at_copy.append(tracemalloc.get_traced_memory())
        tracemalloc.reset_peak()
        return variable_copy(self, *args, **kwargs)

    monkeypatch.setattr(sc.Variable, "copy", traced_copy)
    tracemalloc.start()
    try:
        out = apply_gamma_filter(data, preserve_variance=preserve_variance)
        _, peak_after_copy = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert (_bits(out.values) != _bits(data.values)).any()
    assert len(at_copy) == 1, f"expected one copy of the data, got {len(at_copy)}"
    live_at_copy, peak_before_copy = at_copy[0]
    assert live_at_copy / stack_bytes < 0.5, (
        f"{live_at_copy / stack_bytes:.2f} times the values array is live while the output is copied"
    )
    peak = max(peak_before_copy, peak_after_copy + copy_bytes)
    assert peak / stack_bytes < 4.0, f"peak is {peak / stack_bytes:.2f} times the values array"
