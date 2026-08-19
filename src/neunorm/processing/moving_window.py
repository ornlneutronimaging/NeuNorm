"""
A mask-aware moving window over named dimensions.

Each pixel is replaced by the average — or, with ``kind="sum"``, the total — of the pixels in a box
around it. Ported from iBeatles' ``moving_average`` (``core/processing/moving_average.py``), whose
box filter is ``scipy.ndimage.convolve(data, np.ones(kernel) / kernel.sum())`` with scipy's default
``mode="reflect"``. That behaviour is the specification here; this module reproduces it and deviates
in exactly three places, each because iBeatles' choice cannot be carried into NeuNorm unchanged:

**Mask awareness.** iBeatles filters a bare numpy array. A dead or hot pixel inside the window drags
its whole neighbourhood down — one dead pixel in an otherwise uniform 100-count field leaves a 3x3
kernel reading 88.89 across ``k**2`` pixels. NeuNorm detects those pixels precisely so they can be
excluded, so the window here is a *normalized convolution*: the values and a good-pixel indicator are
filtered separately and divided, which returns the true level and contaminates nothing.

**Variance propagation.** iBeatles carries no uncertainties. Every NeuNorm operation propagates
variances, so this one does too: with weights ``w_p`` summing to one over the usable pixels,
``Var_out = sum_p w_p**2 Var_p``. The weights are per distinct SOURCE pixel, which matters at the
frame edge: a boundary mode makes the window read some real pixels more than once, and a pixel read
``m`` times carries weight ``m / k`` rather than ``1 / k``. Summing over window slots instead would
understate the reported variance by up to 2.78x at a mirrored 3x3 corner — 1.67x in sigma, which is
what a user reads off an error bar. This assumes the input
pixels are independent, which holds at the point the pipeline applies the filter (before
normalization, with no earlier smoothing) — it is not true of the *output*, whose neighbouring pixels
are correlated by construction.

**Sizes addressed by dim name.** iBeatles indexes positionally as ``(y, x, lambda)``. NeuNorm names
its axes, and the event path produces ``(tof, x, y)`` — x before y — so a positional tuple would
silently transpose the two spatial axes. Sizes are given as ``{dim: length}`` and any dim left out
gets a length of 1.

Notes
-----
The window trades spatial resolution for per-pixel precision at a fixed rate: a ``k x k`` kernel
improves per-pixel precision by a factor ``k`` and coarsens resolution by the same factor. The array
keeps its shape, so the result presents as full resolution while carrying roughly ``1 / k**2`` as
many independent values. ``docs/moving_window.md`` has the measured tables.
"""

from typing import Literal, Mapping, Optional

import numpy as np
import scipp as sc
from scipy.ndimage import correlate1d, uniform_filter

from neunorm.utils.masks import combined_mask
from neunorm.utils.progress import STAGE_MOVING_WINDOW, ProgressLike, resolve_progress

#: What the window does with the pixels it collects: their mean, or their total.
MovingWindowKind = Literal["average", "sum"]

#: Edge policies, passed straight through to :func:`scipy.ndimage.uniform_filter`. The default
#: mirrors the frame edge, which is what iBeatles does; a mirrored and a shrink-to-real-pixels edge
#: differ only within a ``k // 2`` border, and the interior is identical either way.
#:
#: The two constant modes are the only ones whose meaning depends on whether a mask applies — see
#: the ``mode`` parameter of :func:`moving_window`. Every other mode reads only real pixels, so the
#: weight pass returns exactly 1 and changes nothing.
EDGE_MODES = ("reflect", "constant", "nearest", "mirror", "wrap", "grid-mirror", "grid-constant", "grid-wrap")


def moving_window_step_count(data: sc.DataArray, masks: Optional[Mapping[str, sc.Variable]] = None) -> int:
    """How many progress steps :func:`moving_window` reports for these arguments.

    Declared here rather than at the call site so the two cannot drift apart, the same arrangement
    :func:`~neunorm.processing.normalizer.normalize_step_count` uses.

    Parameters
    ----------
    data : sc.DataArray
        The array that will be filtered.
    masks : Mapping[str, sc.Variable], optional
        The masks that will be passed to :func:`moving_window`, or ``None`` to use ``data.masks``.

    Returns
    -------
    int
        The number of counted steps: the value pass, plus a weight pass when any mask applies, plus
        a variance pass when the data carries variances.
    """
    n_steps = 1  # the value pass always runs
    if _applicable_masks(data, data.masks if masks is None else masks):
        n_steps += 1
    if data.variances is not None:
        n_steps += 1
    return n_steps


def _applicable_masks(data: sc.DataArray, masks: Mapping[str, sc.Variable]) -> list:
    """The masks that can be broadcast onto ``data``, i.e. whose dims are all dims ``data`` has.

    A mask carrying a dim the data does not — a per-frame spectral mask on an array whose spectral
    axis has already been reduced away — cannot select pixels here and is skipped rather than raising:
    it is meaningful to its own array, just not to this one.
    """
    return [mask for mask in masks.values() if set(mask.dims) <= set(data.dims)]


def _usable_mask(data: sc.DataArray, masks: Mapping[str, sc.Variable]) -> Optional[np.ndarray]:
    """Boolean array, ``True`` where a pixel may be used, in ``data``'s shape.

    Returns ``None`` when no mask applies, which is the caller's signal to take the plain
    (single-pass, exactly-scipy) path.
    """
    bad = combined_mask(data.sizes, masks, skip_mismatched=True)
    return None if bad is None else ~bad


def _slot_sources(length: int, axis_size: int, mode: str, origin: int):
    """Which source index each of the ``length`` window slots reads, for every output position.

    Recovered from scipy itself rather than by re-deriving its index arithmetic: correlating a ramp
    with a one-hot kernel returns, at each output, the value of the source pixel that slot reads. The
    ramp starts at 1 so that a slot filled by ``cval`` under the constant modes comes back as 0 and is
    reported as invalid, which index arithmetic alone could not distinguish from source index 0.

    Returns ``(sources, valid)``, each ``(length, axis_size)``.
    """
    ramp = np.arange(1, axis_size + 1, dtype=np.float64)
    sources = np.empty((length, axis_size), dtype=np.int64)
    valid = np.empty((length, axis_size), dtype=bool)
    for slot in range(length):
        one_hot = np.zeros(length, dtype=np.float64)
        one_hot[slot] = 1.0
        probe = correlate1d(ramp, one_hot, mode=mode, origin=origin, cval=0.0)
        valid[slot] = probe > 0.5
        sources[slot] = np.rint(probe).astype(np.int64) - 1
    return sources, valid


def _summed_with_squared_multiplicity(values: np.ndarray, axis: int, length: int, mode: str, origin: int):
    """``sum_p m**2 values[p]`` along one axis, where ``m`` counts the slots that read source ``p``.

    A boundary mode duplicates real pixels: at a mirrored corner a 3x3 window reads the corner pixel
    four times, its two neighbours twice each and the diagonal once. The VALUE is unaffected — it is
    linear, and a plain box filter already sums the duplicates correctly — but a variance is not.
    ``Var(sum_p w_p x_p) = sum_p w_p**2 Var_p`` needs the weight on each distinct SOURCE pixel, so a
    pixel read ``m`` times contributes ``m**2``, not ``m``. Summing over slots instead understates the
    reported variance wherever the window overhangs the frame — measured 2.78x low at a 3x3 corner,
    which is 1.67x in sigma.

    Computed as the plain slot sum plus a correction, because ``m`` is 1 everywhere except within
    ``length // 2`` of an edge: for each pair of slots that read the SAME source, that source's
    coefficient gains 2 (summing ``m**2 - m = 2 * (number of unordered coincident pairs)``). The
    correction touches only the few border positions where a coincidence occurs.
    """
    total = correlate1d(values, np.ones(length, dtype=values.dtype), axis=axis, mode=mode, origin=origin)
    sources, valid = _slot_sources(length, values.shape[axis], mode, origin)
    for first in range(length):
        for second in range(first + 1, length):
            coincident = valid[first] & valid[second] & (sources[first] == sources[second])
            positions = np.flatnonzero(coincident)
            if positions.size == 0:
                continue
            target = [slice(None)] * values.ndim
            target[axis] = positions
            total[tuple(target)] += 2.0 * np.take(values, sources[first][positions], axis=axis)
    return total


def _squared_multiplicity_sum(values: np.ndarray, window, mode: str, origin) -> np.ndarray:
    """``sum_p m**2 values[p]`` over the whole window, applied axis by axis.

    Separable, like the filter itself: the multiplicity of a source pixel factorizes across axes, so
    squaring it does too.
    """
    result = values
    for axis, length in enumerate(window):
        if length == 1:
            continue
        result = _summed_with_squared_multiplicity(result, axis, length, mode, origin[axis])
    return result


def _validate(data: sc.DataArray, sizes: Mapping[str, int], kind: str, mode: str) -> None:
    """Reject the arguments that would otherwise fail deep inside scipy, or not fail at all."""
    if kind not in ("average", "sum"):
        raise ValueError(f"kind must be 'average' or 'sum', got {kind!r}")
    if mode not in EDGE_MODES:
        raise ValueError(f"mode must be one of {', '.join(EDGE_MODES)}; got {mode!r}")
    if not sizes:
        raise ValueError("sizes must name at least one dimension, e.g. {'x': 3, 'y': 3}")
    for dim, size in sizes.items():
        if dim not in data.dims:
            raise ValueError(
                f"moving-window size given for dimension {dim!r}, which the data does not have "
                f"(its dims are {', '.join(data.dims)}). Sizes are addressed by dimension NAME, so a "
                "kernel written for one detector's axis order still applies to another's."
            )
        # A bare `isinstance(size, int)` would accept True, which numpy then treats as 1.
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise ValueError(f"moving-window size for {dim!r} must be an integer, got {size!r}")
        if size < 1:
            raise ValueError(
                f"moving-window size for {dim!r} must be >= 1, got {size}. A window of 1 leaves that "
                "dimension untouched."
            )


def moving_window(
    data: sc.DataArray,
    sizes: Mapping[str, int],
    *,
    kind: MovingWindowKind = "average",
    mode: str = "reflect",
    masks: Optional[Mapping[str, sc.Variable]] = None,
    progress: ProgressLike = False,
    stage: str = STAGE_MOVING_WINDOW,
) -> sc.DataArray:
    """Replace each pixel by the average (or total) of a box of pixels around it.

    Masked pixels are excluded from both the value and the weight, so a dead or hot pixel neither
    contributes to nor depresses its neighbours. The masks themselves are carried through untouched:
    a dead pixel is still a dead pixel after filtering, whatever value the window computed there.

    Parameters
    ----------
    data : sc.DataArray
        The array to filter. Integer data is promoted to ``float64``; a float dtype is preserved, so
        a ``float32`` stack stays ``float32`` and does not double in memory.
    sizes : Mapping[str, int]
        Window length per dimension, addressed by dim name, e.g. ``{"x": 3, "y": 3}``. Any dim not
        named gets a length of 1. Even lengths are accepted, as iBeatles accepts them: a window with
        no centre pixel shifts the response by exactly -0.50 px along that axis.
    kind : {"average", "sum"}, optional
        ``"average"`` divides by the number of pixels collected; ``"sum"`` does not. Applied before
        normalization the two are indistinguishable in the result, because the kernel count cancels
        in the sample/open-beam ratio.

        Where a mask applies, ``"sum"`` is the mask-aware mean scaled by the NOMINAL window size —
        ``k * average`` — and not the bare total of the pixels that survived. The two differ as soon
        as one pixel is masked. Scaling is deliberate: a bare total falls with every masked pixel, so
        it would not cancel in the sample/open-beam ratio, and a moving sum would stop agreeing with
        a moving average exactly where the detector is worst.
    mode : str, optional
        Edge policy, passed to :func:`scipy.ndimage.uniform_filter`. Defaults to mirroring the frame
        edge, as iBeatles does. See :data:`EDGE_MODES`.

        The two **constant** modes read differently depending on whether a mask applies, and the
        difference is worth knowing before choosing one. With no mask they are scipy's: out-of-frame
        slots contribute ``cval`` (0), so a pixel at the edge is pulled toward zero. With a mask the
        normalized convolution divides by the weight actually collected, so those same out-of-frame
        slots drop out and the edge pixel is the mean of the real pixels instead. Every other mode —
        including the default — reads only real pixels either way, so nothing changes for them.
    masks : Mapping[str, sc.Variable], optional
        The masks that decide which pixels are usable. Defaults to ``data.masks``. Passed explicitly
        when filtering an array that does not carry the masks itself — the pipelines detect dead and
        hot pixels from the open beam but attach them to the sample, and the same bad detector pixels
        must be excluded from both stacks.
    progress : bool or callable, optional
        Progress reporting, off by default. This function has no item axis, so it reports named
        whole-array steps; :func:`moving_window_step_count` gives the total in advance.
    stage : str, optional
        Stage label the events carry. Defaults to ``STAGE_MOVING_WINDOW``.

    Returns
    -------
    sc.DataArray
        The filtered array, with the same dims, shape, unit, coords and masks. Returned unchanged
        (the same object) when every size is 1, which is the identity window.

    Raises
    ------
    ValueError
        If ``kind`` or ``mode`` is not recognized, if ``sizes`` is empty, if it names a dimension the
        data does not have, or if a size is not an integer >= 1.

    Notes
    -----
    The variance follows the weights: with ``w_p = m_p g_p / G`` over the distinct source pixels the
    window reads — ``g`` marking the usable ones, ``m_p`` how many slots read pixel ``p`` once the
    boundary mode has been applied, and ``G = sum_p m_p g_p`` — ``Var_out = sum_p w_p**2 Var_p``. In
    the interior every ``m_p`` is 1 and this reduces to ``sum(Var) / k**2`` for an average over ``k``
    unmasked pixels and to ``sum(Var)`` for a sum. That is correct per pixel and assumes
    the *inputs* are independent. The outputs are not: neighbouring pixels share window members and
    are strongly correlated (+0.67 at 3x3, +0.81 at 5x5), which no later reduction can undo because
    scipp carries no covariance. ``docs/moving_window.md`` records the measurements.

    A window in which every pixel is masked has nothing to average. Those pixels keep their input
    value and variance rather than becoming ``NaN``; they are masked in the output either way.

    Examples
    --------
    >>> import numpy as np, scipp as sc
    >>> data = sc.DataArray(sc.array(dims=["y", "x"], values=np.full((5, 5), 100.0), unit="counts"))
    >>> smoothed = moving_window(data, {"x": 3, "y": 3})
    >>> float(smoothed.values[2, 2])
    100.0
    >>> float(moving_window(data, {"x": 3, "y": 3}, kind="sum").values[2, 2])
    900.0
    """
    _validate(data, sizes, kind, mode)

    window = [int(sizes.get(dim, 1)) for dim in data.dims]
    n_kernel = int(np.prod(window))
    if n_kernel == 1:
        # The identity window. Returned as the same object, as ``rebin_spatial`` does for factor=1.
        return data

    # An even window has no centre pixel, and the two scipy entry points disagree about which way it
    # leans: iBeatles' ``convolve`` puts the extra pixel BEFORE the centre (a -0.50 px shift),
    # ``uniform_filter`` puts it after (+0.50). Offsetting the origin on the even axes reproduces
    # iBeatles exactly while keeping the separable — and measurably faster — filter.
    origin = [0 if length % 2 else -1 for length in window]

    values = data.values
    # Counts may arrive as an integer dtype; an average of integers is not an integer. A float dtype
    # is kept as it is, so a float32 stack is filtered in float32 and does not double in memory.
    float_dtype = values.dtype if values.dtype.kind == "f" else np.dtype(np.float64)
    source = values.astype(float_dtype, copy=False)

    usable = _usable_mask(data, data.masks if masks is None else masks)

    with resolve_progress(progress, stage, total=moving_window_step_count(data, masks)) as report:
        if usable is None:
            # No mask applies: one pass, and bit-for-bit what scipy's uniform_filter gives, which is
            # what iBeatles' normalized box convolution computes.
            report(detail="moving window: values")
            filtered = uniform_filter(source, size=window, mode=mode, origin=origin)
            usable_count = None
            starved = None
        else:
            report(detail="moving window: usable-pixel weights")
            # uniform_filter divides by the kernel size, so this is the FRACTION of the window that
            # is usable. The same divisor sits in the numerator and cancels in the quotient.
            weight = uniform_filter(usable.astype(float_dtype), size=window, mode=mode, origin=origin)
            report(detail="moving window: values")
            # SELECT rather than multiply by an indicator: IEEE makes NaN * 0 and inf * 0 NaN, so a
            # masked non-finite value — exactly the kind of pixel a mask exists to remove — would
            # otherwise poison every output within a window of it.
            selected = np.where(usable, source, float_dtype.type(0))
            numerator = uniform_filter(selected, size=window, mode=mode, origin=origin)
            # A window with G usable pixels has weight G / n_kernel, and G is a non-negative INTEGER,
            # so a non-starved window has weight >= 1 / n_kernel. Testing `weight <= 0` instead
            # assumed the weight pass returns exact zero for a fully masked window; scipy's separable
            # running sum leaves a residue of order 1e-16 on float64, which slipped through and made
            # the division a residue-by-residue quotient — measured negative counts and NEGATIVE
            # variances. Half a pixel's weight separates the two cases by 16 orders of magnitude.
            starved = weight < 0.5 / n_kernel
            usable_count = n_kernel * np.where(starved, 1.0, weight)
            filtered = numerator / np.where(starved, float_dtype.type(1), weight)

        out_variances = None
        if data.variances is not None:
            report(detail="moving window: variances")
            source_var = data.variances.astype(float_dtype, copy=False)
            if usable is None:
                # Var(sum_p w_p x_p) = sum_p w_p**2 Var_p with w_p = m_p / n_kernel.
                out_variances = _squared_multiplicity_sum(source_var, window, mode, origin) / (n_kernel * n_kernel)
            else:
                # Same, with w_p = m_p g_p / G and G = sum_p m_p g_p usable pixels in the window.
                selected_var = np.where(usable, source_var, float_dtype.type(0))
                out_variances = _squared_multiplicity_sum(selected_var, window, mode, origin) / (
                    usable_count * usable_count
                )

    if kind == "sum":
        # The same window without the divisor: a sum over k pixels is k times their mean, and a
        # variance scales by the square.
        filtered = filtered * n_kernel
        if out_variances is not None:
            out_variances = out_variances * n_kernel * n_kernel

    if starved is not None and starved.any():
        # Nothing usable anywhere in the window. Leave the pixel as it came in rather than inventing
        # a NaN; it is masked in the output regardless.
        filtered[starved] = source[starved]
        if out_variances is not None:
            out_variances[starved] = data.variances.astype(float_dtype, copy=False)[starved]

    # Shallow copy so coords, masks, name and coord alignment come across exactly; only the data
    # variable is replaced, so the caller's array is left untouched.
    result = data.copy(deep=False)
    result.data = sc.array(dims=data.dims, values=filtered, variances=out_variances, unit=data.unit)
    return result
