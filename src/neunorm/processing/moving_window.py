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
variances, so this one does too: with weights ``w_j`` summing to one over the usable pixels,
``Var_out = sum_j w_j**2 Var_j``. This assumes the input pixels are independent, which holds at the
point the pipeline applies the filter (before normalization, with no earlier smoothing) — it is not
true of the *output*, whose neighbouring pixels are correlated by construction.

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
from scipy.ndimage import uniform_filter

from neunorm.utils.progress import STAGE_MOVING_WINDOW, ProgressLike, resolve_progress

#: What the window does with the pixels it collects: their mean, or their total.
MovingWindowKind = Literal["average", "sum"]

#: Edge policies, passed straight through to :func:`scipy.ndimage.uniform_filter`. The default
#: mirrors the frame edge, which is what iBeatles does; a mirrored and a shrink-to-real-pixels edge
#: differ only within a ``k // 2`` border, and the interior is identical either way.
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


def _good_indicator(data: sc.DataArray, masks: Mapping[str, sc.Variable], dtype: np.dtype) -> Optional[np.ndarray]:
    """1.0 where a pixel may be used, 0.0 where any mask flags it, in ``data``'s shape.

    Returns ``None`` when no mask applies, which is the caller's signal to take the plain
    (single-pass, exactly-scipy) path.
    """
    applicable = _applicable_masks(data, masks)
    if not applicable:
        return None
    bad = np.zeros(data.shape, dtype=bool)
    for mask in applicable:
        # Broadcast by dim NAME: a 2-D (x, y) dead-pixel mask expands correctly over a (tof, x, y)
        # stack whatever the dim order is.
        bad |= sc.broadcast(mask, sizes=data.sizes).values
    return (~bad).astype(dtype)


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
    mode : str, optional
        Edge policy, passed to :func:`scipy.ndimage.uniform_filter`. Defaults to mirroring the frame
        edge, as iBeatles does. See :data:`EDGE_MODES`.
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
    The variance follows the weights: with ``w_j = g_j / sum(g)`` over the good pixels ``g`` in the
    window, ``Var_out = sum_j w_j**2 Var_j``, which reduces to ``sum(Var) / k**2`` for an average
    over ``k`` unmasked pixels and to ``sum(Var)`` for a sum. That is correct per pixel and assumes
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

    good = _good_indicator(data, data.masks if masks is None else masks, float_dtype)

    with resolve_progress(progress, stage, total=moving_window_step_count(data, masks)) as report:
        if good is None:
            # No mask applies: one pass, and bit-for-bit what scipy's uniform_filter gives, which is
            # what iBeatles' normalized box convolution computes.
            report(detail="moving window: values")
            filtered = uniform_filter(source, size=window, mode=mode, origin=origin)
            weight = None
            starved = None
        else:
            report(detail="moving window: usable-pixel weights")
            # uniform_filter divides by the kernel size, so this is the FRACTION of the window that
            # is usable. The same divisor sits in the numerator and cancels in the quotient.
            weight = uniform_filter(good, size=window, mode=mode, origin=origin)
            report(detail="moving window: values")
            numerator = uniform_filter(source * good, size=window, mode=mode, origin=origin)
            starved = weight <= 0
            with np.errstate(invalid="ignore", divide="ignore"):
                filtered = numerator / weight

        out_variances = None
        if data.variances is not None:
            report(detail="moving window: variances")
            source_var = data.variances.astype(float_dtype, copy=False)
            if good is None:
                # Var(mean of k) = sum(Var) / k**2, and uniform_filter already divided by k once.
                out_variances = uniform_filter(source_var, size=window, mode=mode, origin=origin) / n_kernel
            else:
                # With G = n_kernel * weight usable pixels, Var = sum(g Var) / G**2, and
                # sum(g Var) = n_kernel * uniform_filter(Var * g).
                with np.errstate(invalid="ignore", divide="ignore"):
                    out_variances = uniform_filter(source_var * good, size=window, mode=mode, origin=origin) / (
                        n_kernel * weight * weight
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
