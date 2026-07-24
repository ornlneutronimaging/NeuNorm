"""Flexible TOF rebinning — reduce explicit frame-index bins by mean, sum, or median.

Unlike :func:`neunorm.tof.histogram_rebinner.rebin_tof`, which combines *adjacent* TOF bins
by **summing** counts, this module reduces user-defined **ranges of image frames** into one
frame each and lets the caller choose how the frames are combined (mean, sum, or median). It
is the core reducer behind the flexible list-based rebinning requested in the project (a list
such as ``[[0, 4], [5, 30]]`` grouping frames into non-uniform bins).

This module has two entry points: :func:`reduce_tof_bins`, the low-level reducer over
already-canonical, contiguous ``(start, stop)`` index ranges, and :func:`rebin_tof_by_list`,
the user-facing entry that parses the ``[[start, stop], ...]`` list, validates it, and
represents dropped-frame gaps as missing data. Both attach a ``spectra_tof`` point coordinate
giving each bin's representative time (the mean of its member frames' left-edge times), so the
spectra can be updated on export.

``spectra_tof`` is a TOF-frame provenance coordinate. The pipeline path (rebin -> normalize ->
``convert_tof_to_energy`` coordinate labelling -> export) carries it unchanged. Two TOF-axis
transforms handle it specially: :func:`neunorm.tof.binning.get_energy_histogram` reverses it (and
any ``tof``-dependent mask) together with the data, and a second
:func:`neunorm.tof.histogram_rebinner.rebin_tof` drops it — its per-bin mean time cannot be
recomputed from combined bins without the original frame times.
"""

from typing import Literal, Optional, Sequence

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.tof.histogram_rebinner import rebin_tof

#: Reductions understood by :func:`reduce_tof_bins`.
ReductionMode = Literal["mean", "sum", "median"]

#: Point coord (one value per output bin) holding each bin's representative time — the mean of the
#: member frames' left-edge times — carried alongside the bin-edge ``tof`` axis so the spectra can
#: be updated on export (GitHub #192).
SPECTRA_TOF_COORD = "spectra_tof"

#: Mask name flagging output bins that are dropped-frame gaps (True = not real data).
DROPPED_FRAMES_MASK = "dropped_frames"


def _is_integer_index(value: object) -> bool:
    """True only for a genuine integer index (``int`` / NumPy integer), excluding ``bool``."""
    return not isinstance(value, bool) and isinstance(value, (int, np.integer))


def _require_positive_int(value: object, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a positive integer (int / NumPy integer, not bool)."""
    if not _is_integer_index(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _coerce_int_ranges(bins: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Coerce ``(start, stop)`` pairs to plain ints, rejecting malformed / non-integer / bool entries."""
    ranges = []
    for entry in bins:
        if isinstance(entry, (str, bytes)):
            raise ValueError(f"each bin must be a [start, stop] pair of integers; got {entry!r}")
        try:
            start, stop = entry
        except (TypeError, ValueError):
            raise ValueError(f"each bin must be a [start, stop] pair (exactly two indices); got {entry!r}") from None
        if not (_is_integer_index(start) and _is_integer_index(stop)):
            raise ValueError(f"bin indices must be integers; got ({start!r}, {stop!r})")
        ranges.append((int(start), int(stop)))
    return ranges


def _validate_bins(
    data: sc.DataArray, bins: Sequence[Sequence[int]], tof_dim: str
) -> tuple[list[tuple[int, int]], sc.Variable]:
    """Check the TOF axis and bin ranges, returning canonical int ranges and the bin-edge coord.

    Enforces: ``tof_dim`` present with a bin-edge coordinate (length N+1); at least one range;
    every range in-bounds and increasing (``0 <= start < stop <= N``); ranges tile contiguously.
    """
    if tof_dim not in data.dims:
        raise ValueError(f"TOF dimension '{tof_dim}' not found in data dimensions {data.dims}")
    if len(bins) == 0:
        raise ValueError("bins must contain at least one (start, stop) range")

    n_frames = data.sizes[tof_dim]
    if tof_dim not in data.coords:
        raise ValueError(f"data must carry a '{tof_dim}' coordinate to rebuild the rebinned axis")
    tof_edges = data.coords[tof_dim]
    if tof_edges.sizes[tof_dim] != n_frames + 1:
        raise ValueError(
            f"reduce_tof_bins requires a bin-edge '{tof_dim}' coordinate of length N+1 "
            f"({n_frames + 1}); got length {tof_edges.sizes[tof_dim]}"
        )

    ranges = _coerce_int_ranges(bins)
    for start, stop in ranges:
        if not (0 <= start < stop <= n_frames):
            raise ValueError(
                f"bin range ({start}, {stop}) is invalid for a TOF axis of length {n_frames}: "
                "require 0 <= start < stop <= N (half-open, Python convention)"
            )
    for (_, stop), (next_start, _) in zip(ranges[:-1], ranges[1:]):
        if stop != next_start:
            raise ValueError(
                f"bins must tile contiguously to form a bin-edge TOF axis; a bin ending at "
                f"{stop} is followed by a bin starting at {next_start}. Represent dropped "
                "frames as an explicit gap bin upstream rather than as a hole."
            )
    return ranges, tof_edges


def _reduce_one_bin(
    chunk: sc.DataArray, reduction: ReductionMode, tof_dim: str, has_variances: bool
) -> tuple[sc.DataArray, bool]:
    """Collapse one contiguous chunk of frames to a single frame.

    Returns the reduced frame and whether the median variance was reported as unavailable (NaN).
    """
    if reduction == "sum":
        return sc.sum(chunk, tof_dim), False
    if reduction == "mean":
        return sc.mean(chunk, tof_dim), False
    # median: sc.median rejects variance-bearing input, so strip variances first when present.
    # sc.median itself promotes integer data to float, so no manual dtype handling is needed
    # (and sc.values() would itself reject an integer dtype, so only call it when variances exist).
    frame = sc.median(sc.values(chunk) if has_variances else chunk, tof_dim)
    if not has_variances:
        return frame, False
    # Median uncertainty. For N <= 2 the sample median equals the arithmetic mean exactly (N=1: the
    # frame; N=2: the average of the two frames), so its variance is exactly Var(mean). For N >= 3
    # the variance of the sample median of a few HETEROGENEOUS frames (different per-frame means /
    # Poisson variances, as TOF frames generally are) has no reliable closed form — the large-N
    # i.i.d.-normal (π/2)·Var(mean) result materially misstates it — so the uncertainty is reported
    # as NaN (unavailable) rather than a misleading number. The median VALUE itself is exact.
    if chunk.sizes[tof_dim] <= 2:
        frame.variances = sc.variances(sc.mean(chunk, tof_dim)).values
        return frame, False
    frame.variances = np.full(frame.shape, np.nan)
    return frame, True


def reduce_tof_bins(
    data: sc.DataArray,
    bins: Sequence[Sequence[int]],
    reduction: ReductionMode = "mean",
    tof_dim: str = "tof",
) -> sc.DataArray:
    """Reduce contiguous ranges of TOF frames into one frame each.

    Each ``(start, stop)`` range (half-open, Python convention: ``start`` inclusive, ``stop``
    exclusive) collapses the frames ``data[tof_dim, start:stop]`` into a single output frame
    using the chosen ``reduction``. The output ``tof`` axis is rebuilt as a **bin-edge**
    coordinate from the ranges' boundary edges, matching the convention of un-rebinned data.

    Parameters
    ----------
    data : scipp.DataArray
        Image stack with a TOF dimension ``tof_dim``. Must carry a **bin-edge** ``tof_dim``
        coordinate (length ``N + 1`` for a ``tof_dim`` of length ``N``). Variances, spatial
        (``y``, ``x``) coordinates, scalar coordinates, and ``(y, x)`` masks are preserved.
    bins : sequence of (start, stop)
        Contiguous, ordered, half-open frame-index ranges, e.g. ``[(0, 4), (4, 7)]``. Adjacent
        ranges must tile without holes (``stop`` of one equals ``start`` of the next) so the
        output forms a valid bin-edge axis; represent dropped frames as an explicit gap bin
        upstream rather than as a hole here.
    reduction : {"mean", "sum", "median"}, optional
        How to combine the frames in each bin. Default ``"mean"``.

        - ``"mean"``  — value ``= (1/N)·Σxᵢ``; variance ``= ΣVar(xᵢ)/N²``.
        - ``"sum"``   — value ``= Σxᵢ``;        variance ``= ΣVar(xᵢ)``.
        - ``"median"``— value ``= median(xᵢ)`` (exact). Variance: for ``N ≤ 2`` the median equals
          the mean, so the exact ``Var(mean)`` is used; for ``N ≥ 3`` the sample-median variance of a
          few heterogeneous frames has no reliable closed form, so it is reported as ``NaN``
          (unavailable) with a warning rather than a misleading estimate. Integer input is promoted
          to float.
    tof_dim : str, optional
        Name of the TOF dimension. Default ``"tof"``.

    Returns
    -------
    scipp.DataArray
        Stack with ``tof_dim`` reduced to ``len(bins)`` frames, propagated variances, a rebuilt
        bin-edge ``tof_dim`` coordinate, and a ``spectra_tof`` point coordinate giving each bin's
        representative time (the mean of its member frames' left-edge times).

    Raises
    ------
    ValueError
        If ``tof_dim`` is absent, ``reduction`` is not recognised, ``bins`` is empty, an index is
        not an integer, a range is out of bounds or non-increasing, the ranges are not contiguous,
        or ``data`` lacks a bin-edge ``tof_dim`` coordinate.
    """
    if reduction not in ("mean", "sum", "median"):
        raise ValueError(f"reduction must be 'mean', 'sum', or 'median'; got {reduction!r}")
    ranges, tof_edges = _validate_bins(data, bins, tof_dim)

    has_variances = data.variances is not None
    median_variance_unavailable = False
    reduced_frames = []
    for start, stop in ranges:
        frame, unavailable = _reduce_one_bin(data[tof_dim, start:stop], reduction, tof_dim, has_variances)
        median_variance_unavailable = median_variance_unavailable or unavailable
        reduced_frames.append(frame)

    if median_variance_unavailable:
        logger.warning(
            "MEDIAN rebinning: the sample-median variance of a 3+-frame bin has no reliable estimate "
            "for heterogeneous TOF frames, so those bins' variance is reported as NaN (unavailable). "
            "The median values are exact."
        )

    result = sc.concat(reduced_frames, tof_dim)

    # Rebuild the bin-edge tof coordinate: first bin's lower edge, then every bin's upper edge.
    # Contiguity (checked in _validate_bins) makes these shared, giving exactly len(bins)+1 edges.
    edge_indices = [ranges[0][0]] + [stop for _, stop in ranges]
    result.coords[tof_dim] = sc.concat([tof_edges[tof_dim, i] for i in edge_indices], tof_dim)

    # Per-bin representative time = mean of the member frames' left-edge times (the VENUS spectra
    # convention: tof_edges[start:stop] are the left edges of frames start..stop-1). Carried as a
    # point coord alongside the bin-edge tof axis so the spectra can be updated on export (#192).
    bin_times = np.array([tof_edges[tof_dim, start:stop].values.mean() for start, stop in ranges])
    result.coords[SPECTRA_TOF_COORD] = sc.array(dims=[tof_dim], values=bin_times, unit=tof_edges.unit)
    return result


def _parse_bin_list(bin_list: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Coerce a user bin list to integer ``(start, stop)`` pairs, rejecting malformed entries."""
    if len(bin_list) == 0:
        raise ValueError("bin_list must contain at least one [start, stop) range")
    ranges: list[tuple[int, int]] = []
    for entry in bin_list:
        if isinstance(entry, (str, bytes)):
            raise ValueError(f"each bin must be a [start, stop] pair of integers; got {entry!r}")
        try:
            pair = tuple(entry)
        except TypeError:
            raise ValueError(f"each bin must be a [start, stop] pair of integers; got {entry!r}") from None
        if len(pair) != 2:
            raise ValueError(f"each bin must be a [start, stop] pair (exactly two indices); got {entry!r}")
        start, stop = pair
        if not (_is_integer_index(start) and _is_integer_index(stop)):
            raise ValueError(f"bin indices must be integers; got [{start!r}, {stop!r}]")
        ranges.append((int(start), int(stop)))
    return ranges


def _validate_bin_list(ranges: list[tuple[int, int]], n_frames: int) -> None:
    """Reject out-of-bounds, empty/reversed, unordered, or overlapping ranges.

    Interior gaps between bins are legitimate (the user may drop frames) and are NOT flagged here.
    """
    for start, stop in ranges:
        if start < 0 or stop > n_frames:
            raise ValueError(
                f"bin [{start}, {stop}) is out of bounds for a TOF axis of {n_frames} frames "
                f"(require 0 <= start < stop <= {n_frames})"
            )
        if start >= stop:
            raise ValueError(
                f"bin [{start}, {stop}) is empty or reversed: require start < stop "
                "(half-open, at least one frame per bin)"
            )
    for (prev_start, prev_stop), (start, stop) in zip(ranges[:-1], ranges[1:]):
        if start < prev_stop:
            if start >= prev_start:
                raise ValueError(
                    f"bins overlap: [{prev_start}, {prev_stop}) and [{start}, {stop}) share frame(s); "
                    "bins must be disjoint (gaps between bins are allowed, overlaps are not)"
                )
            raise ValueError(
                f"bins must be given in increasing order; bin [{start}, {stop}) starts before the "
                f"preceding bin [{prev_start}, {prev_stop})"
            )


def rebin_tof_by_list(
    data: sc.DataArray,
    bin_list: Sequence[Sequence[int]],
    reduction: ReductionMode = "mean",
    tof_dim: str = "tof",
) -> sc.DataArray:
    """Rebin a TOF stack by an explicit list of half-open frame-index ranges.

    This is the user-facing entry point for flexible rebinning. Each ``[start, stop]`` in
    ``bin_list`` (half-open, Python convention) selects the frames ``start .. stop - 1`` and
    reduces them to one output frame via :func:`reduce_tof_bins`.

    **Gaps are allowed.** Where consecutive bins leave an interior hole (one bin's ``stop`` is
    less than the next bin's ``start``), the dropped frames are represented as an explicit **gap
    bin flagged as missing data** — added to the ``"dropped_frames"`` mask along ``tof``
    (``True`` = dropped) and given ``NaN`` values/variances — so the output ``tof`` axis stays a
    contiguous, monotonic bin-edge coordinate spanning the gap. Frames before the first bin or
    after the last bin are simply excluded (the axis spans the first ``start`` to the last
    ``stop``); only interior gaps become masked bins.

    Reconciling with the design rule "do not provide the option to skip images": that rule forbids
    skipping frames *within* a bin (a bin like ``0, 1, 5, 8`` that is not a contiguous run), which is
    impossible here — each ``[start, stop)`` is a contiguous range by construction. Dropping frames
    *between* bins is a different, deliberate operation (excluding unwanted data), and those dropped
    frames are explicitly recorded as missing (the gap bin above), not silently discarded.

    Parameters
    ----------
    data : scipp.DataArray
        Image stack with a bin-edge ``tof_dim`` coordinate (see :func:`reduce_tof_bins`).
    bin_list : sequence of [start, stop]
        Ordered, non-overlapping half-open frame-index ranges, e.g. ``[[0, 4], [5, 30]]``. Ranges
        must be increasing and not overlap; interior gaps between them are permitted.
    reduction : {"mean", "sum", "median"}, optional
        How to combine the frames in each real bin. Default ``"mean"``.
    tof_dim : str, optional
        Name of the TOF dimension. Default ``"tof"``.

    Returns
    -------
    scipp.DataArray
        Rebinned stack with ``len(bin_list)`` real bins plus one masked bin per interior gap,
        propagated variances, a bin-edge ``tof_dim`` coordinate, and a ``spectra_tof`` point
        coordinate (mean member time per real bin; ``NaN`` on gap bins).

    Raises
    ------
    ValueError
        If ``bin_list`` is empty, or (via :func:`reduce_tof_bins`) a range is out of bounds,
        non-increasing, overlapping, or unordered.
    """
    ranges = _parse_bin_list(bin_list)
    if tof_dim not in data.dims:
        raise ValueError(f"TOF dimension '{tof_dim}' not found in data dimensions {data.dims}")
    _validate_bin_list(ranges, data.sizes[tof_dim])

    # Fill interior gaps with explicit bins so the reduced axis stays contiguous, tracking which
    # output bins are dropped-frame gaps. _validate_bin_list above has already guaranteed the
    # ranges are sorted, disjoint and in-bounds, so reduce_tof_bins' contiguity guard is only a
    # backstop here.
    filled: list[tuple[int, int]] = []
    is_gap: list[bool] = []
    for index, (start, stop) in enumerate(ranges):
        if index > 0:
            prev_stop = ranges[index - 1][1]
            if start > prev_stop:  # interior gap: frames [prev_stop, start) were dropped
                filled.append((prev_stop, start))
                is_gap.append(True)
        filled.append((start, stop))
        is_gap.append(False)

    result = reduce_tof_bins(data, filled, reduction=reduction, tof_dim=tof_dim)

    if any(is_gap):
        gap = np.array(is_gap)
        # A dropped-frame bin carries no meaningful value: NaN it (values + variances) and flag it
        # with the tof mask. NaN needs a float dtype, so promote an integer result (e.g. an int-count
        # sum, which scipp keeps integer) to float first. (result.dtype is a scipp DType.)
        if result.dtype not in (sc.DType.float32, sc.DType.float64):
            result = result.astype(sc.DType.float64)
        result.values[gap] = np.nan
        if result.variances is not None:
            result.variances[gap] = np.nan
        result.coords[SPECTRA_TOF_COORD].values[gap] = np.nan  # dropped bins have no representative time
        result.masks[DROPPED_FRAMES_MASK] = sc.array(dims=[tof_dim], values=gap)

    return result


def linear_bin_list(n_frames: int, step: int) -> list[tuple[int, int]]:
    """Uniform frame-count bins for :func:`rebin_tof_by_list`.

    Returns contiguous ``(start, stop)`` ranges ``[(0, step), (step, 2*step), ...]`` covering
    ``n_frames`` frames; the final bin is truncated when ``n_frames`` is not a multiple of ``step``.
    This is the mean/median-capable analogue of iBeatles' linear-by-file-index binning (and of
    ``rebin_tof(unit="bins")``, which sums): pair it with :func:`rebin_tof_by_list` to average or
    take the median of every ``step`` frames.

    Parameters
    ----------
    n_frames : int
        Number of TOF frames to bin (``data.sizes["tof"]``).
    step : int
        Frames per bin (> 0).
    """
    _require_positive_int(n_frames, "n_frames")
    _require_positive_int(step, "step (frames per bin)")
    return [(start, min(start + step, n_frames)) for start in range(0, n_frames, step)]


def log_bin_list(n_frames: int, factor: float) -> list[tuple[int, int]]:
    """Geometric (logarithmic) frame-index bins for :func:`rebin_tof_by_list`.

    Bin edges grow by roughly ``(1 + factor)`` along the frame index — fine bins early, coarser
    bins later — the frame-index logarithmic mode that
    :func:`neunorm.tof.histogram_rebinner.rebin_tof` does not offer (it rejects ``logarithmic`` for
    ``unit="bins"``). Edges are forced to strictly increasing integers (at least one frame per bin),
    which also avoids the zero-start infinite loop in the original iBeatles implementation
    (``edge += edge * factor`` never advances from ``0``). The final bin is truncated to ``n_frames``.

    Parameters
    ----------
    n_frames : int
        Number of TOF frames to bin.
    factor : float
        Geometric growth factor per edge (> 0); a larger ``factor`` gives fewer, wider late bins.
    """
    _require_positive_int(n_frames, "n_frames")
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float, np.integer, np.floating))
        or not np.isfinite(factor)
        or factor <= 0
    ):
        raise ValueError(f"factor must be a positive, finite number; got {factor!r}")
    edges = [0]
    while edges[-1] < n_frames:
        grown = int(round(edges[-1] * (1.0 + factor)))
        # Force progress by at least one frame: handles the edge==0 start (0*(1+f)=0) and any
        # rounding that would repeat an edge, so every bin has >= 1 frame and the loop terminates.
        edges.append(min(max(grown, edges[-1] + 1), n_frames))
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def apply_tof_rebin(
    data: sc.DataArray,
    spec: "int | Sequence[Sequence[int]]",
    reduction: Optional[ReductionMode] = None,
    tof_dim: str = "tof",
) -> sc.DataArray:
    """Dispatch a pipeline ``rebin_by_tof`` request to the appropriate rebinner.

    ``spec`` is a resolved rebin specification — an integer uniform factor (a ``True`` auto-request
    must be resolved to a factor by the caller via ``analyze_statistics``) or an explicit
    ``[[start, stop], ...]`` bin list. ``reduction`` selects how frames combine:

    - ``None`` — preserves existing behavior: an integer factor **sums** (via
      :func:`neunorm.tof.histogram_rebinner.rebin_tof`), a bin list takes the **mean**.
    - ``"sum"`` / ``"mean"`` / ``"median"`` — applied to either spec. An integer factor with a
      mean/median reduction is expanded to uniform bins via :func:`linear_bin_list` and reduced by
      :func:`rebin_tof_by_list`.

    Parameters
    ----------
    data : scipp.DataArray
        Histogram stack with a bin-edge ``tof_dim`` coordinate.
    spec : int or sequence of [start, stop]
        Uniform factor (frames per bin) or an explicit half-open frame-index bin list.
    reduction : {"mean", "sum", "median"}, optional
        See above; ``None`` picks sum for a factor and mean for a bin list.
    tof_dim : str, optional
        Name of the TOF dimension. Default ``"tof"``.

    Raises
    ------
    ValueError
        If ``spec`` is a bool (resolve it to a factor first) or is neither an int nor a bin list.
    """
    if isinstance(spec, bool):
        raise ValueError("resolve a boolean rebin_by_tof to an integer factor before calling apply_tof_rebin")
    if isinstance(spec, (int, np.integer)):
        if reduction is None or reduction == "sum":
            return rebin_tof(data, int(spec))
        return rebin_tof_by_list(
            data, linear_bin_list(data.sizes[tof_dim], int(spec)), reduction=reduction, tof_dim=tof_dim
        )
    if isinstance(spec, (list, tuple)):
        # Default to mean for a bin list only when reduction is genuinely omitted (None); a falsy
        # but invalid value like "" must still reach rebin_tof_by_list's validation, not be masked.
        return rebin_tof_by_list(data, spec, reduction=("mean" if reduction is None else reduction), tof_dim=tof_dim)
    raise ValueError(f"rebin_by_tof must be a bool, an int factor, or a list of [start, stop] pairs; got {spec!r}")
