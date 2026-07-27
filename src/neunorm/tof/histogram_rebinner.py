"""Rebin TOF histograms — sum adjacent bins, or reduce explicit/uniform frame groups.

:func:`rebin_tof` is the single entry point. By default (and for an integer / time / wavelength /
manual ``width``) it combines *adjacent* TOF bins by **summing** counts — histogram-mode rebinning
via :func:`scipp.rebin`, which can only sum. Passing ``reduction="mean"`` or ``"median"``, or an
explicit ``[[start, stop], ...]`` frame-index bin list as ``width``, instead reduces user-defined
ranges of image frames into one frame each with the chosen reduction — the flexible list-based
rebinning requested in the project (e.g. ``[[0, 4], [5, 30]]`` grouping frames into non-uniform
bins, with dropped-frame gaps represented as missing data).

Public helpers backing the reduction path: :func:`reduce_tof_bins` (the low-level reducer over
already-canonical, contiguous ``(start, stop)`` ranges) and the :func:`linear_bin_list` /
:func:`log_bin_list` bin-list generators. Reduced output carries a ``spectra_tof`` point coordinate
giving each bin's representative time (the mean of its member frames' left-edge times) alongside the
bin-edge ``tof`` axis, so the spectra can be updated on export.

``spectra_tof`` is a TOF-frame provenance coordinate. The pipeline path (rebin -> normalize ->
``convert_tof_to_energy`` coordinate labelling -> export) carries it unchanged. Two TOF-axis
transforms handle it specially: :func:`neunorm.tof.binning.get_energy_histogram` reverses it (and
any ``tof``-dependent mask) together with the data, and a second sum-mode :func:`rebin_tof` drops it
— its per-bin mean time cannot be recomputed from combined bins without the original frame times.
"""

from typing import Literal, Optional, Sequence, Union

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.tof.coordinate_converter import convert_tof_to_wavelength, convert_wavelength_to_tof

#: Reductions understood by :func:`reduce_tof_bins` / the reduction path of :func:`rebin_tof`.
ReductionMode = Literal["mean", "sum", "median"]

#: Point coord (one value per output bin) holding each bin's representative time — the mean of the
#: member frames' left-edge times — carried alongside the bin-edge ``tof`` axis so the spectra can
#: be updated on export (GitHub #192).
SPECTRA_TOF_COORD = "spectra_tof"

#: Mask name flagging output bins that are dropped-frame gaps (True = not real data).
DROPPED_FRAMES_MASK = "dropped_frames"


def rebin_with_snapped_boundaries(old_edges: sc.Variable, requested_tof_edges: sc.Variable):
    """
    For requested TOF edges that don't align with original TOF edges, snap to the nearest original edge on the left.
    This ensures that we only combine adjacent bins and don't create arbitrary bin schemes,
    which is a requirement for histogram-mode data.

    Parameters
    ----------
    old_edges : sc.Variable
        Original TOF bin edges.
    requested_tof_edges : sc.Variable
        Desired TOF bin edges that may not align with original edges.

    Returns
    -------
    sc.Variable
        New TOF edges snapped to the nearest original edges on the left.
    """
    # Map requested edges to indices of the nearest original edge on the left.
    old_vals = np.asarray(old_edges.values)
    req_vals = np.asarray(requested_tof_edges.values)
    idx = np.searchsorted(old_vals, req_vals, side="right") - 1
    # Prevent negative indices (which would wrap to the last element) or indices
    # beyond the last edge. This keeps snapping within the valid range of edges.
    idx = np.clip(idx, 0, len(old_vals) - 1)
    snapped_vals = old_vals[idx]
    # Validate that snapped edges form a strictly increasing sequence of original edges.
    if snapped_vals.size == 0 or not np.all(np.diff(snapped_vals) > 0):
        raise ValueError(
            "Requested TOF binning would require splitting existing bins or "
            "would produce non-increasing/zero-width bins. "
            "Adjust the requested TOF edges or bin width so that consecutive "
            "snapped edges correspond to strictly increasing original bin edges."
        )

    return sc.array(dims=old_edges.dims, values=snapped_vals, unit=old_edges.unit)


# --------------------------------------------------------------------------------------------
# Flexible reduction path (mean / sum / median over explicit or uniform frame-index bins).
# --------------------------------------------------------------------------------------------


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

    This is the low-level primitive behind the reduction path of :func:`rebin_tof`. Each
    ``(start, stop)`` range (half-open, Python convention: ``start`` inclusive, ``stop``
    exclusive) collapses the frames ``data[tof_dim, start:stop]`` into a single output frame
    using the chosen ``reduction``. The output ``tof`` axis is rebuilt as a **bin-edge**
    coordinate from the ranges' boundary edges, matching the convention of un-rebinned data.

    Ranges must tile contiguously (no gaps); to rebin by a list that may contain gaps, pass the
    list as the ``width`` of :func:`rebin_tof`, which inserts the gap bins for you.

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


def _reduce_tof_by_bin_list(
    data: sc.DataArray,
    bin_list: Sequence[Sequence[int]],
    reduction: ReductionMode,
    tof_dim: str,
) -> sc.DataArray:
    """Rebin a TOF stack by an explicit list of half-open frame-index ranges (the list-input path
    of :func:`rebin_tof`).

    Each ``[start, stop]`` in ``bin_list`` (half-open, Python convention) selects the frames
    ``start .. stop - 1`` and reduces them to one output frame via :func:`reduce_tof_bins`.

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
    """Uniform frame-count bins for the reduction path of :func:`rebin_tof`.

    Returns contiguous ``(start, stop)`` ranges ``[(0, step), (step, 2*step), ...]`` covering
    ``n_frames`` frames; the final bin is truncated when ``n_frames`` is not a multiple of ``step``.
    This is the mean/median-capable analogue of iBeatles' linear-by-file-index binning (and of
    ``rebin_tof(unit="bins")``, which sums): pass the returned list as the ``width`` of
    :func:`rebin_tof` with ``reduction="mean"`` (or ``"median"``) to average or take the median of
    every ``step`` frames.

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
    """Geometric (logarithmic) frame-index bins for the reduction path of :func:`rebin_tof`.

    Bin edges grow by roughly ``(1 + factor)`` along the frame index — fine bins early, coarser
    bins later — the frame-index logarithmic mode :func:`rebin_tof` does not build from a plain
    factor (pass the generated list as ``width`` instead). Edges are forced to strictly increasing
    integers (at least one frame per bin), which also avoids the zero-start infinite loop in the
    original iBeatles implementation (``edge += edge * factor`` never advances from ``0``). The
    final bin is truncated to ``n_frames``.

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


def _rebin_tof_uniform_reduced(
    data: sc.DataArray, width: object, reduction: ReductionMode, unit: str, logarithmic: bool, tof_dim: str
) -> sc.DataArray:
    """Uniform frame-index grouping reduced by mean/median — the non-sum analogue of ``unit='bins'``.

    ``scipp.rebin`` (the time/wavelength/manual sum machinery) can only sum, so mean/median support
    only uniform frame-index bins here; for non-uniform or log bins with mean/median, pass an
    explicit bin list (or :func:`log_bin_list` output) as ``width``.
    """
    if unit != "bins":
        raise ValueError(
            f"reduction={reduction!r} supports only uniform frame-index bins (unit='bins') or an "
            f"explicit [[start, stop], ...] bin list; got unit={unit!r}. For non-uniform or "
            "time/wavelength bins with mean/median, pass an explicit bin list as width."
        )
    if logarithmic:
        raise ValueError(
            f"reduction={reduction!r} does not build logarithmic bins from a plain factor; "
            "generate a bin list with log_bin_list(n_frames, factor) and pass it as width."
        )
    _require_positive_int(width, "width (frames per bin)")
    return _reduce_tof_by_bin_list(
        data, linear_bin_list(data.sizes[tof_dim], int(width)), reduction=reduction, tof_dim=tof_dim
    )


def rebin_tof(  # noqa: C901
    data: sc.DataArray,
    width: Union[int, float, sc.Variable, list, tuple],
    unit: str = "bins",
    logarithmic: bool = False,
    tof_dim: str = "tof",
    l_source_to_detector: float = 25.0,
    detector_time_offset: float = 5000.0,
    *,
    reduction: Optional[ReductionMode] = None,
) -> sc.DataArray:
    """Rebin a TOF stack — sum adjacent bins, or reduce explicit/uniform frame groups.

    Two modes, selected by ``width`` and ``reduction``:

    - **Sum (default).** With ``reduction`` ``None`` or ``"sum"`` and a scalar/``sc.Variable``
      ``width``, adjacent TOF bins are combined by **summing** counts via :func:`scipp.rebin`
      (histogram-mode rebinning). This is the original, backward-compatible behavior; ``unit``,
      ``logarithmic``, ``l_source_to_detector`` and ``detector_time_offset`` apply here.
    - **Reduction.** With ``reduction="mean"`` or ``"median"``, or with ``width`` given as an
      explicit ``[[start, stop], ...]`` frame-index bin list, user-defined ranges of frames are
      reduced into one frame each (see :func:`reduce_tof_bins`). A bin list may leave interior
      gaps (dropped frames), represented as missing-data bins (``dropped_frames`` mask + ``NaN``);
      an integer ``width`` with a mean/median reduction is expanded to uniform bins. Reduced output
      carries a ``spectra_tof`` point coordinate. mean/median support only ``unit="bins"`` (or an
      explicit list) — ``scipp.rebin`` is sum-only.

    Requirements (sum mode)
    - Combine N adjacent TOF bins by summing counts
    - Update TOF bin edges accordingly
    - Propagate variance correctly through summation

    Constraints

    For histogram-mode data (TPX1, TPX3 histogram mode):
    - Can ONLY combine adjacent bins
    - Cannot create arbitrary bin schemes (would require raw events)
    - Cannot split bins

    Parameters
    ----------
    data : sc.DataArray
        Input data with TOF dimension.
    width : int, float, sc.Variable, or list/tuple of [start, stop]
        Sum mode: the new bin width in terms of ``unit`` (positive), or a ``sc.Variable`` of desired
        edges when ``unit="manual"``. Reduction mode: an integer number of frames per uniform bin,
        or an explicit ``[[start, stop], ...]`` list (or tuple) of half-open frame-index ranges.
    unit : str
        Unit by which the new bin width is specified (sum mode). Must be one of `time`, `wavelength`,
        `bins`, or `manual`. Default is `bins`.
        If `bins`, width is interpreted as the number of adjacent bins to combine.
        If `time`, width is interpreted as the desired width of the new TOF bins in the same unit as the coordinates.
        If `wavelength`, width is interpreted as the desired width of the new TOF bins in Angstrom units,
        and converted to time using the provided source-to-detector distance and detector time offset.
        If `manual`, width is required to be a 1-D sc.Variable (at least two values) representing the
        desired edges of the new TOF bins. Its interpretation depends on the variable's unit:
        if dimensionless, the values are treated as integer bin indices into the existing TOF coordinate
        (must be of integer dtype and within bounds) and the corresponding original edges are selected;
        otherwise the values are interpreted as explicit TOF edges in, or convertible to, the unit of the
        TOF coordinates (a wavelength unit is also accepted and converted using `l_source_to_detector` and
        `detector_time_offset`), and are then snapped to the nearest original edges on the left.
    reduction : {"sum", "mean", "median"}, optional
        How frames combine. ``None`` (default) sums (backward compatible). ``"mean"``/``"median"``
        select the reduction path; ``"sum"`` is the explicit form of the default. See
        :func:`reduce_tof_bins` for the median-variance convention.
    logarithmic : bool
        Whether to use logarithmic binning (sum mode only). Default is False.
    tof_dim : str
        Name of the TOF dimension in the DataArray. Default is "tof".
    l_source_to_detector : float
        Distance from the source to the detector in meters. Required for wavelength binning. Default is 25.0.
    detector_time_offset : float
        Time offset of the detector in same unit as TOF. Required for wavelength binning. Default is 5000.0.

    Returns
    -------
    sc.DataArray
        Rebinned DataArray with updated TOF bins and propagated variance.
    """

    if tof_dim not in data.dims:
        raise ValueError(f"Specified TOF dimension '{tof_dim}' not found in data dimensions {data.dims}")

    # A NumPy-integer factor is a valid integer factor: normalize it to a Python int so both the sum
    # path (which checks `isinstance(width, int)`, and np.int64 is not a Python int) and the reduction
    # path accept it. Booleans are deliberately NOT coerced here: ``np.bool_`` is not an ``np.integer``
    # and is rejected downstream, while a Python ``bool`` subclasses ``int`` and so is treated as an
    # integer factor by the sum path (``True`` acts as width 1, i.e. a no-op) and rejected by the
    # reduction path. Callers resolve a boolean rebin request to a factor before calling.
    if isinstance(width, np.integer):
        width = int(width)

    if reduction not in (None, "sum", "mean", "median"):
        raise ValueError(f"reduction must be None, 'sum', 'mean', or 'median'; got {reduction!r}")

    # Reduction path: an explicit frame-index bin list, or an integer factor with mean/median.
    if isinstance(width, (list, tuple)):
        return _reduce_tof_by_bin_list(
            data, width, reduction=("mean" if reduction is None else reduction), tof_dim=tof_dim
        )
    if reduction in ("mean", "median"):
        return _rebin_tof_uniform_reduced(data, width, reduction, unit, logarithmic, tof_dim)

    # --- Sum path (reduction None or "sum"): combine adjacent bins via scipp.rebin (unchanged) ---
    if isinstance(width, sc.Variable) and unit != "manual":
        raise ValueError(
            "When width is provided as a sc.Variable, unit must be set to 'manual' and "
            "the variable should represent the desired edges of the new TOF bins."
        )

    if unit == "manual":
        if not isinstance(width, sc.Variable):
            raise ValueError(
                "When unit is 'manual', width must be provided as a sc.Variable "
                "representing the desired edges of the new TOF bins."
            )
        if width.size < 2:
            raise ValueError("Manual TOF edges must have at least two values.")

        if width.unit == sc.units.dimensionless:
            # Interpret as bin indices and extract TOF edges
            if not np.issubdtype(width.values.dtype, np.integer):
                raise ValueError(
                    "When width is a dimensionless sc.Variable, it must have an integer dtype representing bin indices."
                )
            if np.any(width.values < 0) or np.any(width.values >= data.coords[tof_dim].size):
                raise ValueError("Bin indices in width are out of bounds for the TOF dimension.")
            new_tof_edges = data.coords[tof_dim][width.values]
        else:
            # Try to convert to the unit of the TOF coordinates
            try:
                converted_width = sc.to_unit(width, data.coords[tof_dim].unit)
            except sc.UnitError as e:
                # now try wavelength
                try:
                    lsd = sc.scalar(l_source_to_detector, unit="m")
                    offset = sc.scalar(detector_time_offset, unit=data.coords[tof_dim].unit)
                    converted_width = (
                        sc.to_unit(
                            sc.to_unit(width, unit="Angstrom") * sc.constants.m_n * lsd / sc.constants.h,
                            data.coords[tof_dim].unit,
                        )
                        - offset
                    )

                except sc.UnitError as e2:
                    raise ValueError(
                        f"Width provided as a sc.Variable could not be converted to the unit of the TOF coordinates. "
                        f"Conversion to time failed with error: {e}. "
                        f"Conversion to wavelength failed with error: {e2}."
                    )
            new_tof_edges = rebin_with_snapped_boundaries(data.coords[tof_dim], converted_width)
    elif unit == "bins":
        if width <= 0:
            raise ValueError("Rebinning width must be positive.")

        if logarithmic:
            raise ValueError("Logarithmic binning is not supported when unit is 'bins'.")

        # check if width is an integer and if not, raise an error
        if not isinstance(width, int):
            raise ValueError(
                "When unit is 'bins', width must be an integer representing the number of adjacent bins to combine."
            )

        if width == 1:
            return data  # No rebinning needed
        # create new TOF edges by taking every Nth edge from the original TOF edges
        new_tof_edges = data.coords[tof_dim][::width]
        # add last edge if not included
        if not sc.identical(new_tof_edges[-1], data.coords[tof_dim][-1]):
            new_tof_edges = sc.concat([new_tof_edges, data.coords[tof_dim][-1:]], dim=tof_dim)
    elif unit == "time":
        if width <= 0:
            raise ValueError("Rebinning width must be positive.")

        tof_edges = data.coords[tof_dim]
        if logarithmic:
            last_bin = np.ceil(np.log(tof_edges.values[-1] / tof_edges.values[0]) / np.log1p(width))
            requested_tof_edges = sc.array(
                dims=[tof_dim], values=tof_edges.values[0] * (1 + width) ** np.arange(last_bin + 1), unit=tof_edges.unit
            )
        else:
            requested_tof_edges = sc.arange(
                dim=tof_dim,
                start=tof_edges.values[0],
                stop=tof_edges.values[-1] + width,
                step=width,
                unit=tof_edges.unit,
            )

        new_tof_edges = rebin_with_snapped_boundaries(tof_edges, requested_tof_edges)

    elif unit == "wavelength":
        if width <= 0:
            raise ValueError("Rebinning width must be positive.")
        tof_edges = data.coords[tof_dim]
        lsd = sc.scalar(l_source_to_detector, unit="m")
        offset = sc.scalar(detector_time_offset, unit=tof_edges.unit)
        if logarithmic:
            # convert to wavelength edges, create logarithmic wavelength edges, then convert back to TOF edges
            wavelength_edges = convert_tof_to_wavelength(tof_edges, lsd, offset)
            last_bin = np.ceil(np.log(wavelength_edges.values[-1] / wavelength_edges.values[0]) / np.log1p(width))
            requested_wavelength_edges = sc.array(
                dims=[tof_dim],
                values=wavelength_edges.values[0] * (1 + width) ** np.arange(last_bin + 1),
                unit="Angstrom",
            )
            requested_tof_edges = convert_wavelength_to_tof(requested_wavelength_edges, lsd, offset)
        else:
            requested_tof_width = convert_wavelength_to_tof(
                sc.scalar(width, unit="Angstrom"), lsd, sc.scalar(0, unit=tof_edges.unit)
            )
            requested_tof_edges = sc.arange(
                dim=tof_dim,
                start=tof_edges.values[0],
                stop=tof_edges.values[-1] + requested_tof_width.values,
                step=requested_tof_width.values,
                unit=tof_edges.unit,
            )

        new_tof_edges = rebin_with_snapped_boundaries(tof_edges, requested_tof_edges)
    else:
        raise ValueError("Invalid unit for rebinning width. Must be one of 'manual', 'time', 'wavelength', or 'bins'.")

    # rebin histogrammed data by summing over the specified factor
    rebinned_data = sc.rebin(data, {tof_dim: new_tof_edges})

    # copy over unaligned coords; only DataArray/Dataset can be passed to sc.rebin
    # so for coord Variables we build rebinned edges and preserve the rest as-is.
    for coord in data.coords:
        if not data.coords[coord].aligned:
            if tof_dim in data.coords[coord].dims:
                # turn into DataArray to use sc.rebin for edge rebinning, then convert back to Variable
                rebinned_edges = sc.rebin(
                    sc.DataArray(data.coords[coord], coords={tof_dim: data.coords[tof_dim]}),
                    {tof_dim: new_tof_edges},
                ).data
                rebinned_data.coords[coord] = rebinned_edges
            else:
                rebinned_data.coords[coord] = data.coords[coord]
            rebinned_data.coords.set_aligned(coord, False)

    return rebinned_data
