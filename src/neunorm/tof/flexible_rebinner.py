"""Flexible TOF rebinning — reduce explicit frame-index bins by mean, sum, or median.

Unlike :func:`neunorm.tof.histogram_rebinner.rebin_tof`, which combines *adjacent* TOF bins
by **summing** counts, this module reduces user-defined **ranges of image frames** into one
frame each and lets the caller choose how the frames are combined (mean, sum, or median). It
is the core reducer behind the flexible list-based rebinning requested in the project (a list
such as ``[[0, 4], [5, 30]]`` grouping frames into non-uniform bins).

This module provides only the reduction primitive over already-canonical, contiguous
``(start, stop)`` index ranges. Parsing the user-facing list, representing dropped-frame gaps
as missing data, input validation, and the per-bin mean-time spectra provenance are layered on
top of it elsewhere.
"""

from typing import Literal, Sequence

import numpy as np
import scipp as sc
from loguru import logger

#: Reductions understood by :func:`reduce_tof_bins`.
ReductionMode = Literal["mean", "sum", "median"]

#: Large-N approximation for the variance of the sample median of normally distributed values:
#: ``Var(median) ≈ (π/2)·Var(mean)``. Applied only to bins with two or more frames — for a
#: single-frame bin the median is the frame itself, so its variance is exact.
_MEDIAN_VARIANCE_FACTOR = np.pi / 2.0


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

    ranges = [(int(start), int(stop)) for start, stop in bins]
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

    Returns the reduced frame and whether the median large-N variance approximation was applied.
    """
    if reduction == "sum":
        return sc.sum(chunk, tof_dim), False
    if reduction == "mean":
        return sc.mean(chunk, tof_dim), False
    # median: sc.median rejects variance-bearing input, so take the value from a values-only copy.
    frame = sc.median(sc.values(chunk), tof_dim)
    if not has_variances:
        return frame, False
    # Var(median) has no simple closed form: use (π/2)·Var(mean) for N >= 2 (the large-N normal
    # approximation) and the exact Var(mean) for a single-frame bin, where the median IS the frame.
    mean_variance = sc.variances(sc.mean(chunk, tof_dim)).values
    if chunk.sizes[tof_dim] > 1:
        frame.variances = _MEDIAN_VARIANCE_FACTOR * mean_variance
        return frame, True
    frame.variances = mean_variance
    return frame, False


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
        - ``"median"``— value ``= median(xᵢ)``; variance ``≈ (π/2)·ΣVar(xᵢ)/N²`` for ``N ≥ 2``
          (a large-N approximation; a warning is emitted), and the exact ``Var`` for ``N = 1``.
    tof_dim : str, optional
        Name of the TOF dimension. Default ``"tof"``.

    Returns
    -------
    scipp.DataArray
        Stack with ``tof_dim`` reduced to ``len(bins)`` frames, propagated variances, and a
        rebuilt bin-edge ``tof_dim`` coordinate.

    Raises
    ------
    ValueError
        If ``tof_dim`` is absent, ``reduction`` is not recognised, ``bins`` is empty, a range
        is out of bounds or non-increasing, the ranges are not contiguous, or ``data`` lacks a
        bin-edge ``tof_dim`` coordinate.
    """
    if reduction not in ("mean", "sum", "median"):
        raise ValueError(f"reduction must be 'mean', 'sum', or 'median'; got {reduction!r}")
    ranges, tof_edges = _validate_bins(data, bins, tof_dim)

    has_variances = data.variances is not None
    approximated_median = False
    reduced_frames = []
    for start, stop in ranges:
        frame, approximated = _reduce_one_bin(data[tof_dim, start:stop], reduction, tof_dim, has_variances)
        approximated_median = approximated_median or approximated
        reduced_frames.append(frame)

    if approximated_median:
        logger.warning(
            "MEDIAN rebinning: propagated variance uses the large-N approximation "
            "Var(median) ≈ (π/2)·Var(mean); treat median uncertainties as approximate."
        )

    result = sc.concat(reduced_frames, tof_dim)

    # Rebuild the bin-edge tof coordinate: first bin's lower edge, then every bin's upper edge.
    # Contiguity (checked in _validate_bins) makes these shared, giving exactly len(bins)+1 edges.
    edge_indices = [ranges[0][0]] + [stop for _, stop in ranges]
    result.coords[tof_dim] = sc.concat([tof_edges[tof_dim, i] for i in edge_indices], tof_dim)
    return result
