"""
Pulse ID reconstruction from TOF data with rollover correction.

Implements a three-pass algorithm to reconstruct pulse assignments from
time-of-flight data that exhibits rollover behavior and short-range temporal
disorder from TPX3 readout FIFOs.

The algorithm handles:
- TOF rollovers when neutron pulse period resets
- Clustered false positive detections from temporal disorder
- Fine-grained boundary refinement near rollover regions
- Multi-chip processing (4 independent TPX3 chips)
- Parallel processing for multi-chip detectors (P4 optimization)

See Also
--------
examples/pulse_reconstruction_tutorial.ipynb : Interactive tutorial with visualizations
tests/unit/test_pulse_reconstruction.py : Unit tests and usage examples
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numpy as np
from loguru import logger

from neunorm.utils._numba_compat import njit


@njit(cache=True)
def _assign_pulse_ids_single_pass(
    pulse_ids: np.ndarray,
    rollover_indices: np.ndarray,
) -> None:
    """
    Assign pulse IDs in a single O(N) pass.

    This is the JIT-compiled helper for _coarse_pulse_assignment.
    Instead of O(N*R) slice assignments, this does O(N) single-pass assignment.

    Parameters
    ----------
    pulse_ids : np.ndarray
        Output array (int32), modified in-place. Should be initialized to zeros.
    rollover_indices : np.ndarray
        Sorted array of rollover positions (int64 from np.where)
    """
    if len(rollover_indices) == 0:
        return

    n = len(pulse_ids)
    n_rollovers = len(rollover_indices)
    rollover_ptr = 0
    current_pulse = 0

    for i in range(n):
        # Check if we've reached the next rollover
        while rollover_ptr < n_rollovers and i >= rollover_indices[rollover_ptr]:
            current_pulse = rollover_ptr + 1
            rollover_ptr += 1
        pulse_ids[i] = current_pulse


@njit(cache=True)
def _cluster_rollovers(
    rollover_indices: np.ndarray,
    cluster_threshold: float,
) -> np.ndarray:
    """
    Identify first rollover in each cluster.

    This is the JIT-compiled helper for _clean_clustered_rollovers.
    Returns an array of indices (into rollover_indices) that should be kept.

    Parameters
    ----------
    rollover_indices : np.ndarray
        Array of rollover positions (int64 from np.where)
    cluster_threshold : float
        Maximum spacing to consider rollovers as clustered

    Returns
    -------
    np.ndarray
        Boolean mask over rollover_indices indicating which to keep
    """
    n = len(rollover_indices)
    keep_mask = np.zeros(n, dtype=np.bool_)

    if n == 0:
        return keep_mask

    i = 0
    while i < n:
        # Keep first rollover in cluster
        keep_mask[i] = True

        # Skip subsequent rollovers in this cluster
        j = i + 1
        while j < n:
            if rollover_indices[j] - rollover_indices[j - 1] < cluster_threshold:
                j += 1
            else:
                break
        i = j

    return keep_mask


def _detect_rollovers(tof: np.ndarray, threshold: float = -10.0) -> np.ndarray:
    """
    Detect suspected rollovers using vectorized diff operation.

    A rollover is detected when the TOF value drops significantly,
    indicating the timer has reset to a new pulse period.

    Parameters
    ----------
    tof : np.ndarray
        Time-of-flight values (1D array, milliseconds)
    threshold : float, optional
        Negative threshold for detecting rollovers (default: -10.0 ms)
        A diff < threshold indicates a suspected rollover

    Returns
    -------
    np.ndarray
        Boolean mask with True at suspected rollover positions
    """
    diffs = np.diff(tof, prepend=tof[0])
    rollover_mask = diffs < threshold
    return rollover_mask


def _clean_clustered_rollovers(rollover_mask: np.ndarray) -> np.ndarray:
    """
    Clean clustered rollovers by keeping only the first in each cluster.

    Due to short-range disorder from TPX3 FIFO reordering, multiple events
    near a rollover boundary may be flagged. This function identifies clusters
    of closely-spaced rollovers and keeps only the first one in each cluster.

    Parameters
    ----------
    rollover_mask : np.ndarray
        Boolean array with True at suspected rollover positions

    Returns
    -------
    np.ndarray
        Cleaned boolean mask with clustered rollovers deduplicated
    """
    rollover_indices = np.where(rollover_mask)[0]

    if len(rollover_indices) == 0:
        return rollover_mask.copy()

    spacing = np.diff(rollover_indices)

    if len(spacing) == 0:
        return rollover_mask.copy()

    # Use 75th percentile for robust clustering
    # Assumes most rollovers are true pulses with large spacing,
    # only a few are clustered false positives with small spacing
    # Note: np.percentile is not Numba-compatible, so computed outside JIT helper
    percentile_spacing = np.percentile(spacing, 75)
    cluster_threshold = 0.5 * percentile_spacing

    # Use JIT-compiled helper for clustering loop
    keep_mask = _cluster_rollovers(rollover_indices, cluster_threshold)

    # Build output mask
    cleaned_mask = np.zeros_like(rollover_mask, dtype=bool)
    cleaned_mask[rollover_indices[keep_mask]] = True

    return cleaned_mask


def _coarse_pulse_assignment(rollover_mask: np.ndarray, data_length: int) -> np.ndarray:
    """
    Assign coarse pulse IDs based on cleaned rollover positions.

    Everything before the first rollover belongs to pulse 0.
    Everything from rollover[i] onwards belongs to pulse i+1.

    Uses Numba JIT compilation for O(N) single-pass assignment when available.

    Parameters
    ----------
    rollover_mask : np.ndarray
        Cleaned boolean mask with True at rollover positions
    data_length : int
        Total number of events in the dataset

    Returns
    -------
    np.ndarray
        Pulse ID array (int32) with coarse assignments
    """
    pulse_ids = np.zeros(data_length, dtype=np.int32)
    rollover_indices = np.where(rollover_mask)[0]

    if len(rollover_indices) == 0:
        return pulse_ids

    # Use JIT-compiled single-pass assignment (O(N) instead of O(N*R))
    _assign_pulse_ids_single_pass(pulse_ids, rollover_indices)

    return pulse_ids


@njit(cache=True)
def _refine_single_boundary(
    tof: np.ndarray,
    start_idx: int,
    end_idx: int,
    rollover_idx: int,
    late_margin: float,
) -> int:
    """
    Find optimal boundary position for a single rollover.

    This is the JIT-compiled inner loop of _refine_rollover_boundaries.
    Uses Numba for 50-100x speedup on the nested loop computation.

    Parameters
    ----------
    tof : np.ndarray
        Time-of-flight values (full array)
    start_idx : int
        Start index of the window
    end_idx : int
        End index of the window (exclusive)
    rollover_idx : int
        Index of the detected rollover
    late_margin : float
        TOF threshold for late hit detection (ms)

    Returns
    -------
    int
        Optimal boundary position
    """
    half_margin = late_margin / 2.0
    best_boundary = rollover_idx
    min_score = np.inf

    # Find optimal boundary by minimizing misclassification errors
    for candidate_boundary in range(start_idx, end_idx):
        # Count early events before boundary (should be late, so these are errors)
        errors_before = 0
        for i in range(start_idx, candidate_boundary):
            if tof[i] < half_margin:
                errors_before += 1

        # Count late events after boundary (should be early, so these are errors)
        errors_after = 0
        for i in range(candidate_boundary, end_idx):
            if tof[i] > late_margin:
                errors_after += 1

        total_errors = errors_before + errors_after

        # Add small penalty for distance from expected rollover position
        distance_penalty = 0.01 * abs(candidate_boundary - (rollover_idx + 1))
        score = total_errors + distance_penalty

        if score < min_score:
            min_score = score
            best_boundary = candidate_boundary

    # Adjust boundary based on TOF patterns
    # Move past isolated early events that appear before late events
    while best_boundary < end_idx - 1 and tof[best_boundary] < half_margin and tof[best_boundary + 1] > late_margin:
        best_boundary += 1

    # Handle edge cases at boundary
    if best_boundary < end_idx - 2 and tof[best_boundary] < half_margin and tof[best_boundary + 1] < half_margin:
        # Two consecutive low-TOF events, keep boundary here
        pass
    elif best_boundary < end_idx - 1 and tof[best_boundary] < half_margin:
        # Single low-TOF event, move past it
        best_boundary += 1

    return best_boundary


def _refine_rollover_boundaries(
    tof: np.ndarray,
    pulse_ids: np.ndarray,
    rollover_mask: np.ndarray,
    window: int = 20,
    late_margin: float = 14.0,
) -> np.ndarray:
    """
    Refine pulse assignments near rollover boundaries.

    For each rollover position, find optimal boundary by minimizing
    misclassified events within a local window. Uses Numba JIT compilation
    for the inner loop computation when available, with automatic fallback
    to pure Python.

    Parameters
    ----------
    tof : np.ndarray
        Time-of-flight values (1D array, milliseconds)
    pulse_ids : np.ndarray
        Coarse pulse ID array from _coarse_pulse_assignment
    rollover_mask : np.ndarray
        Cleaned boolean mask with True at rollover positions
    window : int, optional
        Number of events to examine on each side of rollover (default: 20)
    late_margin : float, optional
        TOF value above which events are considered late hits (default: 14.0 ms)

    Returns
    -------
    np.ndarray
        Refined pulse ID array with corrected assignments near boundaries

    Notes
    -----
    Performance: The inner loop is JIT-compiled with Numba for 50-100x speedup.
    Install with `pip install neunorm[performance]` to enable acceleration.
    """
    refined_pulse_ids = pulse_ids.copy()
    rollover_indices = np.where(rollover_mask)[0]

    if len(rollover_indices) == 0:
        return refined_pulse_ids

    n_tof = len(tof)
    for rollover_idx in rollover_indices:
        start_idx = max(0, rollover_idx - window)
        end_idx = min(n_tof, rollover_idx + window)

        pulse_before = pulse_ids[rollover_idx - 1] if rollover_idx > 0 else 0
        pulse_after = pulse_ids[rollover_idx]

        # Use JIT-compiled helper for the inner loop
        best_boundary = _refine_single_boundary(tof, start_idx, end_idx, rollover_idx, late_margin)

        # Assign pulse IDs
        refined_pulse_ids[start_idx:best_boundary] = pulse_before
        refined_pulse_ids[best_boundary:end_idx] = pulse_after

    return refined_pulse_ids


def _process_chip_worker(
    args: tuple[np.ndarray, float, int, float],
) -> np.ndarray:
    """
    Worker function for parallel chip processing.

    This function is called in separate processes by ProcessPoolExecutor.
    It must be a module-level function (not a lambda or nested function)
    to work with multiprocessing.

    Parameters
    ----------
    args : tuple
        Tuple of (chip_tof, threshold, window, late_margin)

    Returns
    -------
    np.ndarray
        Pulse ID array for this chip
    """
    chip_tof, threshold, window, late_margin = args
    return _reconstruct_pulse_ids_single_chip(chip_tof, threshold, window, late_margin)


def reconstruct_pulse_ids(
    tof: np.ndarray,
    chip_id: np.ndarray | None = None,
    threshold: float = -10.0,
    window: int = 20,
    late_margin: float = 14.0,
    n_jobs: int | None = None,
) -> np.ndarray:
    """
    Reconstruct pulse IDs from TOF data using three-pass algorithm.

    Handles TOF rollovers when neutron pulse period resets and corrects for
    short-range temporal disorder from TPX3 readout FIFOs. For multi-chip
    detectors, processes each chip independently (pulse IDs naturally sync
    because all chips measure the same physical pulses).

    Parameters
    ----------
    tof : np.ndarray
        Time-of-flight values (1D array, milliseconds)
    chip_id : np.ndarray, optional
        Chip ID for each event (0-3 for quad detector). If None, assumes
        single chip. If provided, processes each chip independently.
    threshold : float, optional
        Negative threshold for rollover detection (default: -10.0 ms)
        TOF drop below this value indicates pulse boundary
    window : int, optional
        Number of events to examine on each side of rollover (default: 20)
        for boundary refinement
    late_margin : float, optional
        TOF value above which events are considered late hits from previous
        pulse (default: 14.0 ms, appropriate for 16.67ms pulse period)
    n_jobs : int, optional
        Number of parallel workers for multi-chip processing.
        - None or 1: Sequential processing (default, safe)
        - -1: Use all available CPU cores
        - N > 1: Use N parallel workers
        Only affects multi-chip processing; single chip always runs sequentially.

    Returns
    -------
    np.ndarray
        Pulse ID array (int32) with same length as tof
        Values: 0, 1, 2, ... for sequential pulses

    Raises
    ------
    ValueError
        If n_jobs is 0 or < -1
        If chip_id length doesn't match tof length

    Notes
    -----
    The three-pass algorithm:
    1. Pass 1: Vectorized rollover detection using np.diff
    2. Pass 2: Clean clustered rollovers (false positives from disorder)
    3. Pass 2b: Coarse pulse assignment based on cleaned rollover positions
    4. Pass 3: Refine boundaries by optimizing within local window

    For multi-chip detectors (VENUS quad TPX3), each chip is processed
    independently. Pulse IDs automatically synchronize because:
    - All chips measure the same physical neutron pulses
    - TDC triggers are synchronized to pulse generation
    - Rollover detection finds same pulse boundaries per chip

    Parallel processing (n_jobs > 1 or n_jobs=-1) uses ProcessPoolExecutor
    to process each chip in a separate process. This provides ~3-4x speedup
    for 4-chip detectors on multi-core systems.

    Test accuracy: 99.67% on synthetic data with extreme temporal disorder
    (window size 8-11 events). Real TPX3 data likely has better performance.

    Examples
    --------
    Single chip detector:

    >>> from neunorm.loaders.event_loader import load_event_data
    >>> from neunorm.tof.pulse_reconstruction import reconstruct_pulse_ids
    >>> events = load_event_data('run_12557.h5')
    >>> pulse_ids = reconstruct_pulse_ids(events.tof)
    >>> print(f"Found {pulse_ids.max() + 1} pulses")

    Multi-chip detector (VENUS quad) with parallel processing:

    >>> events = load_event_data('run_14749.h5')  # Has chip_id field
    >>> pulse_ids = reconstruct_pulse_ids(
    ...     events.tof,
    ...     chip_id=events.chip_id,
    ...     threshold=-10.0,
    ...     late_margin=14.0,
    ...     n_jobs=4,  # Process 4 chips in parallel
    ... )
    >>> # Pulse IDs synchronized across all 4 chips
    >>> for chip in range(4):
    ...     mask = events.chip_id == chip
    ...     print(f"Chip {chip}: {pulse_ids[mask].max() + 1} pulses")

    Filter events by pulse:

    >>> # Skip first 5 pulses (warmup)
    >>> valid_mask = pulse_ids >= 5
    >>> events_filtered = events[valid_mask]
    """
    # Validate n_jobs parameter
    if n_jobs is not None:
        if n_jobs == 0:
            raise ValueError("n_jobs cannot be 0. Use 1 for sequential, -1 for all cores, or N > 1 for N workers.")
        if n_jobs < -1:
            raise ValueError(f"n_jobs must be -1, 1, or > 1. Got {n_jobs}.")

    # Validate chip_id length if provided
    if chip_id is not None and len(chip_id) != len(tof):
        raise ValueError(f"chip_id length ({len(chip_id)}) must match tof length ({len(tof)})")

    # Handle empty data
    if len(tof) == 0:
        return np.array([], dtype=np.int32)

    # Single chip processing (no chip_id provided)
    if chip_id is None:
        logger.info("Reconstructing pulse IDs (single chip)")
        return _reconstruct_pulse_ids_single_chip(tof, threshold, window, late_margin)

    # Multi-chip processing
    logger.info("Reconstructing pulse IDs (multi-chip)")
    pulse_ids = np.zeros(len(tof), dtype=np.int32)

    unique_chips = np.unique(chip_id)
    n_chips = len(unique_chips)
    logger.info(f"  Processing {n_chips} chips")

    # Determine effective number of workers
    effective_n_jobs = 1 if n_jobs is None else n_jobs
    if effective_n_jobs == -1:
        effective_n_jobs = min(cpu_count(), n_chips)
    elif effective_n_jobs > 1:
        effective_n_jobs = min(effective_n_jobs, n_chips)

    # Sequential processing
    if effective_n_jobs == 1:
        logger.info("  Using sequential processing")
        for chip in unique_chips:
            chip_mask = chip_id == chip
            n_events = chip_mask.sum()
            logger.info(f"  Chip {chip}: {n_events:,} events")
            pulse_ids[chip_mask] = _reconstruct_pulse_ids_single_chip(
                tof[chip_mask], threshold, window, late_margin
            )
    else:
        # Parallel processing
        logger.info(f"  Using parallel processing with {effective_n_jobs} workers")

        # Prepare arguments for each chip
        chip_masks = []
        chip_args = []
        for chip in unique_chips:
            chip_mask = chip_id == chip
            chip_masks.append(chip_mask)
            chip_tof = tof[chip_mask]
            n_events = len(chip_tof)
            logger.info(f"  Chip {chip}: {n_events:,} events")
            chip_args.append((chip_tof, threshold, window, late_margin))

        # Process chips in parallel
        with ProcessPoolExecutor(max_workers=effective_n_jobs) as executor:
            results = list(executor.map(_process_chip_worker, chip_args))

        # Combine results back into the output array
        for chip_mask, result in zip(chip_masks, results, strict=True):
            pulse_ids[chip_mask] = result

    logger.info(f"  Reconstructed pulse range: 0 - {pulse_ids.max()}")
    return pulse_ids


def _reconstruct_pulse_ids_single_chip(
    tof: np.ndarray,
    threshold: float = -10.0,
    window: int = 20,
    late_margin: float = 14.0,
) -> np.ndarray:
    """
    Reconstruct pulse IDs for single chip.

    Internal function called by reconstruct_pulse_ids for each chip.

    Parameters
    ----------
    tof : np.ndarray
        Time-of-flight values from single chip (milliseconds)
    threshold : float
        Negative threshold for rollover detection (ms)
    window : int
        Events to examine around rollover
    late_margin : float
        TOF threshold for late hit detection (ms)

    Returns
    -------
    np.ndarray
        Pulse ID array (int32)
    """
    # Pass 1: Detect rollovers
    rollover_mask = _detect_rollovers(tof, threshold=threshold)

    # Pass 2: Clean clustered rollovers
    cleaned_rollover_mask = _clean_clustered_rollovers(rollover_mask)

    # Pass 2b: Coarse pulse assignment
    coarse_pulse_ids = _coarse_pulse_assignment(cleaned_rollover_mask, len(tof))

    # Pass 3: Refine rollover regions
    final_pulse_ids = _refine_rollover_boundaries(tof, coarse_pulse_ids, cleaned_rollover_mask, window, late_margin)

    return final_pulse_ids
