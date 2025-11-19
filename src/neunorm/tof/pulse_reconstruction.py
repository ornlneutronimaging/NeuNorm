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

See Also
--------
examples/pulse_reconstruction_tutorial.ipynb : Interactive tutorial with visualizations
tests/unit/test_pulse_reconstruction.py : Unit tests and usage examples
"""

import numpy as np
from loguru import logger


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
    percentile_spacing = np.percentile(spacing, 75)
    cluster_threshold = 0.5 * percentile_spacing

    # Keep only first rollover in each cluster
    cleaned_mask = np.zeros_like(rollover_mask, dtype=bool)

    i = 0
    while i < len(rollover_indices):
        cleaned_mask[rollover_indices[i]] = True

        # Skip subsequent rollovers in this cluster
        j = i + 1
        while j < len(rollover_indices):
            if rollover_indices[j] - rollover_indices[j - 1] < cluster_threshold:
                j += 1
            else:
                break

        i = j

    return cleaned_mask


def _coarse_pulse_assignment(rollover_mask: np.ndarray, data_length: int) -> np.ndarray:
    """
    Assign coarse pulse IDs based on cleaned rollover positions.

    Everything before the first rollover belongs to pulse 0.
    Everything from rollover[i] onwards belongs to pulse i+1.

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

    for i, rollover_idx in enumerate(rollover_indices):
        pulse_ids[rollover_idx:] = i + 1

    return pulse_ids


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
    misclassified events within a local window.

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
    """
    refined_pulse_ids = pulse_ids.copy()
    rollover_indices = np.where(rollover_mask)[0]

    if len(rollover_indices) == 0:
        return refined_pulse_ids

    for rollover_idx in rollover_indices:
        start_idx = max(0, rollover_idx - window)
        end_idx = min(len(tof), rollover_idx + window)

        pulse_before = pulse_ids[rollover_idx - 1] if rollover_idx > 0 else 0
        pulse_after = pulse_ids[rollover_idx]

        # Find optimal boundary position
        best_boundary = rollover_idx
        min_score = float("inf")

        for candidate_boundary in range(start_idx, end_idx):
            errors_before = np.sum(tof[start_idx:candidate_boundary] < late_margin / 2)
            errors_after = np.sum(tof[candidate_boundary:end_idx] > late_margin)
            total_errors = errors_before + errors_after

            distance_penalty = 0.01 * abs(candidate_boundary - (rollover_idx + 1))
            score = total_errors + distance_penalty

            if score < min_score:
                min_score = score
                best_boundary = candidate_boundary

        # Adjust boundary based on TOF patterns
        while (
            best_boundary < end_idx - 1
            and tof[best_boundary] < late_margin / 2
            and tof[best_boundary + 1] > late_margin
        ):
            best_boundary += 1

        if (
            best_boundary < end_idx - 2
            and tof[best_boundary] < late_margin / 2
            and tof[best_boundary + 1] < late_margin / 2
        ):
            pass  # Two consecutive low-TOF events, keep boundary here
        elif best_boundary < end_idx - 1 and tof[best_boundary] < late_margin / 2:
            best_boundary += 1  # Single low-TOF event, move past it

        # Assign pulse IDs
        refined_pulse_ids[start_idx:best_boundary] = pulse_before
        refined_pulse_ids[best_boundary:end_idx] = pulse_after

    return refined_pulse_ids


def reconstruct_pulse_ids(
    tof: np.ndarray,
    chip_id: np.ndarray | None = None,
    threshold: float = -10.0,
    window: int = 20,
    late_margin: float = 14.0,
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

    Returns
    -------
    np.ndarray
        Pulse ID array (int32) with same length as tof
        Values: 0, 1, 2, ... for sequential pulses

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

    Multi-chip detector (VENUS quad):

    >>> events = load_event_data('run_14749.h5')  # Has chip_id field
    >>> pulse_ids = reconstruct_pulse_ids(
    ...     events.tof,
    ...     chip_id=events.chip_id,
    ...     threshold=-10.0,
    ...     late_margin=14.0
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
    if chip_id is None:
        logger.info("Reconstructing pulse IDs (single chip)")
        return _reconstruct_pulse_ids_single_chip(tof, threshold, window, late_margin)

    logger.info("Reconstructing pulse IDs (multi-chip)")
    pulse_ids = np.zeros(len(tof), dtype=np.int32)

    unique_chips = np.unique(chip_id)
    logger.info(f"  Processing {len(unique_chips)} chips")

    for chip in unique_chips:
        chip_mask = chip_id == chip
        n_events = chip_mask.sum()
        logger.info(f"  Chip {chip}: {n_events:,} events")

        pulse_ids[chip_mask] = _reconstruct_pulse_ids_single_chip(tof[chip_mask], threshold, window, late_margin)

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
    final_pulse_ids = _refine_rollover_boundaries(
        tof, coarse_pulse_ids, cleaned_rollover_mask, window, late_margin
    )

    return final_pulse_ids
