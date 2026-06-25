"""
Event data loader for TPX3/TPX4 HDF5 files.

Loads raw event-mode data from HDF5 files containing tof, x, y arrays.
"""

from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
from loguru import logger

from neunorm.data_models.core import EventData


def load_event_data(
    file_path: Union[str, Path], tof_clock: float = 25.0, max_events: Optional[int] = None
) -> EventData:
    """
    Load event-mode data from HDF5 file.

    Expected HDF5 structure:
    - 'tof': Time-of-flight values (int32/int64 array)
    - 'x': X pixel coordinates (int16/int32 array)
    - 'y': Y pixel coordinates (int16/int32 array)

    Parameters
    ----------
    file_path : str or Path
        Path to HDF5 file containing event data
    tof_clock : float, optional
        TOF clock period in nanoseconds (default: 25.0 for TPX3)
        Raw TOF ticks are multiplied by this value
    max_events : int, optional
        Maximum number of events to load (for testing/memory limits)
        If None, loads all events

    Returns
    -------
    EventData
        Pydantic model containing event arrays and metadata

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    KeyError
        If HDF5 file missing required fields (tof, x, y)

    Examples
    --------
    >>> events = load_event_data('run_12557.h5', tof_clock=25.0)
    >>> print(f"Loaded {events.total_events:,} events")
    >>> print(f"TOF range: {events.tof.min()} - {events.tof.max()} ns")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Event data file not found: {file_path}")

    logger.info(f"Loading event data from {file_path.name}")

    with h5py.File(file_path, "r") as f:
        # Check required fields exist
        required_fields = ["tof", "x", "y"]
        for field in required_fields:
            if field not in f:
                raise KeyError(
                    f"HDF5 file missing required field '{field}'. "
                    f"Expected fields: {required_fields}, found: {list(f.keys())}"
                )

        # Get total event count
        total_events_in_file = f["tof"].shape[0]

        # Determine how many events to load
        if max_events is not None:
            n_events = min(max_events, total_events_in_file)
            logger.info(f"  Loading {n_events:,} / {total_events_in_file:,} events (max_events={max_events:,})")
        else:
            n_events = total_events_in_file
            logger.info(f"  Loading {n_events:,} events")

        # Load arrays (convert dtype as needed)
        tof_raw = f["tof"][:n_events].astype(np.int64)
        x = f["x"][:n_events].astype(np.int32)
        y = f["y"][:n_events].astype(np.int32)

    # Convert TOF ticks to nanoseconds
    tof_ns = tof_raw * int(tof_clock)

    logger.info(f"  TOF range: {tof_ns.min():,} - {tof_ns.max():,} ns")
    logger.info(f"  X range: [{x.min()}, {x.max()}]")
    logger.info(f"  Y range: [{y.min()}, {y.max()}]")

    # Create EventData model (validation runs automatically via model_validator)
    events = EventData(tof=tof_ns, x=x, y=y, file_path=file_path, total_events=n_events, tof_clock=tof_clock)

    logger.success(f"✓ Loaded {events.total_events:,} events from {file_path.name}")

    return events


def load_event_nexus(  # noqa: C901
    file_path: Union[str, Path],
    detector_bank: str = "bank1",
    detector_shape: tuple[int, int] = (512, 512),
    event_id_offset: int = 0,
    max_events: Optional[int] = None,
) -> EventData:
    """
    Load event-mode data from SNS NeXus HDF5 file.

    Reads from the SNS NeXus event bank structure:
    /entry/<bank>_events/ with datasets:
    - event_id: Linearized pixel detector IDs
    - event_time_offset: Time-of-flight values (in microseconds)

    Parameters
    ----------
    file_path : str or Path
        Path to NeXus HDF5 file containing event data
    detector_bank : str
        Specific detector bank to load (e.g., 'bank100').
    detector_shape : tuple[int, int], optional
        Detector dimensions (x_bins, y_bins) for unrolling event_id to x, y.
        Default: (512, 512) for SNS VENUS detectors
    event_id_offset : int, optional
        Base offset subtracted from each event_id before unrolling to x, y
        pixel coordinates (default: 0)
    max_events : int, optional
        Maximum number of events to load (for testing/memory limits)
        If None, loads all events

    Returns
    -------
    EventData
        Pydantic model containing event arrays (tof in ns, x, y) and metadata

    Raises
    ------
    FileNotFoundError
        If file doesn't exist
    KeyError
        If the NeXus structure, required fields, or the requested
        detector_bank are not found

    Examples
    --------
    >>> # Use the default detector bank ('bank1')
    >>> events = load_event_nexus('VENUS_15159.nxs.h5')

    >>> # Specify detector bank
    >>> events = load_event_nexus('VENUS_15159.nxs.h5', detector_bank='bank100')

    >>> # Custom detector shape
    >>> events = load_event_nexus('file.nxs.h5', detector_shape=(256, 256))
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"NeXus file not found: {file_path}")

    logger.info(f"Loading SNS NeXus event data from {file_path.name}")

    with h5py.File(file_path, "r") as f:
        # Navigate to entry group
        if "entry" not in f:
            raise KeyError("NeXus 'entry' group not found in file")

        entry = f["entry"]

        # Find event bank group
        bank_key = f"{detector_bank}_events" if not detector_bank.endswith("_events") else detector_bank

        if bank_key not in entry:
            raise KeyError(
                f"Detector bank '{bank_key}' not found under 'entry'. Available groups: {list(entry.keys())}"
            )
        event_bank_group = entry[bank_key]

        logger.info(f"  Using detector bank: {detector_bank}")

        # Check for required datasets
        required_fields = ["event_id", "event_time_offset"]
        for field in required_fields:
            if field not in event_bank_group:
                raise KeyError(
                    f"Dataset '{field}' not found in {detector_bank}_events. Found: {list(event_bank_group.keys())}"
                )

        # Get total event count
        total_events_in_file = event_bank_group["event_id"].shape[0]

        # Determine how many events to load
        if max_events is not None:
            n_events = min(max_events, total_events_in_file)
            logger.info(f"  Loading {n_events:,} / {total_events_in_file:,} events (max_events={max_events:,})")
        else:
            n_events = total_events_in_file
            logger.info(f"  Loading {n_events:,} events")

        # Load event_id and time_offset
        event_id = event_bank_group["event_id"][:n_events].astype(np.int32)
        tof_raw = event_bank_group["event_time_offset"][:n_events].astype(np.float64)

    # Unroll event_id to x, y pixel coordinates
    # event_id is linearized: pixel_id = y * y_bins + x
    x_bins, y_bins = detector_shape
    y = ((event_id - event_id_offset) // y_bins).astype(np.int32)
    x = ((event_id - event_id_offset) % y_bins).astype(np.int32)

    # Convert TOF from microseconds to nanoseconds
    tof_ns = (tof_raw * 1000).astype(np.int64)

    logger.info(f"  TOF range: {tof_ns.min():,} - {tof_ns.max():,} ns")
    logger.info(f"  X range: [{x.min()}, {x.max()}]")
    logger.info(f"  Y range: [{y.min()}, {y.max()}]")

    # Create EventData model
    events = EventData(
        tof=tof_ns,
        x=x,
        y=y,
        file_path=file_path,
        total_events=n_events,
    )

    logger.success(f"✓ Loaded {events.total_events:,} events from {file_path.name}")

    return events
