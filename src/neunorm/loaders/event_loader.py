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

    # Create EventData model
    events = EventData(tof=tof_ns, x=x, y=y, file_path=file_path, total_events=n_events, tof_clock=tof_clock)

    # Validate arrays have consistent lengths
    events.validate_lengths()

    logger.success(f"✓ Loaded {events.total_events:,} events from {file_path.name}")

    return events
