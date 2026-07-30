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
from neunorm.utils.progress import STAGE_LOAD_SAMPLE, ProgressLike, resolve_progress


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

    # Convert TOF ticks to nanoseconds. Multiply by the full (possibly fractional) clock
    # period, then round to integer ns — int(tof_clock) would truncate a fractional clock
    # (e.g. the ~1.5625 ns TPX3 fine clock to 1, a ~37% error).
    tof_ns = np.round(tof_raw * tof_clock).astype(np.int64)

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
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_LOAD_SAMPLE,
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
    progress : bool or callable, optional
        Progress reporting, off by default. ``True`` draws a :mod:`tqdm` bar; a callable receives a
        :class:`~neunorm.utils.progress.ProgressEvent`. There is no item axis here, so this counts
        the four full-event-length allocations — two HDF5 slab reads, the event-id unroll and the TOF
        conversion — naming each before it runs and counting it only after it returns, so a failed
        read never reports work that did not happen. Those allocations are where the event path peaks
        in memory. See :mod:`neunorm.utils.progress`.
    stage : str, optional
        Stage label the events carry. Defaults to ``STAGE_LOAD_SAMPLE``; pass ``STAGE_LOAD_OB`` or
        ``STAGE_LOAD_DARK`` when loading those, so a callback can tell the loads of a run apart.

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

    # No item axis, but a known step sequence: one h5py slab read per dataset, then two numpy passes,
    # each allocating a full event-length array. Counting those four steps renders honestly (25, 50,
    # 75, 100%) where a single-step total=1 bar would sit at 0% and never draw a completion.
    with resolve_progress(progress, stage, total=4) as report:
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

            # No item axis, but a known four-step sequence: two h5py slab reads then two numpy
            # passes, each allocating a full event-length array. That is where the event path peaks
            # in memory.
            #
            # Each step is NAMED before it runs and COUNTED after it returns. The name is what makes
            # a stalled bar diagnostic — it says which allocation is thrashing — while counting only
            # on success means a failed read never reports work that did not happen.
            report.note(f"reading {n_events:,} events")
            event_id = event_bank_group["event_id"][:n_events].astype(np.int32)
            report()
            report.note(f"reading {n_events:,} time offsets")
            tof_raw = event_bank_group["event_time_offset"][:n_events].astype(np.float64)
            report()

        # Unroll event_id to x, y pixel coordinates
        # event_id is linearized: pixel_id = y * y_bins + x
        x_bins, y_bins = detector_shape
        report.note("unrolling event ids to x, y")
        y = ((event_id - event_id_offset) // y_bins).astype(np.int32)
        x = ((event_id - event_id_offset) % y_bins).astype(np.int32)
        report()

        # Convert TOF from microseconds to nanoseconds
        report.note("converting tof to ns")
        tof_ns = (tof_raw * 1000).astype(np.int64)
        report()

        if n_events:
            # Guarded because ndarray.min() on an empty array raises: an empty bank is a legitimate
            # (if useless) file, and it should not turn a diagnostic log line into a crash.
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
