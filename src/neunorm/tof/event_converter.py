"""
Event-to-histogram converter for TOF event data.

Converts event-mode data to 3D histograms using chunked processing
for memory efficiency. Ported from venus_tof with enhancements.
"""

import numpy as np
import scipp as sc
from loguru import logger

from neunorm.data_models.core import EventData
from neunorm.data_models.tof import BinningConfig
from neunorm.processing.uncertainty_calculator import attach_poisson_variance
from neunorm.tof.binning import create_tof_bins
from neunorm.utils.progress import STAGE_HISTOGRAM, ProgressLike, resolve_progress


def convert_events_to_histogram(
    events: EventData,
    binning: BinningConfig,
    flight_path: sc.Variable,
    x_bins: int = 514,
    y_bins: int = 514,
    chunk_size: int = 500_000_000,
    compute_variance: bool = True,
    detector_time_offset: sc.Variable = sc.scalar(0, unit="us"),
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_HISTOGRAM,
) -> sc.DataArray:
    """
    Convert event-mode data to 3D TOF histogram.

    Uses chunked processing for memory efficiency (can handle billions of events).
    Optionally attaches Poisson variance for uncertainty quantification.

    Parameters
    ----------
    events : EventData
        Event data (tof, x, y arrays)
    binning : BinningConfig
        TOF/energy/wavelength binning configuration
    flight_path : sc.Variable
        Flight path in meters (required for energy/wavelength binning)
    x_bins : int, optional
        Number of spatial bins in x (default: 514, native TPX3 resolution)
    y_bins : int, optional
        Number of spatial bins in y (default: 514)
    chunk_size : int, optional
        Events per chunk for processing (default: 500M)
        Larger = faster but more memory
    compute_variance : bool, optional
        Attach Poisson variance (var = counts). Default: True
    detector_time_offset : sc.Variable, optional
        Detector time offset (e.g. TIDelay) applied when building energy/wavelength bin edges
        so they live in raw detector-TOF space, matching the raw event TOF histogrammed into
        them. Default: 0 us. Has no effect for ``bin_space='tof'``.
    progress : bool or callable, optional
        Progress reporting, off by default. Emits one event per chunk, plus a note before the
        variance attach. At the default 500M chunk size a typical run is a single chunk, so this is
        normally one tick; it becomes a moving bar for billion-event datasets.
        See :mod:`neunorm.utils.progress`.
    stage : str, optional
        Stage label the events carry.

    Returns
    -------
    sc.DataArray
        3D histogram (tof, x, y) with optional variance

    Notes
    -----
    Chunked processing allows handling datasets larger than RAM.
    Based on venus_tof implementation with performance optimizations.

    Examples
    --------
    >>> events = load_event_data('data.h5')
    >>> binning = BinningConfig(bins=5000, bin_space='energy', energy_range=(1, 100))
    >>> hist = convert_events_to_histogram(events, binning, flight_path=sc.scalar(25.0, unit='m'))
    >>> print(hist.shape)  # (5000, 514, 514)
    """
    logger.info(f"Converting {events.total_events:,} events to histogram")
    logger.info(f"  TOF bins: {binning.bins}, Spatial: {x_bins}×{y_bins}")
    logger.info(f"  Bin space: {binning.bin_space}, Log: {binning.use_log_bin}")

    # Create TOF bin edges (raw detector-TOF; detector_time_offset shifts energy/wavelength
    # edges so binning agrees with the later coordinate labeling).
    tof_bins = create_tof_bins(binning, flight_path, detector_time_offset)
    logger.info(f"  TOF range: {tof_bins.values.min():.1f} - {tof_bins.values.max():.1f} ns")

    # Create spatial bin edges (explicit to ensure consistency across chunks)
    x_edges = sc.arange("x", 0, x_bins + 1, unit=sc.units.dimensionless)
    y_edges = sc.arange("y", 0, y_bins + 1, unit=sc.units.dimensionless)

    # Process in chunks
    n_events = events.total_events
    n_chunks = int(np.ceil(n_events / chunk_size))

    if n_chunks > 1:
        logger.info(f"  Processing in {n_chunks} chunks of {chunk_size:,} events...")

    hist_3d = None

    # One event per chunk. At the default 500M chunk size a typical run is a single chunk, so this is
    # normally one "the stage is running" tick rather than a moving bar; it becomes a real
    # progression only for billion-event datasets, which is exactly when it is wanted. This replaces
    # an ad-hoc `logger.info` percent print that fired every tenth chunk and so produced nothing at
    # all for any run under 5 billion events.
    with resolve_progress(progress, stage, total=n_chunks) as report:
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_events)
            n_chunk = end_idx - start_idx

            # Extract chunk
            tof_chunk = events.tof[start_idx:end_idx]
            x_chunk = events.x[start_idx:end_idx]
            y_chunk = events.y[start_idx:end_idx]

            # Create scipp event DataArray for this chunk
            events_chunk = sc.DataArray(
                data=sc.ones(dims=["event"], shape=[n_chunk], unit="counts", dtype="float32"),
                coords={
                    "tof": sc.array(dims=["event"], values=tof_chunk, unit="ns"),
                    "x": sc.array(dims=["event"], values=x_chunk),
                    "y": sc.array(dims=["event"], values=y_chunk),
                },
            )

            # Histogram chunk (use explicit bin edges for spatial dims)
            hist_chunk = events_chunk.hist(tof=tof_bins, x=x_edges, y=y_edges)

            # Accumulate
            if hist_3d is None:
                hist_3d = hist_chunk
            else:
                hist_3d += hist_chunk

            report(detail=f"chunk {i + 1} of {n_chunks}")

        # Handle case with no events (create empty histogram)
        if hist_3d is None:
            logger.warning("No events in dataset, creating empty histogram")
            hist_3d = sc.DataArray(
                data=sc.zeros(
                    dims=["tof", "x", "y"], shape=[len(tof_bins) - 1, x_bins, y_bins], unit="counts", dtype="float32"
                ),
                coords={"tof": tof_bins, "x": x_edges, "y": y_edges},
            )

        # Attach Poisson variance if requested
        if compute_variance:
            report.note("attaching Poisson variance")
            hist_3d = attach_poisson_variance(hist_3d)
            logger.info("  ✓ Poisson variance attached")

        logger.success(f"✓ Histogram created: shape={hist_3d.shape}")

        return hist_3d


def convert_events_to_2d_histogram(
    events: EventData,
    detector_shape: tuple[int, int],
    chunk_size: int = 500_000_000,
    *,
    progress: ProgressLike = False,
    stage: str = STAGE_HISTOGRAM,
) -> sc.DataArray:
    """Convert events to 2D spatial histogram (no TOF).

    Parameters
    ----------
    events : EventData
        Event data (tof, x, y arrays)
    detector_shape : tuple[int, int]
        (x_bins, y_bins) defining the spatial dimensions of the histogram
    chunk_size : int, optional
        Events per chunk for processing (default: 500M)
        Larger = faster but more memory
    progress : bool or callable, optional
        Progress reporting, off by default. Emits one event per chunk, plus a note before the
        variance attach. This is the converter ``mars_tpx3`` uses, so instrumenting only the 3-D
        :func:`convert_events_to_histogram` would leave that pipeline silent.
        See :mod:`neunorm.utils.progress`.
    stage : str, optional
        Stage label the events carry.

    Returns
    -------
    sc.DataArray
        2D histogram (x, y) with counts
    """

    x_bins, y_bins = detector_shape
    x_edges = sc.arange("x", 0, x_bins + 1, unit=sc.units.dimensionless)
    y_edges = sc.arange("y", 0, y_bins + 1, unit=sc.units.dimensionless)

    # Initialize accumulator histogram with zeros to enable chunked accumulation.
    hist_2d = sc.DataArray(
        data=sc.zeros(
            dims=["x", "y"],
            shape=[x_bins, y_bins],
            unit="counts",
            dtype="float32",
        ),
        coords={
            "x": x_edges,  # bin centers are implicit; store lower edges
            "y": y_edges,
        },
    )

    n_events = len(events)
    # A zero-event run still reports, matching the 3-D converter: two sibling functions must not give
    # a caller two different contracts for the same input.
    n_chunks = int(np.ceil(n_events / chunk_size))
    with resolve_progress(progress, stage, total=n_chunks) as report:
        if n_events == 0:
            # No events: just attach variance to the empty histogram and return.
            report.note("attaching Poisson variance")
            return attach_poisson_variance(hist_2d)
        # Process events in chunks to keep memory usage bounded.
        for index, start in enumerate(range(0, n_events, chunk_size), start=1):
            end = min(start + chunk_size, n_events)
            chunk_len = end - start
            events_2d_chunk = sc.DataArray(
                data=sc.ones(
                    dims=["event"],
                    shape=[chunk_len],
                    unit="counts",
                    dtype="float32",
                ),
                coords={
                    "x": sc.array(
                        dims=["event"],
                        values=events.x[start:end],
                        unit="",
                    ),
                    "y": sc.array(
                        dims=["event"],
                        values=events.y[start:end],
                        unit="",
                    ),
                },
            )
            hist_2d_chunk = events_2d_chunk.hist(x=x_edges, y=y_edges)
            hist_2d.data += hist_2d_chunk.data
            report(detail=f"chunk {index} of {n_chunks}")

        # Attach Poisson variance
        report.note("attaching Poisson variance")
        return attach_poisson_variance(hist_2d)
