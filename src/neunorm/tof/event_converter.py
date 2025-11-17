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


def convert_events_to_histogram(
    events: EventData,
    binning: BinningConfig,
    flight_path: sc.Variable,
    x_bins: int = 514,
    y_bins: int = 514,
    chunk_size: int = 500_000_000,
    compute_variance: bool = True,
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

    # Create TOF bin edges
    tof_bins = create_tof_bins(binning, flight_path)
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

        # Progress
        if n_chunks > 1 and ((i + 1) % 10 == 0 or (i + 1) == n_chunks):
            progress = (i + 1) / n_chunks * 100
            logger.info(f"    Progress: {i + 1}/{n_chunks} chunks ({progress:.1f}%)")

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
        hist_3d = attach_poisson_variance(hist_3d)
        logger.info("  ✓ Poisson variance attached")

    logger.success(f"✓ Histogram created: shape={hist_3d.shape}")

    return hist_3d
