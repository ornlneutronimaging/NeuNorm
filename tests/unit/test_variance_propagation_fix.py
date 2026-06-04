"""
Test for correct variance propagation in chunked event conversion.

This test verifies that Poisson variance is correctly propagated when
accumulating multiple histogram chunks.
"""

from pathlib import Path

import numpy as np
import scipp as sc


def test_chunked_variance_propagation_is_correct():
    """
    Test that variance is correctly accumulated across chunks.

    When histogramming in chunks:
    1. Each chunk should have variance attached BEFORE accumulation
    2. Scipp should propagate variance through addition
    3. Final variance = sum of individual variances

    This is the CORRECT statistical approach for independent measurements.
    """
    from neunorm.data_models.core import EventData
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.event_converter import convert_events_to_histogram

    # Create events that will be split into 2 chunks
    # 6000 events total, chunk_size=3000 → 2 chunks
    # All events go to same TOF bin and spatial pixel
    n_events = 6000
    events = EventData(
        tof=np.full(n_events, 5000, dtype=np.int64),  # All at same TOF
        x=np.full(n_events, 100, dtype=np.int32),  # All at pixel (100, 100)
        y=np.full(n_events, 100, dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=n_events,
    )

    binning = BinningConfig(bins=10, bin_space="tof", tof_range=(0, 10000))
    flight_path = sc.scalar(25.0, unit="m")

    # Convert with chunking
    hist = convert_events_to_histogram(
        events,
        binning,
        flight_path,
        x_bins=200,
        y_bins=200,
        chunk_size=3000,  # Force 2 chunks
        compute_variance=True,
    )

    # All 6000 events in one bin at pixel (100, 100)
    counts_at_pixel = hist.values[:, 100, 100].sum()
    variance_at_pixel = hist.variances[:, 100, 100].sum()

    # For Poisson: variance should equal counts
    # If variance attached correctly (before accumulation):
    #   Chunk 1: counts=3000, var=3000
    #   Chunk 2: counts=3000, var=3000
    #   Total: counts=6000, var=6000 ✓ CORRECT
    #
    # If variance attached incorrectly (after accumulation):
    #   Chunk 1: counts=3000
    #   Chunk 2: counts=3000
    #   Total counts=6000, then var=6000
    #   But Scipp sees no variance in chunks, so propagation fails!

    assert counts_at_pixel == 6000

    # CRITICAL TEST: Variance should equal counts for Poisson
    # This will FAIL with current implementation!
    np.testing.assert_allclose(variance_at_pixel, counts_at_pixel, rtol=0.01)


def test_variance_in_single_chunk_vs_multi_chunk_identical():
    """
    Test that single-chunk and multi-chunk give same variance.

    Statistical correctness requires:
    - Single chunk: var = N
    - Multi chunk: var = var1 + var2 + ... = N

    Results must be identical.
    """
    from neunorm.data_models.core import EventData
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.event_converter import convert_events_to_histogram

    # Create same event data
    n_events = 10000
    events = EventData(
        tof=np.random.randint(1000, 9000, size=n_events, dtype=np.int64),
        x=np.random.randint(0, 50, size=n_events, dtype=np.int32),
        y=np.random.randint(0, 50, size=n_events, dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=n_events,
    )

    binning = BinningConfig(bins=50, bin_space="tof")
    flight_path = sc.scalar(25.0, unit="m")

    # Convert with single chunk (reference)
    hist_single = convert_events_to_histogram(
        events,
        binning,
        flight_path,
        x_bins=50,
        y_bins=50,
        chunk_size=1000000,  # Single chunk
        compute_variance=True,
    )

    # Convert with multiple chunks
    hist_multi = convert_events_to_histogram(
        events,
        binning,
        flight_path,
        x_bins=50,
        y_bins=50,
        chunk_size=2000,  # 5 chunks
        compute_variance=True,
    )

    # Counts should be identical
    np.testing.assert_array_equal(hist_single.values, hist_multi.values)

    # CRITICAL: Variances should be identical
    # This will FAIL with current implementation!
    np.testing.assert_allclose(hist_single.variances, hist_multi.variances, rtol=0.01)
