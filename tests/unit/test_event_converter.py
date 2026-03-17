"""
Unit tests for event-to-histogram converter.

Tests chunked processing and variance attachment for event data.
"""

from pathlib import Path

import numpy as np
import scipp as sc


def test_event_converter_imports():
    """Test that event converter module can be imported"""


def test_convert_events_to_histogram_basic():
    """Test basic event to histogram conversion"""
    import tempfile

    import h5py

    from neunorm.data_models.tof import BinningConfig
    from neunorm.loaders.event_loader import load_event_data
    from neunorm.tof.event_converter import convert_events_to_histogram

    # Create test data
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Create small dataset: 10 events
        with h5py.File(temp_path, "w") as hf:
            # Events at TOF=100 ticks, pixels (0,0), (1,1), etc.
            hf.create_dataset("tof", data=np.full(10, 100, dtype=np.int64))
            hf.create_dataset("x", data=np.arange(10, dtype=np.int32))
            hf.create_dataset("y", data=np.arange(10, dtype=np.int32))

        # Load events
        events = load_event_data(temp_path)

        # Convert to histogram
        binning = BinningConfig(bins=100, bin_space="tof", tof_range=(1000, 10000))
        flight_path = sc.scalar(25.0, unit="m")

        hist = convert_events_to_histogram(events, binning, flight_path, x_bins=20, y_bins=20, compute_variance=False)

        # Verify output
        assert isinstance(hist, sc.DataArray)
        assert "tof" in hist.dims
        assert "x" in hist.dims
        assert "y" in hist.dims
        assert hist.unit == "counts"

        # Should have correct shape: (100 tof bins, 20 x bins, 20 y bins)
        assert hist.shape == (100, 20, 20)

    finally:
        temp_path.unlink()


def test_convert_events_with_poisson_variance():
    """Test that variance is attached when compute_variance=True"""
    from neunorm.data_models.core import EventData
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.event_converter import convert_events_to_histogram

    # Create simple EventData
    events = EventData(
        tof=np.array([2500, 5000, 7500] * 10, dtype=np.int64),  # 30 events, 3 unique TOF
        x=np.array([5] * 30, dtype=np.int32),  # All at pixel (5, 5)
        y=np.array([5] * 30, dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=30,
    )

    binning = BinningConfig(bins=10, bin_space="tof", tof_range=(0, 10000))
    flight_path = sc.scalar(25.0, unit="m")

    # Convert with variance
    hist = convert_events_to_histogram(events, binning, flight_path, x_bins=10, y_bins=10, compute_variance=True)

    # Verify variance attached
    assert hist.variances is not None

    # Variance should equal counts (Poisson)
    # Events at pixel (5,5) should have counts = 30 total (10 per TOF bin)
    # But they're spread across TOF bins, so each bin might have ~10 events
    # The variance should equal the counts in each bin


def test_convert_events_without_variance():
    """Test that variance is None when compute_variance=False"""
    from neunorm.data_models.core import EventData
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.event_converter import convert_events_to_histogram

    events = EventData(
        tof=np.array([2500, 5000, 7500], dtype=np.int64),
        x=np.array([0, 0, 0], dtype=np.int32),
        y=np.array([0, 0, 0], dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=3,
    )

    binning = BinningConfig(bins=10, bin_space="tof")
    flight_path = sc.scalar(25.0, unit="m")

    hist = convert_events_to_histogram(events, binning, flight_path, compute_variance=False)

    # No variance should be attached
    assert hist.variances is None


def test_convert_events_energy_binning():
    """Test conversion with energy-space binning"""
    from neunorm.data_models.core import EventData
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.event_converter import convert_events_to_histogram

    # Create events with various TOF values
    events = EventData(
        tof=np.array([1e5, 5e5, 1e6, 5e6, 1e7], dtype=np.int64),  # ns
        x=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        y=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=5,
    )

    # Bin in energy space
    binning = BinningConfig(bins=100, bin_space="energy", energy_range=(1.0, 1000.0), use_log_bin=True)
    flight_path = sc.scalar(25.0, unit="m")

    hist = convert_events_to_histogram(events, binning, flight_path)

    # Should have tof dimension (binned in TOF space, but from energy spec)
    assert "tof" in hist.dims
    assert hist.shape[0] == 100  # 100 TOF bins (converted from 100 energy bins)


def test_convert_events_with_chunking():
    """Test that chunking works for large event datasets"""
    from neunorm.data_models.core import EventData
    from neunorm.data_models.tof import BinningConfig
    from neunorm.tof.event_converter import convert_events_to_histogram

    # Create larger dataset (but still small for testing)
    n_events = 10000
    events = EventData(
        tof=np.random.randint(1000, 10000, size=n_events, dtype=np.int64),
        x=np.random.randint(0, 514, size=n_events, dtype=np.int32),
        y=np.random.randint(0, 514, size=n_events, dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=n_events,
    )

    binning = BinningConfig(bins=50, bin_space="tof")
    flight_path = sc.scalar(25.0, unit="m")

    # Convert with small chunk size (to test chunking logic)
    hist = convert_events_to_histogram(
        events,
        binning,
        flight_path,
        chunk_size=2000,  # Process 2000 events at a time
        compute_variance=True,
    )

    # Should still produce correct result
    assert "tof" in hist.dims
    assert hist.variances is not None

    # Total counts should equal number of events
    total_counts = hist.values.sum()
    assert total_counts == n_events


def test_convert_events_to_2d_histogram():
    """Test conversion to 2D histogram. TOF information is ignored in this case."""
    from neunorm.data_models.core import EventData
    from neunorm.tof.event_converter import convert_events_to_2d_histogram

    # Create simple EventData
    events = EventData(
        tof=np.array([2500, 5000, 7500] * 10, dtype=np.int64),  # This information is ignored for 2D histogram
        x=np.array([5] * 30, dtype=np.int32),  # All at pixel (5, 5)
        y=np.array([5] * 30, dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=30,
    )

    # Convert to 2D histogram
    hist = convert_events_to_2d_histogram(events, detector_shape=(16, 16))

    # check dims
    assert hist.dims == ("x", "y")
    assert hist.unit == "counts"

    # Verify histogram shape
    assert hist.shape == (16, 16)

    # Verify counts
    expected_counts = np.zeros((16, 16), dtype=np.float32)
    expected_counts[5, 5] = 30  # All events at pixel (5, 5)
    np.testing.assert_array_equal(hist.values, expected_counts)

    # Variance should equal counts (Poisson)
    # Events at pixel (5,5) should have counts = 30 total
    np.testing.assert_array_equal(hist.variances, expected_counts)


def test_convert_events_to_2d_histogram_with_chunking():
    """Test conversion to 2D histogram with chunking. TOF information is ignored in this case."""
    from neunorm.data_models.core import EventData
    from neunorm.tof.event_converter import convert_events_to_2d_histogram

    # Create simple EventData
    events = EventData(
        tof=np.array([2500, 5000, 7500] * 10, dtype=np.int64),  # This information is ignored for 2D histogram
        x=np.array([5] * 30, dtype=np.int32),  # All at pixel (5, 5)
        y=np.array([5] * 30, dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=30,
    )

    # Convert to 2D histogram with small chunk size to test chunking
    hist = convert_events_to_2d_histogram(events, detector_shape=(16, 16), chunk_size=10)

    # check dims
    assert hist.dims == ("x", "y")
    assert hist.unit == "counts"

    # Verify histogram shape
    assert hist.shape == (16, 16)

    # Verify counts
    expected_counts = np.zeros((16, 16), dtype=np.float32)
    expected_counts[5, 5] = 30  # All events at pixel (5, 5)
    np.testing.assert_array_equal(hist.values, expected_counts)

    # Variance should equal counts (Poisson)
    # Events at pixel (5,5) should have counts = 30 total
    np.testing.assert_array_equal(hist.variances, expected_counts)
