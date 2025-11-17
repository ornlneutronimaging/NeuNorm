"""
Unit tests for event data loader.

Tests HDF5 event file loading for TPX3/TPX4 detectors.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest


def test_event_loader_imports():
    """Test that event loader module can be imported"""


def test_load_event_data_from_hdf5():
    """Test loading event data from HDF5 file"""
    from neunorm.loaders.event_loader import load_event_data

    # Create temporary HDF5 file with test data
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Create test HDF5 file
        with h5py.File(temp_path, "w") as hf:
            # TPX3 format: tof, x, y arrays
            hf.create_dataset("tof", data=np.array([1000, 2000, 3000, 4000], dtype=np.int64))
            hf.create_dataset("x", data=np.array([100, 200, 150, 250], dtype=np.int32))
            hf.create_dataset("y", data=np.array([300, 350, 325, 375], dtype=np.int32))

        # Load events
        events = load_event_data(temp_path)

        # Verify EventData returned
        from neunorm.data_models.core import EventData

        assert isinstance(events, EventData)

        # Verify data
        assert len(events) == 4
        assert events.total_events == 4
        # TOF values are multiplied by clock (default 25 ns): [1000*25, 2000*25, ...]
        np.testing.assert_array_equal(events.tof, [25000, 50000, 75000, 100000])
        np.testing.assert_array_equal(events.x, [100, 200, 150, 250])
        np.testing.assert_array_equal(events.y, [300, 350, 325, 375])
        assert events.file_path == temp_path

    finally:
        temp_path.unlink()


def test_load_event_data_with_tof_clock_conversion():
    """Test that TOF values are converted using tof_clock"""
    from neunorm.loaders.event_loader import load_event_data

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Create HDF5 with raw TOF ticks (need to multiply by clock period)
        with h5py.File(temp_path, "w") as hf:
            # Raw ticks: 1, 2, 3, 4
            hf.create_dataset("tof", data=np.array([1, 2, 3, 4], dtype=np.int64))
            hf.create_dataset("x", data=np.array([0, 0, 0, 0], dtype=np.int32))
            hf.create_dataset("y", data=np.array([0, 0, 0, 0], dtype=np.int32))

        # Load with clock period = 25 ns (TPX3 default)
        events = load_event_data(temp_path, tof_clock=25.0)

        # TOF should be ticks * clock_period: [25, 50, 75, 100] ns
        np.testing.assert_array_equal(events.tof, [25, 50, 75, 100])
        assert events.tof_clock == 25.0

    finally:
        temp_path.unlink()


def test_load_event_data_missing_file():
    """Test error handling for missing file"""
    from neunorm.loaders.event_loader import load_event_data

    with pytest.raises(FileNotFoundError):
        load_event_data(Path("/nonexistent/file.h5"))


def test_load_event_data_invalid_hdf5_structure():
    """Test error handling for HDF5 without required fields"""
    from neunorm.loaders.event_loader import load_event_data

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Create HDF5 missing 'x' field
        with h5py.File(temp_path, "w") as hf:
            hf.create_dataset("tof", data=np.array([1, 2, 3]))
            hf.create_dataset("y", data=np.array([0, 0, 0]))
            # Missing 'x' field!

        with pytest.raises(KeyError, match="'x'"):
            load_event_data(temp_path)

    finally:
        temp_path.unlink()


def test_load_event_data_with_subset():
    """Test loading subset of events (for memory efficiency)"""
    from neunorm.loaders.event_loader import load_event_data

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Create large dataset
        n_events = 1000
        with h5py.File(temp_path, "w") as hf:
            hf.create_dataset("tof", data=np.arange(n_events, dtype=np.int64))
            hf.create_dataset("x", data=np.zeros(n_events, dtype=np.int32))
            hf.create_dataset("y", data=np.zeros(n_events, dtype=np.int32))

        # Load only first 100 events
        events = load_event_data(temp_path, max_events=100)

        assert len(events) == 100
        assert events.total_events == 100  # Subset size, not file size
        # TOF multiplied by clock (default 25 ns): [0, 25, 50, ...]
        np.testing.assert_array_equal(events.tof, np.arange(100) * 25)

    finally:
        temp_path.unlink()


def test_event_data_model_validation():
    """Test EventData validation"""
    from neunorm.data_models.core import EventData

    # Valid data
    events = EventData(
        tof=np.array([1000, 2000], dtype=np.int64),
        x=np.array([100, 200], dtype=np.int32),
        y=np.array([300, 400], dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=2,
    )

    assert len(events) == 2
    events.validate_lengths()  # Should not raise


def test_event_data_model_length_mismatch():
    """Test EventData validation catches length mismatch"""
    from neunorm.data_models.core import EventData

    # Mismatched array lengths
    events = EventData(
        tof=np.array([1000, 2000, 3000], dtype=np.int64),  # 3 events
        x=np.array([100, 200], dtype=np.int32),  # 2 events - mismatch!
        y=np.array([300, 400], dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=2,
    )

    # Should raise when validating
    with pytest.raises(ValueError, match="Array length mismatch"):
        events.validate_lengths()


def test_event_data_model_total_events_mismatch():
    """Test validation catches total_events mismatch"""
    from neunorm.data_models.core import EventData

    events = EventData(
        tof=np.array([1000, 2000], dtype=np.int64),
        x=np.array([100, 200], dtype=np.int32),
        y=np.array([300, 400], dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=5,  # Wrong! Arrays have 2 events
    )

    with pytest.raises(ValueError, match="total_events.*doesn't match array length"):
        events.validate_lengths()
