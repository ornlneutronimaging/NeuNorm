"""
Test that EventData validation runs automatically on instantiation.

Ensures invalid EventData objects cannot be created.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError


def test_eventdata_rejects_mismatched_arrays_on_creation():
    """
    Test that mismatched array lengths are caught during object creation.

    CURRENT BUG: Can create invalid EventData, only fails when validate_lengths() called
    DESIRED: Should fail immediately on __init__
    """
    from neunorm.data_models.core import EventData

    # This SHOULD raise ValidationError immediately
    with pytest.raises(ValidationError, match="Array length mismatch"):
        EventData(
            tof=np.array([1000, 2000, 3000], dtype=np.int64),  # 3 events
            x=np.array([100, 200], dtype=np.int32),  # 2 events - MISMATCH!
            y=np.array([300, 400], dtype=np.int32),  # 2 events
            file_path=Path("test.h5"),
            total_events=2,
        )


def test_eventdata_rejects_total_events_mismatch_on_creation():
    """
    Test that total_events mismatch is caught during creation.
    """
    from neunorm.data_models.core import EventData

    with pytest.raises(ValidationError, match="total_events.*doesn't match"):
        EventData(
            tof=np.array([1000, 2000], dtype=np.int64),  # 2 events
            x=np.array([100, 200], dtype=np.int32),
            y=np.array([300, 400], dtype=np.int32),
            file_path=Path("test.h5"),
            total_events=5,  # WRONG! Should be 2
        )


def test_eventdata_valid_object_created_without_manual_validation():
    """
    Test that valid EventData can be created and used without calling validate_lengths().
    """
    from neunorm.data_models.core import EventData

    # Valid data - should work
    events = EventData(
        tof=np.array([1000, 2000], dtype=np.int64),
        x=np.array([100, 200], dtype=np.int32),
        y=np.array([300, 400], dtype=np.int32),
        file_path=Path("test.h5"),
        total_events=2,
    )

    # Should be usable without calling validate_lengths()
    assert len(events) == 2
    assert events.total_events == 2
