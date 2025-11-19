"""
Core data models for NeuNorm 2.0.

Pydantic models for event data, histogram data, and processing results.
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class EventData(BaseModel):
    """
    Event-mode neutron data container.

    Represents raw list-mode events from TPX3/TPX4 detectors with
    pixel coordinates and time-of-flight values.

    Parameters
    ----------
    tof : np.ndarray
        Time-of-flight values in nanoseconds (1D array)
    x : np.ndarray
        X pixel coordinates (1D array, same length as tof)
    y : np.ndarray
        Y pixel coordinates (1D array, same length as tof)
    file_path : Path
        Source file path
    total_events : int
        Total number of events
    tof_clock : float
        TOF clock period in nanoseconds (default: 25.0 for TPX3)

    Examples
    --------
    >>> import numpy as np
    >>> from pathlib import Path
    >>> events = EventData(
    ...     tof=np.array([1000, 2000, 3000], dtype=np.int64),
    ...     x=np.array([100, 200, 150], dtype=np.int32),
    ...     y=np.array([250, 300, 275], dtype=np.int32),
    ...     file_path=Path('data.h5'),
    ...     total_events=3,
    ...     tof_clock=25.0
    ... )
    """

    tof: np.ndarray = Field(description="TOF values in nanoseconds")
    x: np.ndarray = Field(description="X pixel coordinates")
    y: np.ndarray = Field(description="Y pixel coordinates")
    chip_id: np.ndarray | None = Field(default=None, description="Chip ID for each event (0-3 for quad detector)")
    pulse_id: np.ndarray | None = Field(default=None, description="Reconstructed pulse ID for each event")
    file_path: Path = Field(description="Source HDF5 file path")
    total_events: int = Field(ge=0, description="Total number of events")
    tof_clock: float = Field(default=25.0, gt=0, description="TOF clock period (ns)")

    model_config = {"arbitrary_types_allowed": True}  # For numpy arrays (Pydantic v2)

    @field_validator("tof", "x", "y")
    @classmethod
    def validate_array_1d(cls, v):
        """Ensure arrays are 1D"""
        if v.ndim != 1:
            raise ValueError(f"Arrays must be 1D, got shape {v.shape}")
        return v

    @field_validator("x", "y")
    @classmethod
    def validate_coordinate_dtype(cls, v):
        """Ensure pixel coordinates are integer type"""
        if not np.issubdtype(v.dtype, np.integer):
            raise ValueError(f"Pixel coordinates must be integer type, got {v.dtype}")
        return v

    @field_validator("tof")
    @classmethod
    def validate_tof_dtype(cls, v):
        """Ensure TOF values are numeric"""
        if not np.issubdtype(v.dtype, np.number):
            raise ValueError(f"TOF values must be numeric, got {v.dtype}")
        return v

    @field_validator("chip_id", "pulse_id")
    @classmethod
    def validate_optional_array_1d(cls, v):
        """Ensure optional arrays are 1D if present"""
        if v is not None and v.ndim != 1:
            raise ValueError(f"Arrays must be 1D, got shape {v.shape}")
        return v

    @field_validator("chip_id")
    @classmethod
    def validate_chip_id_dtype(cls, v):
        """Ensure chip IDs are integer type if present"""
        if v is not None and not np.issubdtype(v.dtype, np.integer):
            raise ValueError(f"Chip IDs must be integer type, got {v.dtype}")
        return v

    @field_validator("pulse_id")
    @classmethod
    def validate_pulse_id_dtype(cls, v):
        """Ensure pulse IDs are integer type if present"""
        if v is not None and not np.issubdtype(v.dtype, np.integer):
            raise ValueError(f"Pulse IDs must be integer type, got {v.dtype}")
        return v

    @field_validator("file_path")
    @classmethod
    def validate_file_exists(cls, v):
        """Validate file path (can be non-existent for simulated data)"""
        # Convert to Path if string
        if isinstance(v, str):
            v = Path(v)
        return v

    @model_validator(mode="after")
    def validate_lengths(self):
        """Validate all arrays have same length (runs automatically)"""
        n_tof = len(self.tof)
        n_x = len(self.x)
        n_y = len(self.y)

        if not (n_tof == n_x == n_y):
            raise ValueError(
                f"Array length mismatch: tof={n_tof}, x={n_x}, y={n_y}. All event arrays must have the same length."
            )

        # Check optional arrays if present
        if self.chip_id is not None and len(self.chip_id) != n_tof:
            raise ValueError(f"chip_id length ({len(self.chip_id)}) doesn't match tof length ({n_tof})")

        if self.pulse_id is not None and len(self.pulse_id) != n_tof:
            raise ValueError(f"pulse_id length ({len(self.pulse_id)}) doesn't match tof length ({n_tof})")

        if n_tof != self.total_events:
            raise ValueError(f"total_events ({self.total_events}) doesn't match array length ({n_tof})")

        return self

    def __len__(self):
        """Return number of events in this dataset"""
        return len(self.tof)

    def __repr__(self):
        return (
            f"EventData(\n"
            f"  file: {self.file_path.name}\n"
            f"  events: {self.total_events:,}\n"
            f"  tof_clock: {self.tof_clock} ns\n"
            f"  x_range: [{self.x.min()}, {self.x.max()}]\n"
            f"  y_range: [{self.y.min()}, {self.y.max()}]\n"
            f"  tof_range: [{self.tof.min()}, {self.tof.max()}] ns\n"
            f")"
        )
