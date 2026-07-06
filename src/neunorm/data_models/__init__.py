"""
Data models for NeuNorm 2.0.

Pydantic models for type-safe data handling throughout the processing pipeline.
"""

from neunorm.data_models.roi import ROI, as_roi_bounds
from neunorm.data_models.tof import BinningConfig

__all__ = ["ROI", "BinningConfig", "as_roi_bounds"]
