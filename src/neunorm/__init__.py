"""
NeuNorm 2.0: Neutron Imaging Normalization and TOF Data Processing

A modern Python library for neutron imaging data processing at ORNL facilities
(MARS at HFIR and VENUS at SNS).

Features:
- Time-of-flight (TOF) event-mode data processing and pulse reconstruction
- Energy/wavelength-resolved histogramming and TOF rebinning
- Resonance and Bragg-edge analysis
- Flat-field (open-beam) normalization with automatic uncertainty propagation
- Dark/gamma correction, ROI clipping, and run combination
- End-to-end MARS/VENUS detector pipelines writing HDF5 (primary) and TIFF

Examples
--------
>>> from neunorm import __version__
>>> isinstance(__version__, str)
True
"""

try:
    from neunorm._version import __version__
except ImportError:
    __version__ = "unknown"

from neunorm.data_models.roi import ROI

__author__ = "Jean Bilheux, Chen Zhang"
__email__ = "bilheuxjm@ornl.gov, zhangc@ornl.gov"

__all__ = ["ROI", "__version__"]
