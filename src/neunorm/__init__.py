"""
NeuNorm 2.0: Neutron Imaging Normalization and TOF Data Processing

A modern Python library for neutron imaging data processing at ORNL facilities
(MARS at HFIR and VENUS at SNS).

Features:
- Time-of-flight (TOF) event data processing
- Energy-resolved histogramming
- In-situ time-resolved workflows
- Resonance and Bragg edge detection
- Phase decomposition with NMF
- Traditional flat-field normalization (1.x compatibility)

Examples
--------
>>> from neunorm import __version__
>>> print(__version__)
2.0.0a0
"""

try:
    from neunorm._version import __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Jean Bilheux, Chen Zhang"
__email__ = "bilheuxjm@ornl.gov, zhangc@ornl.gov"

__all__ = ["__version__"]
