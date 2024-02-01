"""Neutron Imaging Normalization package"""
try:
    from ._version import __version__  # noqa: F401
except ImportError:
    __version__ = "unknown"


class DataType:
    sample = "sample"
    ob = "ob"
    df = "df"
    normalized = "normalized"
