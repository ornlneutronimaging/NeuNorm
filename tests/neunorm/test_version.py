"""Test package version and basic imports"""
from neunorm import __version__


def test_version():
    """Verify version is available"""
    assert __version__ is not None
    assert isinstance(__version__, str)
    # Development version should contain 'dev', 'a' (alpha), or be 'unknown'
    assert any(x in __version__ for x in ["dev", "a", "unknown"]) or __version__.count(".") == 2
