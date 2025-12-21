"""Tests for the main module."""

from crypto import __version__


def test_version():
    """Test version is set correctly."""
    assert __version__ == "0.1.0"
