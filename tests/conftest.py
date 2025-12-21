"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture(autouse=True)
def reset_settings():
    """Reset settings singleton between tests."""
    from crypto.config.settings import reset_settings
    reset_settings()
    yield
    reset_settings()
