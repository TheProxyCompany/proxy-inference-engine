"""
Basic tests for the Proxy Inference Engine.

This module contains fundamental tests to verify basic functionality.
"""

import pytest


def test_import():
    """Test that we can import the main package."""
    try:
        import proxy_inference_engine

        assert proxy_inference_engine.__file__ is not None
        assert proxy_inference_engine.pie_core.health_check()
    except ImportError:
        pytest.fail("Failed to import proxy_inference_engine")
