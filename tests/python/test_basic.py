"""
Basic tests for the Proxy Inference Engine.

This module contains fundamental tests to verify basic functionality.
"""

import pytest


def test_import():
    """Test that we can import the main package."""
    try:
        import proxy_inference_engine.pie_core

        assert proxy_inference_engine.pie_core.READY_FOR_PYTHON is not None
    except ImportError:
        pytest.fail("Failed to import proxy_inference_engine")
