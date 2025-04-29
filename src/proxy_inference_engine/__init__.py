import os
from importlib import import_module

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from proxy_inference_engine.engine import InferenceEngine

pie_core = import_module("proxy_inference_engine.pie_core")

__all__ = ["InferenceEngine", "pie_core"]
