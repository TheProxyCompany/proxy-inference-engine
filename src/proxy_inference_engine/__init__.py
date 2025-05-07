from importlib import import_module

from proxy_inference_engine.engine.client import InferenceEngineClient as client
from proxy_inference_engine.engine.utils import get_model_path

pie_core = import_module("proxy_inference_engine.pie_core")

__all__ = ["client", "get_model_path", "pie_core"]
