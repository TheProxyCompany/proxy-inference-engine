"""
Proxy Inference Engine (PIE)
============================

Optimized MLX inference engine for Apple Silicon, powering reliable AI within The Proxy Company ecosystem.
"""

__version__ = "0.1.0"

# Import key components for easier access
from proxy_inference_engine.cache import (
    BaseCache,
    KVCache,
    QuantizedKVCache,
    ReusableKVCache,
    RotatingKVCache,
)
from proxy_inference_engine.generate_step import generate_step
from proxy_inference_engine.logits_processors import repetition_penalty_logits_processor
from proxy_inference_engine.samplers import (
    categorical_sampling,
    min_p_sampling,
    top_k_sampling,
    top_p_sampling,
)

# Aliases for backward compatibility
repetition_penalty = repetition_penalty_logits_processor
categorical_sampler = categorical_sampling
min_p_sampler = min_p_sampling
top_k_sampler = top_k_sampling
top_p_sampler = top_p_sampling