from typing import Any

import mlx.core as mx
from mlx.utils import tree_map
from pydantic import BaseModel, ConfigDict

from proxy_inference_engine.cache.kv_cache import QuantizedKVCache


class BaseModelArgs(BaseModel):
    """
    Base configuration class for models, leveraging Pydantic for robust
    type validation and settings management.
    """

    model_config = ConfigDict(extra="ignore")

def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: int | None = None,
    lengths: mx.array | None = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds < rinds
    if window_size is not None:
        mask = mask | (linds > rinds + window_size)
    if lengths is not None:
        lengths = lengths[:, None, None, None]
        mask = mask | (rinds >= lengths)
    return mask * -1e9


def create_attention_mask(h: mx.array, cache: Any | None = None):
    T = h.shape[1]
    if T > 1:
        window_size = None
        offset = 0
        if cache is not None and len(cache) > 0:
            c = cache[0]
            if hasattr(c, "max_size"):
                offset = min(c.max_size, c.offset)
                window_size = c.max_size
            else:
                offset = c.offset
        mask = create_causal_mask(T, offset, window_size=window_size)
        mask = mask.astype(h.dtype)
    else:
        mask = None
    return mask


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: mx.array | None,
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        scores += mask
    scores = mx.softmax(scores, axis=-1)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: mx.array | None,
) -> mx.array:
    if isinstance(cache, QuantizedKVCache):
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
