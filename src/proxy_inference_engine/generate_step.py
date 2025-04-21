import functools
import logging
from collections.abc import Iterator
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from src.proxy_inference_engine.cache import BaseCache
from src.proxy_inference_engine.cache.reusable import ReusableKVCache

logger = logging.getLogger(__name__)

def generate_step(
    prompt_ids: mx.array,
    model: nn.Module,
    *,
    pixel_values: mx.array | None = None,
    mask: mx.array | None = None,
    max_tokens: int = 256,
    sampler: Callable[[mx.array], mx.array] | None = None,
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
    max_kv_size: int | None = None,
    prompt_cache: list[BaseCache] | None = None,
    prefill_step_size: int = 512,
    reuse_prompt_cache: bool = False,
    computed_ids: mx.array | None = None,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> Iterator[tuple[mx.array, mx.array]]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens. Use``-1`` for an infinite
          generator. Default: ``256``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        prompt_cache (List[Any], optional): A pre-computed prompt cache. Note, if
          provided, the cache will be updated in place.
        prefill_step_size (int): Step size for processing the prompt.
        kv_bits (int, optional): Number of bits to use for KV cache quantization.
          None implies no cache quantization. Default: ``None``.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin using a quantized KV cache.
           when ``kv_bits`` is non-None. Default: ``0``.
        prompt_prorgress_callback (Callable[int, int]): A call-back which takes the
           prompt tokens processed so far and the total number of prompt tokens.

    Yields:
        tuple[mx.array, mx.array]: One token and a vector of log probabilities.
    """
    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = BaseCache.make_kv_cache(
            model,
            max_kv_size=max_kv_size,
            reusable=reuse_prompt_cache,
        )
    elif model.layers is not None and len(prompt_cache) != len(model.layers):
        raise ValueError("Wrong number of layers in the prompt cache.")

    if computed_ids is not None:
        y = reuse_cache(prompt_ids, computed_ids, prompt_cache)

    tokens = None

    quantize_cache_fn = functools.partial(
        BaseCache.maybe_quantize,
        quantized_start=quantized_kv_start,
        group_size=kv_group_size,
        bits=kv_bits,
    )

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    def _step(y: mx.array) -> tuple[mx.array, mx.array]:
        logits = model(y[None], pixel_values=pixel_values, mask=mask, cache=prompt_cache)
        logits = logits[:, -1, :]

        if logits_processors:
            nonlocal tokens
            tokens = y if tokens is None else mx.concat([tokens, y])

            for processor in logits_processors:
                logits = processor(tokens, logits)

        quantize_cache_fn(prompt_cache)

        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs.squeeze(0)

    while y.size > prefill_step_size:
        model(y[:prefill_step_size][None], pixel_values=pixel_values, mask=mask, cache=prompt_cache)
        quantize_cache_fn(prompt_cache)
        mx.eval([c.state for c in prompt_cache])
        y = y[prefill_step_size:]
        mx.clear_cache()

    y, logprobs = _step(y)
    mx.async_eval(y, logprobs)
    n = 0
    while True:
        if n == max_tokens:
            break
        if n == 0:
            mx.eval(y)
        else:
            y, logprobs = _step(y)
            mx.async_eval(y, logprobs)

        yield y, logprobs
        if n % 256 == 0:
            mx.clear_cache()
        n += 1


def reuse_cache(
    prompt_ids: mx.array,
    computed_ids: mx.array,
    cache: list[BaseCache],
) -> mx.array:
    """
    Reuse the cache for the given prompt and precomputed ids.
    """

    if not cache or not all(isinstance(c, ReusableKVCache) for c in cache):
        return prompt_ids

    common_prefix = 0
    for i, id in enumerate(computed_ids):
        if i >= len(prompt_ids) - 1 or prompt_ids[i] != id:
            break
        common_prefix += 1

    if common_prefix == 0:
        return prompt_ids

    for layer_cache in cache:
        assert isinstance(layer_cache, ReusableKVCache)
        layer_cache.reuse(len(prompt_ids), common_prefix)

    return prompt_ids[common_prefix:]
