from collections.abc import Callable

import mlx.core as mx

from proxy_inference_engine.samplers.categorical import categorical_sampling
from proxy_inference_engine.samplers.min_p import min_p_sampling
from proxy_inference_engine.samplers.top_k import top_k_sampling
from proxy_inference_engine.samplers.top_p import top_p_sampling


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = -1,
) -> Callable[[mx.array], mx.array]:
    """
    Make a sampler function for use with ``generate_step``.

    Args:
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        top_k (int, optional): The top k tokens ranked by probability to constrain
          the sampling to.

    Returns:
        Callable[mx.array, mx.array]:
            A sampler which takes log-probabilities and returns tokens.
    """
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)
    elif top_p > 0 and top_p < 1.0:
        return lambda x: top_p_sampling(x, top_p, temp)
    elif min_p != 0.0:
        return lambda x: min_p_sampling(x, min_p, min_tokens_to_keep, temp)
    elif top_k > 0:
        return lambda x: top_k_sampling(x, top_k, temp)
    else:
        return lambda x: categorical_sampling(x, temp)
