import mlx.core as mx


def get_top_logprobs(logprobs: mx.array, top_k: int) -> dict[int, float]:
    """Extract top-k logprobs using MLX arrays.

    Optimized implementation for MLX arrays that handles both 1D and 2D inputs.

    Args:
        logits: MLX array of shape (vocab_size,) or (1, vocab_size)
        top_k: Number of top tokens to return

    Returns:
        Tuple of (indices, values) arrays containing the top-k results

    Raises:
        ImportError: If MLX is not installed
        TypeError: If input is not an MLX array
        ValueError: If input dimensions are invalid
    """
    if top_k == 0:
        return {}

    ndim = logprobs.ndim
    if ndim == 2:
        logprobs = mx.squeeze(logprobs, axis=0, stream=mx.cpu)
    elif ndim != 1:
        raise ValueError(f"Expected 1D or 2D array, got {ndim}D")

    vocab_size = logprobs.shape[0]
    top_k = min(top_k, vocab_size)

    if vocab_size == 0:
        return {}

    # Use argpartition for efficient selection
    top_k_indices = mx.argpartition(-logprobs, top_k - 1, stream=mx.cpu)[:top_k]
    top_k_values = logprobs[top_k_indices]

    # Sort for consistency
    sorted_order = mx.argsort(-top_k_values, stream=mx.cpu)
    result: dict[int, float] = {}
    for i in range(top_k):
        token_id = int(top_k_indices[sorted_order[i]])
        logprob = float(top_k_values[sorted_order[i]])
        result[token_id] = logprob

    return result
