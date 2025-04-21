from collections.abc import Callable

import mlx.core as mx


def make_repetition_penalty(penalty: float = 1.0, context_size: int = 60) -> Callable[[mx.array, mx.array], mx.array]:

    if penalty < 0 or context_size < 0:
        raise ValueError(f"Parameters must be non-negative, got penalty={penalty} and context_size={context_size}")

    def repetition_penalty_processor(tokens: mx.array, logits: mx.array) -> mx.array:
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            selected_logits = logits[:, tokens]
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
                stream=mx.cpu
            )
            logits[:, tokens] = selected_logits
        return logits

    return repetition_penalty_processor
