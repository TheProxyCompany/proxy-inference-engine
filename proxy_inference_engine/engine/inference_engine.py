from collections.abc import Callable, Iterator
from pse.structuring_engine import StructuringEngine

from proxy_inference_engine.tokenizer import Tokenizer
from proxy_inference_engine.cache import PromptCache
from proxy_inference_engine.models import load

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, model_path: str):
        self.model, hf_tokenizer = load(model_path)
        self.tokenizer = Tokenizer(hf_tokenizer)
        self.prompt_cache = PromptCache()
        self.structuring_engine = StructuringEngine(
            hf_tokenizer, multi_token_sampling=True
        )

    def run_inference(
        self,
        prompt: str | list[dict[str, Any]],
        processed_token_ids: mx.array | None = None,
        **inference_kwargs,
    ):
        """
        Generate a completion for the given prompt.

        Args:
            prompt (str | list[dict[str, Any]]): The input prompt for completion.
            **inference_kwargs: Additional keyword arguments to use for inference.
        """
        tokenizer_config = {
            "prompt": prompt,
            **inference_kwargs,
            **self.tokenizer.control_tokens.model_dump(),
        }
        encoded_prompt = self.tokenizer.encode(**tokenizer_config)

        # Try to load from cache first if caching is enabled
        cache_system_prompt = inference_kwargs.get("cache_system_prompt", True)
        reuse_prompt_cache = inference_kwargs.get("reuse_prompt_cache", True)

        if cache_system_prompt and reuse_prompt_cache and not processed_token_ids:
            self.prompt_cache.load_cached_prompt(encoded_prompt)

        logger.info(f"PROMPT:\n{self.tokenizer.decode(encoded_prompt)}")

    def generate_step(
        self,
        prompt_ids: mx.array,
        pixel_values: mx.array | None = None,
        mask: mx.array | None = None,
        sampler: Callable[[mx.array], mx.array] = (lambda x: mx.argmax(x, axis=-1)),
        logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
    ) -> Iterator[tuple[mx.array, mx.array]]:
        """
        Generates tokens autoregressively, yielding one token and its log probabilities per step.

        Args:
            prompt_ids: The initial sequence of token IDs to start generation from.
            pixel_values: Optional pixel values for multi-modal models.
            mask: Optional attention mask.
            sampler: A function that takes log probabilities and returns a sampled token ID.
                     Defaults to greedy sampling (argmax).
            logits_processors: An optional list of functions to modify the logits before sampling.
                               Each processor takes the current sequence of generated IDs and the logits.

        Yields:
            Iterator[tuple[mx.array, mx.array]]: An iterator yielding tuples of
            (next_token_id, log_probabilities_for_that_step).
        """

        CACHE_CLEAR_INTERVAL = 256  # Interval for clearing MLX's computation cache

        def _perform_inference_step(
            current_input_ids: mx.array,
        ) -> tuple[mx.array, mx.array]:
            """Performs one forward pass, updates history, applies processors, and samples."""
            # Perform the forward pass through the model
            logits = self.model(
                current_input_ids[None],  # Add batch dimension for the model
                pixel_values=pixel_values,
                mask=mask,
                cache=self.prompt_cache.cache,  # Use the KV cache
            )
            # Extract logits for the most recent token
            last_token_logits = logits[:, -1, :]
            self.prompt_cache.update(current_input_ids)

            processed_logits = last_token_logits

            # Apply any configured logits processors sequentially
            current_token_history = self.prompt_cache.computed_ids
            for processor in logits_processors or []:
                processed_logits = processor(current_token_history, processed_logits)
            # Calculate log probabilities (log-softmax normalization)
            logprobs = processed_logits - mx.logsumexp(
                processed_logits, axis=-1, keepdims=True
            )
            # Sample the next token ID using the provided sampler function
            next_token_id = sampler(logprobs)
            # Return the sampled next token ID and the log probability distribution
            return next_token_id, logprobs.squeeze(0)

        # Get the tokens that need to be processed
        tokens_to_process = self.prompt_cache(prompt_ids)
        # Perform the first inference step
        next_token_id, current_logprobs = _perform_inference_step(tokens_to_process)
        mx.async_eval(next_token_id, current_logprobs)

        step_count = 0
        while True:
            if step_count == 0:
                # Synchronize computation for the first token
                mx.eval(next_token_id)
            else:
                # Perform the next inference step
                next_token_id, current_logprobs = _perform_inference_step(next_token_id)
                mx.async_eval(next_token_id, current_logprobs)

            # Yield the token and its log probabilities.
            yield next_token_id, current_logprobs

            step_count += 1
            # Periodically clear the MLX computation graph cache to prevent excessive memory growth.
            if step_count % CACHE_CLEAR_INTERVAL == 0:
                mx.clear_cache()
