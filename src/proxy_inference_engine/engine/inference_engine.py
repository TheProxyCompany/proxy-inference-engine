from collections.abc import Callable, Iterator

from pse.structuring_engine import StructuringEngine

from proxy_inference_engine.interaction.interaction import Interaction
from proxy_inference_engine.tokenizer import Tokenizer
from proxy_inference_engine.cache import PromptCache
from proxy_inference_engine.models import load

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self, model_path: str):
        self.model, hf_tokenizer = load(model_path)
        self.tokenizer = Tokenizer(hf_tokenizer)
        self.prompt_cache = PromptCache()
        self.structuring_engine = StructuringEngine(
            hf_tokenizer, multi_token_sampling=True
        )

    def inference(
        self,
        prompt: list[Interaction],
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

        self.prompt_cache.load_cached_prompt(encoded_prompt)

        logger.info(f"PROMPT:\n{self.tokenizer.decode(encoded_prompt)}")

    def generate_step(
        self,
        prompt_ids: mx.array,
        pixel_values: mx.array | None = None,
        mask: mx.array | None = None,
        sampler: Callable[[mx.array], mx.array] = (lambda x: mx.argmax(x, axis=-1)),
        logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
        max_new_tokens: int = -1,
    ) -> Iterator[tuple[mx.array, mx.array]]:
        """
        Generates tokens autoregressively, yielding one token and its log probabilities per step.

        Yields:
            tuples of (next_token_id, log_probabilities).
        """

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
            return next_token_id, logprobs.squeeze(0)

        # Get the tokens that need to be processed
        tokens_to_process = self.prompt_cache(prompt_ids)
        # Perform the first inference step
        next_token_id, current_logprobs = _perform_inference_step(tokens_to_process)
        mx.async_eval(next_token_id, current_logprobs)

        step_count = 0
        while max_new_tokens == -1 or step_count < max_new_tokens:
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
            if step_count % 256 == 0:
                mx.clear_cache()
