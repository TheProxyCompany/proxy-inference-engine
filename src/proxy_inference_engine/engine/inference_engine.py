import logging
from collections.abc import Callable, Iterator
from typing import Any

import mlx.core as mx
from pse.structuring_engine import StructuringEngine

from proxy_inference_engine.cache import PromptCache
from proxy_inference_engine.interaction.interaction import Interaction
from proxy_inference_engine.logits_processors import repetition_penalty_logits_processor
from proxy_inference_engine.models import load
from proxy_inference_engine.samplers import make_sampler
from proxy_inference_engine.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    A class for performing inference with a LLM.
    """

    def __init__(self, model_path: str):
        llm = load(model_path)
        self.model, self.tokenizer_config = llm.model, llm.tokenizer_config
        self.tokenizer = Tokenizer(llm.hf_tokenizer, self.tokenizer_config)
        self.prompt_cache = PromptCache()
        self.structuring_engine = StructuringEngine(llm.hf_tokenizer, multi_token_sampling=True)
        logger.info(f"Inference Engine initialized with model from {model_path}")

    async def __call__(
        self,
        prompt: list[Interaction],
        **inference_kwargs,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (list[Interaction]): The input prompt for completion.
            **inference_kwargs: Additional keyword arguments to use for inference.
        """
        breakpoint()
        tokenizer_config = {
            "prompt": prompt,
            **inference_kwargs,
            **self.tokenizer.control_tokens.model_dump(),
        }
        encoded_prompt = self.tokenizer.encode(**tokenizer_config)
        prompt_length = encoded_prompt.size

        self.prompt_cache.load_cached_prompt(encoded_prompt)
        logger.info(f"PROMPT:\n{self.tokenizer.decode(encoded_prompt)}")
        generated_ids_list, finish_reason = await self.generate(
            encoded_prompt, **inference_kwargs
        )
        generated_text = self.tokenizer.decode(generated_ids_list)
        return generated_text, {
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_length,
            "completion_tokens": len(generated_ids_list),
            "total_tokens": prompt_length + len(generated_ids_list),
        }

    async def generate(
        self,
        prompt_ids: mx.array,
        **inference_kwargs,
    ) -> tuple[mx.array, str]:
        """
        Generate a completion for the given prompt.

        Args:
            prompt_token_ids (mx.array): The input prompt for completion.
        """
        sampler = self.make_sampler(**inference_kwargs)
        logits_processors = self.make_logits_processors(**inference_kwargs)
        max_completion_tokens = int(inference_kwargs.get("max_completion_tokens", -1))

        result: list[int] = []
        stop_reason: str = "finish"

        for token_id, _ in self.generate_step(
            prompt_ids,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            tokens = token_id.tolist()
            assert isinstance(tokens, list)
            for token_id in tokens:
                if token_id in self.tokenizer.stop_tokens:
                    stop_reason = "stop"
                    break

                result.append(token_id)

            if self.structuring_engine.has_reached_accept_state:
                break

            if max_completion_tokens > 0 and len(result) >= max_completion_tokens:
                stop_reason = "length"
                break

        return mx.array(result), stop_reason

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

        Yields:
            tuples of (next_token_id, log_probabilities).
        """

        def _inference(
            current_input_ids: mx.array,
        ) -> tuple[mx.array, mx.array]:
            """Performs one forward pass, updates history, applies processors, and samples."""
            model_kwargs: dict[str, Any] = {"cache": self.prompt_cache.cache}

            # Only add optional parameters if they exist
            if pixel_values is not None:
                model_kwargs["pixel_values"] = pixel_values
            if mask is not None:
                model_kwargs["mask"] = mask

            # Call model with appropriate arguments
            logits = self.model(current_input_ids[None], **model_kwargs)
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

        if len(self.prompt_cache.cache) == 0:
            self.prompt_cache.create_kv_cache(self.model)

        tokens_to_process = self.prompt_cache(prompt_ids)
        next_token_id, current_logprobs = _inference(tokens_to_process)
        mx.async_eval(next_token_id, current_logprobs)

        step_count = 0
        while True:
            if step_count == 0:
                # Synchronize computation for the first token
                mx.eval(next_token_id)
            else:
                # Perform the next inference step
                next_token_id, current_logprobs = _inference(next_token_id)
                mx.async_eval(next_token_id, current_logprobs)

            # Yield the token and its log probabilities.
            yield next_token_id, current_logprobs

            step_count += 1
            # Periodically clear the MLX computation graph cache to prevent excessive memory growth.
            if step_count % 256 == 0:
                mx.clear_cache()

    def make_sampler(self, **kwargs) -> Callable[[mx.array], mx.array]:
        """
        Return a sampler function.
        If structured is True, use the structured sampler.
        Otherwise, use the simple sampler.
        """
        temp = float(kwargs.get("temp", 1.0))
        min_p = float(kwargs.get("min_p", 0.02))
        min_tokens_to_keep = int(kwargs.get("min_tokens_to_keep", 1))
        top_p = float(kwargs.get("top_p", 1.0))
        top_k = int(kwargs.get("top_k", -1))
        sampler = make_sampler(
            temp=temp,
            min_p=min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_p=top_p,
            top_k=top_k,
        )
        if kwargs.get("structured", False) or kwargs.get("json_schema", None):
            return lambda x: self.structuring_engine.sample(x, sampler)
        else:
            return sampler

    def make_logits_processors(
        self, **kwargs
    ) -> list[Callable[[mx.array, mx.array], mx.array]]:
        """
        Return a list of logits processor functions.
        """
        logits_processors = []
        if kwargs.get("structured", False) or kwargs.get("json_schema", None):
            logits_processors.append(self.structuring_engine.process_logits)

        if kwargs.get("repetition_penalty", 1.0) != 1.0:
            repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
            context_size = int(kwargs.get("context_size", 60))
            logits_processors.append(
                repetition_penalty_logits_processor(repetition_penalty, context_size)
            )

        return logits_processors
