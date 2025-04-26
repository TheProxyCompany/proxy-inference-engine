import logging
from collections.abc import Callable, Iterator
from typing import Any

import mlx.core as mx
from pse.structuring_engine import StructuringEngine

from proxy_inference_engine.cache import PromptCache
from proxy_inference_engine.interaction import InteractionRole
from proxy_inference_engine.interaction.content import Content
from proxy_inference_engine.interaction.interaction import Interaction
from proxy_inference_engine.logits_processors import repetition_penalty_logits_processor
from proxy_inference_engine.models import load
from proxy_inference_engine.samplers import make_sampler
from proxy_inference_engine.state_machine import RootStateMachine
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
        self.root_state_machine = RootStateMachine(self.tokenizer.control_tokens)
        self.structuring_engine = StructuringEngine(
            llm.hf_tokenizer,
            whitelist_control_tokens=self.tokenizer.whitelist_control_tokens,
            multi_token_sampling=True,
        )

        self.samplers: dict[str, Callable[[mx.array], mx.array]] = {}
        self.logits_processors: dict[str, list[Callable[[mx.array, mx.array], mx.array]]] = {}
        logger.info(f"Inference Engine initialized with model from {model_path}")

    async def __call__(self, prompt: list[Interaction], **inference_kwargs) -> Interaction:
        """
        Generate a completion for the given prompt.

        Args:
            prompt (list[Interaction]): The input prompt for completion.
            **inference_kwargs: Additional keyword arguments to use for inference.
        """
        tokenizer_config = {
            "prompt": prompt,
            **inference_kwargs,
            **self.tokenizer.control_tokens.model_dump(),
        }
        encoded_prompt = self.tokenizer.encode(**tokenizer_config)
        prompt_length = encoded_prompt.size

        self.prompt_cache.load_cached_prompt(encoded_prompt)
        logger.info(f"PROMPT:\n{self.tokenizer.decode(encoded_prompt)}")

        state_machine_kwargs = {
            "response_format": inference_kwargs.get("response_format"),
            "tools": inference_kwargs.get("tools"),
            "parallel_tool_calls": inference_kwargs.get("parallel_tool_calls"),
            "tool_choice": inference_kwargs.get("tool_choice"),
        }
        self.root_state_machine.configure(**state_machine_kwargs)
        self.structuring_engine.reset()
        self.structuring_engine.configure(self.root_state_machine)

        for state_id, state in self.root_state_machine.available_states.items():
            state_kwargs = state.generation_kwargs or inference_kwargs
            self.samplers[state_id] = self.make_sampler(**state_kwargs)
            self.logits_processors[state_id] = self.make_logits_processors(
                **state_kwargs
            )

        self.samplers["root"] = self.make_sampler(**inference_kwargs)
        self.logits_processors["root"] = self.make_logits_processors(**inference_kwargs)
        logger.info(
            f"Loaded {len(self.samplers)} samplers and {len(self.logits_processors)} logits processors"
        )

        generated_ids, finish_reason = await self.generate(encoded_prompt, **inference_kwargs)
        logger.info(f"\nGENERATED:\n{self.tokenizer.decode(generated_ids)}\n\n")

        metadata = {
            "finish_reason": finish_reason,
            "prompt_tokens": prompt_length,
            "completion_tokens": len(generated_ids),
            "total_tokens": prompt_length + len(generated_ids),
        }

        content = []

        for state_id, output in self.structuring_engine.get_stateful_structured_output():
            if (engine_state := self.root_state_machine.available_states.get(state_id)) is None:
                logger.warning(f"Unknown state: {state_id}")
                continue

            logger.info(f"STATE: {engine_state.identifier}")
            logger.info(f"OUTPUT: {output}")

            match engine_state.identifier:
                case "structured_output" | "text_output":
                    content.append(Content.text(output))
                    if engine_state.identifier == "structured_output":
                        metadata["finish_reason"] = "stop"
                case "tool_call":
                    if not isinstance(output, dict):
                        logger.warning(f"Tool call output is not a dictionary: {output}")
                        continue
                    if "name" not in output or "arguments" not in output:
                        logger.warning(f"Tool call output is missing name or arguments: {output}")
                        continue

                    content.append(Content.tool_call(output["name"], output["arguments"]))
                case _:
                    logger.warning(f"Unknown state: {engine_state.identifier}")

        return Interaction(
            role=InteractionRole.AGENT,
            content=content,
            **metadata,
        )

    async def generate(
        self,
        prompt_ids: mx.array,
        **inference_kwargs,
    ) -> tuple[list[int], str]:
        """
        Generate a completion for the given prompt.

        Args:
            prompt_token_ids (mx.array): The input prompt for completion.
        """
        max_completion_tokens = int(inference_kwargs.get("max_completion_tokens", -1))

        result: list[int] = []
        stop_reason: str = "finish"

        for token_id, _ in self.generate_step(prompt_ids):
            tokens = token_id.tolist()
            assert isinstance(tokens, list)
            for token_id in tokens:
                if token_id in self.tokenizer.stop_tokens:
                    stop_reason = "stop"
                    break

                result.append(token_id)

            if self.structuring_engine.has_reached_accept_state:
                stop_reason = "tool_calls"
                break

            if max_completion_tokens > 0 and len(result) >= max_completion_tokens:
                stop_reason = "length"
                break

        return result, stop_reason

    def generate_step(
        self,
        prompt_ids: mx.array,
        pixel_values: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> Iterator[tuple[mx.array, mx.array]]:
        """
        Generates tokens autoregressively, yielding one token and its log probabilities per step.

        Yields:
            tuples of (next_token_id, log_probabilities).
        """

        def _inference(current_input_ids: mx.array) -> tuple[mx.array, mx.array]:
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
            engine_state = self.structuring_engine.get_current_state() or "root"
            for processor in self.logits_processors[engine_state] or []:
                processed_logits = processor(current_token_history, processed_logits)

            # Calculate log probabilities (log-softmax normalization)
            logprobs = processed_logits - mx.logsumexp(
                processed_logits, axis=-1, keepdims=True
            )
            # Sample the next token ID using the provided sampler function
            next_token_id = self.samplers[engine_state](logprobs)
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
        temp = kwargs.get("temp", 1.0)
        top_p = kwargs.get("top_p", 1.0)
        top_k = kwargs.get("top_k", -1)
        min_p = kwargs.get("min_p", 0.0)
        min_tokens_to_keep = kwargs.get("min_tokens_to_keep", 1)
        sampler = make_sampler(
            temp=temp,
            min_p=min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_p=top_p,
            top_k=top_k,
        )
        return lambda x: self.structuring_engine.sample(x, sampler)

    def make_logits_processors(
        self, **kwargs
    ) -> list[Callable[[mx.array, mx.array], mx.array]]:
        """
        Return a list of logits processor functions.
        """
        logits_processors = []
        logits_processors.append(self.structuring_engine.process_logits)

        if kwargs.get("repetition_penalty", 1.0) != 1.0:
            repetition_penalty = float(kwargs.get("repetition_penalty", 1.0))
            context_size = int(kwargs.get("context_size", 60))
            logits_processors.append(
                repetition_penalty_logits_processor(repetition_penalty, context_size)
            )

        return logits_processors
