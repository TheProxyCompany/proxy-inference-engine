from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from proxy_inference_engine.interaction import Interaction
from proxy_inference_engine.tokenizer.control_tokens import (
    ControlTokens,
    get_control_tokens,
)

logger = logging.getLogger(__name__)


class Tokenizer:
    """A convienience wrapper around a Hugging Face tokenizer.

    The wrapper provides convienient access to control tokens,
    encoding/decoding with templates, and vocabulary management.
    """

    def __init__(
        self,
        hf_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        tokenizer_config: dict[str, Any],
    ) -> None:
        """
        Args:
            tokenizer: The base Hugging Face tokenizer to wrap
            tokenizer_config: The tokenizer config
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self._tokenizer = hf_tokenizer
        self._tokenizer_config = tokenizer_config
        self._control_tokens = get_control_tokens(tokenizer_config)
        self.load_chat_template()

    def load_chat_template(
        self,
        file_name: str | None = None,
        template_string: str | None = None,
    ) -> None:
        """Load a chat template from a file.

        Args:
            file_name: The name of the file to load the chat template from. No extension is needed.
            template_string: A string to use as the chat template.
        """
        if file_name and template_string:
            raise ValueError("Cannot provide both file_name and template_string")

        if template_string:
            self._tokenizer.chat_template = template_string
        else:
            name = file_name or "chat_template.jinja"
            name = f"{name}.jinja" if not name.endswith(".jinja") else name

            current_dir = os.path.dirname(os.path.abspath(__file__))
            template_path = os.path.join(current_dir, name)

            # Fall back to default template if specified one doesn't exist
            if not os.path.exists(template_path):
                template_path = os.path.join(current_dir, "chat_template.jinja")

            with open(template_path) as f:
                self._tokenizer.chat_template = f.read()

    @property
    def control_tokens(self) -> ControlTokens:
        """
        Get the control tokens, or raise an error if they are not set.

        Control tokens such as end-of-sequence or tool-use tokens are used to control the model's behavior.
        """
        if self._control_tokens is None:
            raise ValueError("Control tokens are not set")
        return self._control_tokens

    @property
    def whitelist_control_tokens(self) -> list[str]:
        """
        Get the whitelist control tokens.
        """
        return self.control_tokens.get_whitelist_control_tokens()

    @property
    def stop_tokens(self) -> set[int]:
        """Get the set of token IDs that indicate stopping generation.

        Returns:
            Set of token IDs for EOS and EOM tokens from control_tokens.
            Returns empty set if no control tokens configured.
        """
        if not self._control_tokens:
            return set()

        # Get all end token IDs without special tokens to avoid duplicates
        stop_tokens = set()
        for stop_token in self._control_tokens.end_tokens():
            stop_tokens.add(
                self._tokenizer.encode(stop_token, add_special_tokens=False)[0]
            )

        # Flatten and deduplicate token IDs into a set
        return stop_tokens

    def decode(self, tokens: mx.array | list[int], **kwargs) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: List of token IDs to decode

        Returns:
            Decoded text string
        """
        if isinstance(tokens, mx.array):
            return self._tokenizer.decode(tokens.tolist(), **kwargs)
        else:
            return self._tokenizer.decode(tokens, **kwargs)

    def encode(
        self,
        prompt: str | list[dict[str, Any]] | list[Interaction],
        **kwargs,
    ) -> mx.array:

        if isinstance(prompt, str):
            return mx.array(self._tokenizer.encode(prompt, **kwargs))

        if isinstance(prompt, list):
            prompt = [
                event.to_dict() for event in prompt if isinstance(event, Interaction)
            ]
            kwargs["interactions"] = prompt

        encoded_prompt = self._tokenizer.apply_chat_template(prompt,**kwargs)

        if isinstance(encoded_prompt, str):
            encoded_prompt = self._tokenizer.encode(encoded_prompt, **kwargs)
        elif isinstance(encoded_prompt, list) and any(
            isinstance(item, str) for item in encoded_prompt
        ):
            encoded_prompt = [
                self._tokenizer.encode(item, **kwargs)
                for item in encoded_prompt
                if isinstance(item, str)
            ]

        return mx.array(encoded_prompt)
