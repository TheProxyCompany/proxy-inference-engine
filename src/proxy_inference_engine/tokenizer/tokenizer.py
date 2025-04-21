from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import mlx.core as mx

from proxy_inference_engine.utils import load_template
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from src.proxy_inference_engine.tokenizer.control_tokens import ControlTokens, get_control_tokens

logger = logging.getLogger(__name__)


class Tokenizer:
    """A convienience wrapper around a Hugging Face tokenizer.

    The wrapper provides convienient access to control tokens,
    encoding/decoding with templates, and vocabulary management.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        control_tokens: ControlTokens | None = None,
    ) -> None:
        """
        Args:
            tokenizer: The base Hugging Face tokenizer to wrap
            control_tokens: Optional control tokens - such as end-of-sequence or tool-use tokens
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self._tokenizer = tokenizer
        self._control_tokens = control_tokens

    def load_chat_template(self, file_name: str) -> None:
        """Load a chat template from a file.

        Args:
            file_name: The name of the file to load the chat template from. No extension is needed.
        """
        self._tokenizer.chat_template = load_template(file_name)

    def set_chat_template(self, template: str) -> None:
        """Set the chat template for the tokenizer.

        Args:
            template: The chat template to set
        """
        self._tokenizer.chat_template = template

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

    def decode(self, tokens: mx.array, **kwargs) -> str:
        """Decode token IDs back to text.

        Args:
            tokens: List of token IDs to decode

        Returns:
            Decoded text string
        """
        return self._tokenizer.decode(tokens, **kwargs)

    def encode(
        self, prompt: str | list[dict[str, str]] | dict[str, Any], **kwargs
    ) -> mx.array:
        """Encode text or chat messages into tokens.

        Handles both raw text and chat message formats. For raw text, supports
        template substitution of tools and date strings.

        Args:
            prompt: Text string or list of chat messages to encode
            **kwargs: Additional encoding options

        Returns:
            Token IDs or templated string depending on input format

        Raises:
            ValueError: If chat template produces unsupported format
        """
        if isinstance(prompt, str):
            return mx.array(self._tokenizer.encode(prompt, **kwargs))

        if isinstance(prompt, dict):
            prompt = [event.to_dict() for event in prompt.values()]

        encoded_prompt = self._tokenizer.apply_chat_template(prompt, **kwargs)
        if isinstance(encoded_prompt, str):
            encoded_prompt = self._tokenizer.encode(encoded_prompt, **kwargs)
        elif isinstance(encoded_prompt, list) and any(isinstance(item, str) for item in encoded_prompt):
            encoded_prompt = [
                self._tokenizer.encode(item, **kwargs)
                for item in encoded_prompt
                if isinstance(item, str)
            ]

        return mx.array(encoded_prompt)

    @staticmethod
    def load(model_path: str | Path, **kwargs) -> Tokenizer:
        """Create a TokenizerWrapper by loading a Hugging Face tokenizer.

        Args:
            model_path: Path to the model/tokenizer
            **kwargs: Additional args passed to AutoTokenizer.from_pretrained()

        Returns:
            Configured TokenizerWrapper instance
        """
        # Convert string path to Path object for consistent handling
        model_path = Path(model_path) if isinstance(model_path, str) else model_path

        # Load tokenizer configuration and determine appropriate control tokens
        try:
            with open(model_path / "tokenizer_config.json") as f:
                tokenizer_config = json.load(f)
                control_tokens = get_control_tokens(str(model_path), tokenizer_config)
        except FileNotFoundError:
            logger.warning(
                f"Tokenizer config not found at {model_path}, using default control tokens"
            )
            control_tokens = get_control_tokens(str(model_path), {})

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, **kwargs
        )
        return Tokenizer(tokenizer, control_tokens)
