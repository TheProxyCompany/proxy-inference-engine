import time
import random
import string
from typing import Any
from pydantic import BaseModel, Field

# Based on OpenAI API v1 /v1/chat/completions structure


class CompletionRequest(BaseModel):
    model: str  # Model identifier (though we'll use a fixed one for MVP)
    prompt: str | list[str]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1  # How many completions to generate for each prompt. MVP ignores > 1.
    stream: bool = False  # MVP does not support streaming
    logprobs: int | None = None  # MVP does not support logprobs
    echo: bool = False  # MVP does not support echo
    stop: str | list[str] | None = None  # MVP basic stop support
    presence_penalty: float = 0.0  # Ignored in MVP
    frequency_penalty: float = 0.0  # Ignored in MVP
    best_of: int | None = None  # Ignored in MVP
    logit_bias: dict[str, float] | None = None  # Ignored in MVP
    user: str | None = None  # Ignored in MVP

    # Note: Many parameters are included for API compatibility but ignored by the MVP logic.


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: Any | None = None
    finish_reason: str | None = "length"


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def generate_completion_id() -> str:
    """Generates a unique completion ID."""
    return f"cmpl-{''.join(random.choices(string.ascii_letters + string.digits, k=29))}"


def get_current_timestamp() -> int:
    """Returns the current Unix timestamp."""
    return int(time.time())


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=generate_completion_id)
    object: str = "text_completion"
    created: int = Field(default_factory=get_current_timestamp)
    model: str  # The model name used for the completion.
    choices: list[CompletionChoice]
    usage: CompletionUsage
    system_fingerprint: str | None = None
