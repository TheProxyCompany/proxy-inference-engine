import secrets
import time

from pydantic import BaseModel, Field, ValidationInfo, field_validator

# --- Constants ---
COMPLETION_ID_PREFIX = "cmpl-"
TEXT_COMPLETION_OBJECT = "text_completion"

def generate_completion_id(prefix: str = COMPLETION_ID_PREFIX) -> str:
    """Generates a unique identifier for a completion response."""
    random_part = secrets.token_urlsafe(22)
    return f"{prefix}{random_part}"


def get_current_timestamp() -> int:
    """Returns the current time as a Unix epoch timestamp (seconds)."""
    return int(time.time())

class CompletionRequest(BaseModel):
    """
    Defines the request schema for the legacy text completion endpoint (compatible with OpenAI's API).
    """

    model: str = Field(
        description="The identifier of the model designated for completion generation."
    )
    prompt: str | list[str] = Field(
        default="",
        description="The input prompt(s) for which completions are generated. Accepts a single string or a list for batch processing.",
    )
    max_completion_tokens: int = Field(
        default=256,
        ge=1,
        description="The upper limit on the number of tokens to generate per completion.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Controls randomness via sampling temperature. Values closer to 0.0 yield more deterministic outputs, while higher values increase randomness.",
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Implements nucleus sampling by considering only tokens whose cumulative probability mass exceeds this threshold.",
    )
    min_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold for token consideration during sampling.",
    )
    n: int = Field(
        default=1,
        ge=1,
        description="Specifies the quantity of independent completions to generate for each provided prompt.",
    )
    stream: bool = Field(
        default=False,
        description="Indicates whether to stream intermediate results back as server-sent events.",
    )
    logprobs: int | None = Field(
        default=None,
        ge=0,
        le=5,
        description="If specified, includes the log probabilities for the top `logprobs` most likely tokens at each generation step.",
    )
    best_of: int | None = Field(
        default=None,
        ge=1,
        description="Generates `best_of` completions server-side and returns the one with the highest log probability per token. Requires `best_of > n`.",
    )
    top_k: int = Field(
        default=0,
        ge=0,
        description="Restricts sampling to the `k` most probable tokens. A value of 0 disables top-k filtering.",
    )

    @field_validator("best_of")
    @classmethod
    def check_best_of_greater_than_n(
        cls, v: int | None, info: ValidationInfo
    ) -> int | None:
        """Validates that `best_of` is greater than or equal to `n` if both are provided."""
        # Ensure info.data is accessed safely
        if (
            v is not None
            and "n" in info.data
            and isinstance(info.data.get("n"), int)
            and info.data["n"] > v
        ):
            raise ValueError(
                f"`best_of` ({v}) must be greater than or equal to `n` ({info.data['n']})"
            )
        return v


# --- Response Models ---


class CompletionChoice(BaseModel):
    """Represents a single generated completion choice."""

    index: int = Field(
        description="The sequential index of this choice within the response list."
    )
    text: str = Field(description="The generated completion text for this choice.")
    logprobs: list[float] | None = Field(
        default=None,
        description="Contains log probabilities if requested via the `logprobs` parameter. returns N highest log probabilities for each token.",
    )
    finish_reason: str | None = Field(
        description="The reason the model terminated generation (e.g., 'stop', 'length')."
    )


class CompletionUsage(BaseModel):
    """Provides token usage statistics for the completion request."""

    input_tokens: int = Field(
        description="The number of tokens constituting the input prompt(s)."
    )
    output_tokens: int = Field(
        description="The total number of tokens generated across all completion choices."
    )
    total_tokens: int = Field(
        description="The sum of `input_tokens` and `output_tokens`."
    )


class CompletionResponse(BaseModel):
    """
    Defines the response schema for the legacy text completion endpoint.
    """

    id: str = Field(
        default_factory=generate_completion_id,
        description="A unique identifier assigned to this completion response.",
        examples=["cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7"],
    )
    object: str = Field(
        default=TEXT_COMPLETION_OBJECT,
        description="The type of object returned, consistently 'text_completion'.",
    )
    created: int = Field(
        default_factory=get_current_timestamp,
        description="The Unix timestamp (seconds since epoch) indicating when the response was generated.",
    )
    model: str = Field(
        description="The identifier of the model that executed the completion request."
    )
    choices: list[CompletionChoice] = Field(
        description="A list containing the generated completion(s)."
    )
    usage: CompletionUsage = Field(
        description="An object detailing the token count statistics for the request."
    )
    system_fingerprint: str | None = Field(
        default=None,
        description="An opaque identifier representing the backend configuration that handled the request. May be used for reproducibility tracking.",
    )
