from pydantic import BaseModel, Field


class ResponseRequest(BaseModel):
    """Defines the request schema for the /v1/responses endpoint (MVP)."""

    model: str = Field(description="Model ID used to generate the response.")
    input: str = Field(description="Text input to the model.")
    instructions: str | None = Field(
        default=None,
        description="System/developer instructions for the model.",
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Upper bound for the number of tokens generated.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold.",
    )
