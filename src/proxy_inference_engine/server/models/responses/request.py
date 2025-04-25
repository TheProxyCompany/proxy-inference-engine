from pydantic import BaseModel, Field

from proxy_inference_engine.server.models.responses.format import ResponseFormat
from proxy_inference_engine.server.models.responses.tools import (
    Function,
    FunctionID,
    ToolUseMode,
)


class ResponseRequest(BaseModel):
    """Defines the request schema for the /v1/responses endpoint (MVP)."""

    model: str = Field(description="Model ID used to generate the response.")
    input: str = Field(description="Text input to the model.")
    stream: bool | None = Field(
        default=None,
        description="Whether to stream the response.",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether to allow the model to run tool calls in parallel.",
    )
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
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Controls the number of tokens considered at each step.",
    )
    min_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold for token consideration.",
    )
    tool_choice: ToolUseMode | FunctionID = Field(
        default=ToolUseMode.AUTO,
        description="How the model should select which tool (or tools) to use when generating a response.",
    )
    tools: list[Function] | None = Field(
        default=None,
        description="A list of tools that the model can use to generate a response.",
    )
    text: ResponseFormat | None = Field(
        default=None,
        description="The format of the response.",
    )
