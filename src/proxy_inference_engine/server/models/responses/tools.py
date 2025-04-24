from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Function(BaseModel):
    """Defines a function for the response request."""

    name: str = Field(description="The name of the function to call.")
    type: Literal["function"] = "function"
    description: str = Field(
        description="A description of the function. Used by the model to determine whether or not to call the function."
    )
    strict: bool = Field(
        default=True,
        description="Whether to enforce strict parameter validation.",
    )
    parameters: dict = Field(
        description="A JSON schema object describing the parameters of the function."
    )

class ToolUseMode(Enum):
    """Controls which (if any) tool is called by the model."""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"

class FunctionID(BaseModel):
    """Defines a function tool for the response request."""

    type: Literal["function"] = "function"
    name: str = Field(description="The name of the function to call.")

class ToolChoice(BaseModel):
    """Defines the tool choice for the response request."""

    mode: ToolUseMode = Field(description="The mode for the tool choice.")
    function_tool: FunctionID | None = Field(description="The function to call.")
