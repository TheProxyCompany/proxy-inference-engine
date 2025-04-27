from __future__ import annotations

import json
import secrets
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from proxy_inference_engine.interaction import (
    Content,
    InteractionType,
)

TOOL_CALL_ID_PREFIX = "call-"


def generate_tool_call_id(prefix: str = TOOL_CALL_ID_PREFIX) -> str:
    """Generates a unique identifier string for a tool call."""
    random_part = secrets.token_urlsafe(22)
    return f"{prefix}{random_part}"


class ChatCompletionToolUsage(BaseModel):
    """Represents the usage of a tool in a chat completion."""

    class UsedFunction(BaseModel):
        """Represents a function that was used in a chat completion."""

        name: str = Field(description="The name of the function to call.")
        arguments: str = Field(
            description="The arguments to pass to the function. JSON encoded."
        )

    type: Literal["function"] = "function"
    id: str = Field(description="The unique identifier of the tool.")
    function: UsedFunction = Field(description="The function that was used.")

    @staticmethod
    def from_content(
        content: Content, tool_call_id: str | None = None
    ) -> ChatCompletionToolUsage:
        if content.type != InteractionType.TOOL_CALL:
            raise ValueError("Content is not a tool call.")

        if not isinstance(content.content, dict):
            raise ValueError("tool call content is not a dictionary.")

        function_name = content.content["name"]
        function_arguments = content.content["arguments"]
        if isinstance(function_arguments, dict):
            function_arguments = json.dumps(function_arguments)

        used_function = ChatCompletionToolUsage.UsedFunction(
            name=function_name,
            arguments=function_arguments,
        )

        return ChatCompletionToolUsage(
            id=tool_call_id or generate_tool_call_id(),
            function=used_function,
        )


class ChatCompletionToolChoice(BaseModel):
    """Defines a tool for the chat completion request."""

    class FunctionName(BaseModel):
        """Defines a function name for the chat completion tool."""

        name: str = Field(description="The name of the function to call.")

    type: Literal["function"] = "function"
    function: FunctionName = Field(description="The function to call.")

    def to_dict(self):
        return {"type": "function", "name": self.function.name}


class ChatCompletionToolUseMode(Enum):
    """Controls which (if any) tool is called by the model."""

    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"

    def to_dict(self):
        return self.value


class ChatCompletionFunction(BaseModel):
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


class ChatCompletionTool(BaseModel):
    """Defines a tool for the chat completion request."""

    type: Literal["function"] = "function"
    function: ChatCompletionFunction = Field(description="The function to call.")

    def to_dict(self) -> dict:
        return {
            "name": self.function.name,
            "type": "object",
            "description": self.function.description or self.function.name,
            "properties": {
                "name": {"const": self.function.name},
                "arguments": self.function.parameters,
            },
            "strict": self.function.strict,
            "required": ["name", "arguments"],
        }
