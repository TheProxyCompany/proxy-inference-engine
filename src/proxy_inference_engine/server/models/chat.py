from __future__ import annotations

import json
import secrets
import time
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from proxy_inference_engine.interaction import (
    Content,
    Interaction,
    InteractionRole,
    InteractionType,
)

# --- Constants ---
CHAT_COMPLETION_ID_PREFIX = "chatcmpl-"
CHAT_COMPLETION_OBJECT = "chat.completion"
TOOL_CALL_ID_PREFIX = "call-"

def generate_chat_completion_id(prefix: str = CHAT_COMPLETION_ID_PREFIX) -> str:
    """Generates a unique identifier string for a completion response."""
    random_part = secrets.token_urlsafe(22)
    return f"{prefix}{random_part}"

def generate_tool_call_id(prefix: str = TOOL_CALL_ID_PREFIX) -> str:
    """Generates a unique identifier string for a tool call."""
    random_part = secrets.token_urlsafe(22)
    return f"{prefix}{random_part}"


def get_current_timestamp() -> int:
    """Returns the current time as a Unix epoch timestamp (seconds)."""
    return int(time.time())


class ChatMessage(BaseModel):
    """Represents a single message within the chat conversation."""

    role: str = Field(description="The role of the messages author.")
    content: str | None = Field(description="The contents of the message.")
    tool_calls: list[ChatCompletionToolUsage] = Field(
        default=[],
        description="The tool calls that were made in the message.",
    )

    def to_interaction(self) -> Interaction:
        role = InteractionRole(self.role)
        content = []
        if self.content:
            content.append(Content.text(self.content))

        if self.tool_calls:
            for tool_call in self.tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                content.append(Content.tool_call(name, arguments))

        return Interaction(
            role,
            content,
        )

    @staticmethod
    def from_interaction(interaction: Interaction) -> ChatMessage:
        role = interaction.role.value
        if role == "agent":
            role = "assistant"

        content: str | None = None
        tool_calls: list[ChatCompletionToolUsage] = []
        for item in interaction.content:
            if item.type == InteractionType.TEXT:
                content = item.content
            elif item.type == InteractionType.TOOL_CALL:
                tool_calls.append(ChatCompletionToolUsage.from_content(item))

        return ChatMessage(role=role, content=content, tool_calls=tool_calls)

class ChatCompletionToolUsage(BaseModel):
    """Represents the usage of a tool in a chat completion."""

    class UsedFunction(BaseModel):
        """Represents a function that was used in a chat completion."""
        name: str = Field(description="The name of the function to call.")
        arguments: str = Field(description="The arguments to pass to the function. JSON encoded.")

    type: Literal["function"] = "function"
    id: str = Field(description="The unique identifier of the tool.")
    function: UsedFunction = Field(description="The function that was used.")

    @staticmethod
    def from_content(content: Content, tool_call_id: str | None = None) -> ChatCompletionToolUsage:
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


class ChatCompletionJSONSchemaResponseFormat(BaseModel):
    """Defines the response format for the chat completion request."""

    class JSONSchema(BaseModel):
        """Defines the JSON schema for the response format."""

        name: str = Field(description="The name of the JSON schema.")
        description: str | None = Field(
            default=None, description="The description of the JSON schema."
        )
        strict: bool | None = Field(
            default=None,
            description="Whether to enforce strict validation of the JSON schema.",
        )
        json_schema: dict = Field(
            description="The JSON schema for the response format.", alias="schema"
        )

    type: Literal["json_schema"] = "json_schema"
    json_schema: JSONSchema = Field(
        description="The JSON schema for the response format."
    )

    def to_dict(self):
        return {"type": "json_schema", **self.json_schema.model_dump()}


class ChatCompletionTextResponseFormat(BaseModel):
    """Defines the response format for the chat completion request."""

    type: Literal["text"] = "text"

    def to_dict(self):
        return self.model_dump()


class ChatCompletionRequest(BaseModel):
    """Defines the request schema for the chat completion endpoint."""

    model: str = Field(
        description="The identifier of the model designated for completion generation."
    )
    messages: list[ChatMessage] = Field(
        description="A list of messages comprising the conversation history.",
        min_length=1,
    )
    max_completion_tokens: int | None = Field(
        default=None,
        ge=1,
        description="The upper limit on the number of tokens to generate per completion.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Controls randomness via sampling temperature.",
    )
    top_p: float | None = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Implements nucleus sampling.",
    )
    top_k: int | None = Field(
        default=50,
        ge=1,
        le=100,
        description="Controls the number of tokens considered at each step.",
    )
    min_p: float | None = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold for token consideration.",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether to allow the model to run tool calls in parallel.",
    )
    tool_choice: ChatCompletionToolUseMode | ChatCompletionToolChoice | None = Field(
        default=None,
        description="Controls which (if any) tool is called by the model.",
    )
    tools: list[ChatCompletionTool] | None = Field(
        default=None,
        description="A list of tools that the model can use to generate a response.",
    )
    response_format: (
        ChatCompletionTextResponseFormat | ChatCompletionJSONSchemaResponseFormat | None
    ) = Field(
        default=None,
        description="The format of the response.",
    )


class ChatCompletionChoice(BaseModel):
    """Represents a single generated chat completion choice."""

    index: int = Field(description="The index of this choice.")
    message: ChatMessage = Field(description="The message generated by the model.")
    finish_reason: str | None = Field(
        description="Reason generation stopped (e.g., 'stop', 'length', 'tool_calls')."
    )


class ChatCompletionUsage(BaseModel):
    """Provides token usage statistics for the chat completion request."""

    input_tokens: int = Field(
        description="The number of tokens constituting the input prompt(s)."
    )
    output_tokens: int = Field(
        description="The total number of tokens generated across all completion choices."
    )
    total_tokens: int = Field(
        description="The sum of `input_tokens` and `output_tokens`."
    )


# --- Main Response Model ---
class ChatCompletionResponse(BaseModel):
    """Defines the response schema for the chat completion endpoint."""

    id: str = Field(
        default_factory=generate_chat_completion_id,
        description="A unique identifier for the chat completion.",
    )
    object: str = Field(
        default=CHAT_COMPLETION_OBJECT,
        description="The object type, always 'chat.completion'.",
    )
    created: int = Field(
        default_factory=get_current_timestamp,
        description="The Unix timestamp when the completion was created.",
    )
    model: str = Field(description="The model used for the chat completion.")
    choices: list[ChatCompletionChoice] = Field(
        description="A list of chat completion choices."
    )
    usage: ChatCompletionUsage = Field(
        description="Usage statistics for the completion request."
    )
    system_fingerprint: str | None = Field(
        default=None,
        description="System fingerprint representing the backend configuration.",
    )
