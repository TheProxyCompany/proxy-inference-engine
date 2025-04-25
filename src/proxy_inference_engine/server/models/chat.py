import secrets
import time
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from proxy_inference_engine.interaction import (
    Content,
    Interaction,
    Role,
)

# --- Constants ---
CHAT_COMPLETION_ID_PREFIX = "chatcmpl-"
CHAT_COMPLETION_OBJECT = "chat.completion"


def generate_chat_completion_id(prefix: str = CHAT_COMPLETION_ID_PREFIX) -> str:
    """Generates a unique identifier string for a completion response."""
    random_part = secrets.token_urlsafe(22)
    return f"{prefix}{random_part}"


def get_current_timestamp() -> int:
    """Returns the current time as a Unix epoch timestamp (seconds)."""
    return int(time.time())


class ChatMessage(BaseModel):
    """Represents a single message within the chat conversation."""

    role: str = Field(description="The role of the messages author.")
    content: str = Field(description="The contents of the message.")

    def to_interaction(self) -> Interaction:
        role = Role(self.role)
        return Interaction(
            role,
            [Content.text(self.content)],
        )

class ChatCompletionToolChoice(BaseModel):
    """Defines a tool for the chat completion request."""

    class FunctionName(BaseModel):
        """Defines a function name for the chat completion tool."""

        name: str = Field(description="The name of the function to call.")

    type: Literal["function"] = "function"
    function: FunctionName = Field(description="The function to call.")

class ChatCompletionToolUseMode(Enum):
    """Controls which (if any) tool is called by the model."""

    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"

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


class ChatCompletionJSONSchemaResponseFormat(BaseModel):
    """Defines the response format for the chat completion request."""

    class JSONSchema(BaseModel):
        """Defines the JSON schema for the response format."""

        name: str = Field(description="The name of the JSON schema.")
        description: str | None = Field(
            default=None,
            description="The description of the JSON schema."
        )
        strict: bool | None = Field(
            default=None,
            description="Whether to enforce strict validation of the JSON schema."
        )
        schema: dict = Field(description="The JSON schema for the response format.")
        
        model_config = {"protected_namespaces": ()}

    type: Literal["json_schema"] = "json_schema"
    json_schema: JSONSchema = Field(description="The JSON schema for the response format.")

class ChatCompletionTextResponseFormat(BaseModel):
    """Defines the response format for the chat completion request."""

    type: Literal["text"] = "text"


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
