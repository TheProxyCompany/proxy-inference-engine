from __future__ import annotations

import json

from pydantic import BaseModel, Field

from proxy_inference_engine.interaction import (
    Content,
    Interaction,
    InteractionRole,
    InteractionType,
)
from proxy_inference_engine.server.models.chat.format import (
    ChatCompletionJsonObjectResponseFormat,
    ChatCompletionJSONSchemaResponseFormat,
    ChatCompletionTextResponseFormat,
)
from proxy_inference_engine.server.models.chat.tools import (
    ChatCompletionTool,
    ChatCompletionToolChoice,
    ChatCompletionToolUsage,
    ChatCompletionToolUseMode,
)


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
    logprobs: bool | None = Field(
        default=False,
        description="Whether to include the log probabilities of each token in the response.",
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
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description="The number of top log probabilities to include in the response.",
    )
    response_format: (
        ChatCompletionTextResponseFormat
        | ChatCompletionJSONSchemaResponseFormat
        | ChatCompletionJsonObjectResponseFormat
        | None
    ) = Field(
        default=None,
        description="The format of the response.",
    )
    stop: str | list[str] | None = Field(
        default=None,
        description="A list of tokens to stop generation of the response. The returned text will not contain the stop sequence.",
    )

    stream: bool | None = Field(
        default=False,
        description="Whether to stream the response to the client using Server-Sent Events.",
    )
    stream_options: ChatCompletionStreamOptions | None = Field(
        default=None,
        description="Additional options for streaming the response.",
    )


class ChatCompletionStreamOptions(BaseModel):
    """Additional options for streaming the response."""

    include_usage: bool | None = Field(
        default=False,
        description="Whether to include the usage statistics in the response.",
    )
