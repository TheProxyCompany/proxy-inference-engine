from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
        model_config = ConfigDict(
            populate_by_name=True,
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


class ChatCompletionJsonObjectResponseFormat(BaseModel):
    """Defines the response format for the chat completion request."""

    type: Literal["json_object"] = "json_object"

    def to_dict(self):
        return self.model_dump()
