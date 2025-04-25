from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResponseFormat(BaseModel):
    """Defines the response format for the response request."""

    format: TextResponseFormat | JSONSchemaResponseFormat = Field(description="The response format to use.")

    def to_dict(self):
        return self.format.model_dump()

class TextResponseFormat(BaseModel):
    """Defines the response format for the response request."""

    type: Literal["text"] = "text"

class JSONSchemaResponseFormat(BaseModel):
    """Defines the JSON schema for the response format."""

    type: Literal["json_schema"] = "json_schema"
    json_schema: dict = Field(description="The JSON schema to use.", alias="schema")
    name: str = Field(description="The name of the JSON schema.")
    description: str | None = Field(
        default="",
        description="The description of the JSON schema.",
    )
    strict: bool | None = Field(
        default=False,
        description="Whether to enforce strict validation of the JSON schema.",
    )
