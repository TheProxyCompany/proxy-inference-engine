from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ResponseFormatType(Enum):
    """Defines the type of response format for the response request."""

    TEXT = "text"
    JSON_SCHEMA = "json_schema"

class ResponseFormat(BaseModel):
    """Defines the response format for the response request."""

    type: ResponseFormatType = Field(description="The type of response format to use.")
    response_format: str = Field(description="The response format to use.")

class JSONSchema(BaseModel):
    """Defines the JSON schema for the response format."""

    type: Literal["json_schema"] = "json_schema"
    schema: dict = Field(description="The JSON schema to use.")
    name: str = Field(description="The name of the JSON schema.")
    description: str | None = Field(
        default="",
        description="The description of the JSON schema.",
    )
    strict: bool | None = Field(
        default=False,
        description="Whether to enforce strict validation of the JSON schema.",
    )
