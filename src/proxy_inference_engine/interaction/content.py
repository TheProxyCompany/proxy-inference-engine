from __future__ import annotations

import uuid
from typing import Any

from proxy_inference_engine.interaction import InteractionType


class Content:
    """
    Represents the content of an interaction.
    """

    def __init__(self, type: InteractionType, content: Any):
        self.id = "content-" + str(uuid.uuid4())
        self.type = type
        self.content = content

    def to_dict(self) -> dict:
        """
        Convert the content to a dictionary representation.
        """
        content_dict = {
            "type": self.type.value,
        }
        match self.type:
            case InteractionType.TEXT:
                content_dict["text"] = self.content
            case InteractionType.IMAGE:
                content_dict["image_url"] = self.content
            case InteractionType.TOOL_CALL:
                content_dict["tool_call"] = self.content
            case _:
                content_dict["file_url"] = self.content

        return content_dict

    def __str__(self) -> str:
        return self.content

    @staticmethod
    def text(content: str) -> Content:
        return Content(InteractionType.TEXT, content)

    @staticmethod
    def tool_call(name: str, arguments: dict[str, Any]) -> Content:
        return Content(
            InteractionType.TOOL_CALL, {"name": name, "arguments": arguments}
        )
