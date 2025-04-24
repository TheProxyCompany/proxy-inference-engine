from __future__ import annotations

from typing import Any

from proxy_inference_engine.interaction import Type


class Content:
    """
    Represents the content of an interaction.
    """

    def __init__(self, type: Type, content: Any):
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
            case Type.TEXT:
                content_dict["text"] = self.content
            case Type.IMAGE:
                content_dict["image_url"] = self.content
            case Type.ACTION:
                content_dict["action"] = self.content
            case _:
                content_dict["file_url"] = self.content

        return content_dict

    def __str__(self) -> str:
        return self.content

    @staticmethod
    def text(content: str) -> Content:
        return Content(Type.TEXT, content)
