from enum import Enum


class InteractionRole(Enum):
    """
    Enumeration of possible roles for an interaction.
    """

    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"


class InteractionType(Enum):
    """
    Enumeration of possible types for an interaction.
    """

    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    TOOL_CALL = "tool_call"


from proxy_inference_engine.interaction.content import Content  # noqa: E402
from proxy_inference_engine.interaction.interaction import Interaction  # noqa: E402

__all__ = ["Content", "Interaction"]
