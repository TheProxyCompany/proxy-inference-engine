from enum import Enum


class Role(Enum):
    """
    Enumeration of possible roles for an interaction.
    """

    AGENT = "agent"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"

class Type(Enum):
    """
    Enumeration of possible types for an interaction.
    """

    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    ACTION = "action"
