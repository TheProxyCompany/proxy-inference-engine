from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

from proxy_inference_engine.interaction import InteractionRole
from proxy_inference_engine.interaction.content import Content


class Interaction:
    """
    Represents a single interaction.

    Interactions are uniquely identified by an event_id and can be converted to
    dictionaries for serialization.
    """

    def __init__(
        self,
        role: InteractionRole,
        content: list[Content],
        **kwargs,
    ) -> None:
        """
        Initialize a new Interaction.

        Args:
            event_id: Unique identifier for this interaction (auto-generated if None)
            name: Optional name or identifier for the creator of this interaction
            role: The role of this interaction (SYSTEM, USER, ASSISTANT, or TOOL)
            content: The primary content of the interaction (text, structured data, etc.)
            **kwargs: Additional metadata attributes to store with this interaction
                      Common metadata includes title, color, emoji for display styling
        """
        self.created_at = datetime.now()
        self.event_id = str(uuid.uuid4())
        self.role = role
        self.content = content
        self.metadata = kwargs

    def to_dict(self) -> dict:
        """
        Convert this interaction to a dictionary representation.

        This method serializes the interaction into a dictionary format
        suitable for:
        - Passing to language models as context
        - Storing in memory/databases
        - Converting to JSON for APIs

        Returns:
            A dictionary containing all relevant interaction data
        """
        # Initialize with core attributes
        dict: dict[str, Any] = {
            "event_id": self.event_id,
            "role": self.role.value,
        }

        if self.content:
            content = [str(content) for content in self.content]
            dict["content"] = content[0] if len(content) == 1 else content

        for key, value in self.metadata.items():
            if value and hasattr(value, "to_dict"):
                dict[key] = value.to_dict()
            else:
                dict[key] = value

        return dict

    def __str__(self) -> str:
        """Convert to a JSON string representation for debugging and logging."""
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self) -> str:
        """Return string representation for REPL and debugging."""
        return self.__str__()

    def __eq__(self, other):
        """
        Check equality by comparing event_ids.

        Two interactions are considered equal if they have the same event_id,
        regardless of any other differences in their content or metadata.
        """
        if not isinstance(other, Interaction):
            return False
        return self.event_id == other.event_id

    def __hash__(self):
        """Create a hash based on event_id for use in sets and dictionaries."""
        return hash(self.event_id)

    def __getattribute__(self, name: str) -> Any:
        """
        Enhanced attribute access that transparently exposes metadata attributes.

        This magic method allows metadata attributes to be accessed directly as if they
        were instance attributes. For example, if an Interaction has metadata["title"],
        you can access it using interaction.title.

        The lookup order is:
        1. Look for actual attributes on the instance
        2. If not found, check if it exists in metadata
        3. If not in metadata, return None

        This creates a more convenient API for accessing metadata fields.
        """
        try:
            # First try to get the actual attribute
            return object.__getattribute__(self, name)
        except AttributeError:
            # If not found, check if it's in metadata
            metadata = object.__getattribute__(self, "metadata")
            if name in metadata:
                return metadata[name]
            return None

    @staticmethod
    def simple(role: InteractionRole, content: str) -> Interaction:
        return Interaction(
            role,
            [Content.text(content)],
        )
