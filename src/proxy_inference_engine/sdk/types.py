from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from proxy_inference_engine.interaction import Interaction, InteractionRole


class GenerationRequest(BaseModel):
    """
    A request for a generation.
    """

    prompt: str | list[dict[str, str]] | Interaction | list[Interaction]
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_delimiters: tuple[str, str] | None = None
    parallel_tool_calls: bool | None = None
    tool_choice: str | dict[str, Any] | None = None
    generation_kwargs: dict[str, GenerationKwargs] | None = None

    def to_interactions(self) -> list[Interaction]:
        """
        Convert the generation request to an interaction.
        """
        if isinstance(self.prompt, str):
            return [Interaction.simple(InteractionRole.USER, self.prompt)]
        elif isinstance(self.prompt, Interaction):
            return [self.prompt]
        elif isinstance(self.prompt, list):
            result: list[Interaction] = []
            for p in self.prompt:
                if isinstance(p, dict):
                    role = p.get("role", "user")
                    content = p.get("content", "")
                    try:
                        interaction_role = InteractionRole(role)
                        result.append(Interaction.simple(interaction_role, content))
                    except ValueError:
                        result.append(Interaction.simple(InteractionRole.USER, content))
                elif isinstance(p, Interaction):
                    result.append(p)

            return result
        else:
            raise ValueError("Invalid prompt type")

class GenerationKwargs(BaseModel):
    """
    Generation kwargs.
    """

    min_p: float | None = None
    max_p: float | None = None
    min_tokens_to_keep: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    temperature: float | None = None
    repetition_penalty: float | None = None
