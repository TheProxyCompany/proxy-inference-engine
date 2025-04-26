from typing import Any

from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from proxy_inference_engine.state_machine.sub_state import SubState


class ReasoningState(SubState):
    """
    State for freeform reasoning.
    """

    def __init__(
        self,
        character_min: int | None = None,
        character_max: int | None = None,
        delimiters: tuple[str, str] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize a new StructuredOutputState.

        Args:
            delimiters: delimiters for the freeform reasoning state
        """
        super().__init__(identifier="reasoning", generation_kwargs=generation_kwargs)
        self.delimiters = delimiters or ("```thinking\n", "\n```")
        self.character_min = character_min or 50
        self.character_max = character_max or -1

    @property
    def state_machine(self) -> StateMachine:
        """
        Create a freeform state machine for reasoning.

        Returns:
            A StateMachine instance configured for freeform reasoning
        """
        return FencedFreeformStateMachine(
            self.identifier,
            self.delimiters,
            char_min=self.character_min,
            char_max=self.character_max,
        )
