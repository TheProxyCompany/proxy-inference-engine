from pse.types.misc.freeform import FreeformStateMachine
from pse_core.state_machine import StateMachine

from proxy_inference_engine.state_machine.sub_state import SubState


class FreeformTextState(SubState):
    """
    State for freeform text.
    """

    def __init__(
        self,
        end_delimiters: list[str],
        min_characters: int | None = None,
    ):
        """
        Initialize a new FreeformTextState.

        Args:
            end_delimiters: delimiters for the freeform text state
            min_characters: minimum number of characters to generate
        """
        super().__init__(identifier="text_output")
        self.end_delimiters = end_delimiters
        self.min_characters = min_characters

    @property
    def state_machine(self) -> StateMachine:
        """
        Create a freeform state machine for reasoning.

        Returns:
            A StateMachine instance configured for freeform reasoning
        """
        return FreeformStateMachine(
            end_delimiters=self.end_delimiters or [],
            char_min=self.min_characters,
        )
