from pse.types.misc.fenced_freeform import FencedFreeformStateMachine
from pse_core.state_machine import StateMachine

from proxy_inference_engine.state_machine.sub_state import SubState


class ReasoningState(SubState):
    """
    State for freeform reasoning.
    """

    def __init__(
        self,
        delimiters: tuple[str, str] | None = None,
    ):
        """
        Initialize a new StructuredOutputState.

        Args:
            delimiters: delimiters for the freeform reasoning state
        """
        super().__init__(identifier="reasoning")
        self.delimiters = delimiters or ("```thinking\n", "\n```")

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
            char_min=50,
        )
