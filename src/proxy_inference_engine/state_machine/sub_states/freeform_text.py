from pse.types.base.character import CharacterStateMachine
from pse_core import StateGraph
from pse_core.state_machine import StateMachine

from proxy_inference_engine.state_machine.sub_state import SubState


class FreeformTextState(SubState):
    """
    State for freeform text.
    """

    def __init__(self, secondary_state_machine: StateMachine | None = None):
        """
        Initialize a new FreeformTextState.

        Args:
            delimiters: delimiters for the freeform text state
        """
        super().__init__(identifier="text_output")
        self.secondary_state_machine = secondary_state_machine

    @property
    def state_machine(self) -> StateMachine:
        """
        Create a freeform state machine for reasoning.

        Returns:
            A StateMachine instance configured for freeform reasoning
        """
        return CharacterStateMachine()

class FreeformTextStateMachine(StateMachine):
    """
    State for freeform text.
    """

    def __init__(self, secondary_state_machine: StateMachine | None = None):
        text_output_sm = CharacterStateMachine()
        text_output_sm.identifier = "text_output"
        state_graph: StateGraph = {
            "start": [(text_output_sm, "end")],
        }
        if secondary_state_machine:
            state_graph["end"] = [(secondary_state_machine, "super_end")]

        super().__init__(
            state_graph,
            start_state="start",
            end_states=["end", "super_end"],
            identifier="freeform_text_with_wait_for",
        )
