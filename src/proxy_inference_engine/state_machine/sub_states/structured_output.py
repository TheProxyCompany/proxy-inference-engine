from typing import Any

from pse.types.json import json_schema_state_machine
from pse_core.state_machine import StateMachine

from proxy_inference_engine.state_machine.sub_state import SubState


class StructuredOutputState(SubState):
    """
    State for handling structured output during agent execution.
    """

    def __init__(
        self,
        json_schema: dict[str, Any],
        delimiters: tuple[str, str] | None = ("```json\n", "\n```"),
    ):
        """
        Initialize a new StructuredOutputState.

        Args:
            json_schema: JSON schema for the structured output
            delimiters: Optional custom delimiters for the structured output state
        """
        super().__init__(identifier="structured_output")
        self.delimiters = delimiters
        self.json_schema = json_schema

    @property
    def state_machine(self) -> StateMachine:
        """
        Create a JSON schema-based state machine for structured output.

        This property dynamically generates a state machine that validates structured
        output against their JSON schemas, ensuring all required parameters are provided
        and correctly formatted.

        Returns:
            A StateMachine instance configured for structured output validation
        """
        _, state_machine = json_schema_state_machine(
            self.json_schema,
            self.delimiters
        )
        state_machine.identifier = self.identifier
        return state_machine
