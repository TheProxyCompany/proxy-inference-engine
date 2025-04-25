from typing import Any

from pse.structuring_engine import StateMachine
from pse.types.base.any import AnyStateMachine
from pse_core import StateGraph

from proxy_inference_engine.state_machine.sub_state import SubState
from proxy_inference_engine.state_machine.sub_states import (
    FreeformTextState,
    StructuredOutputState,
    ToolCallState,
)


class RootStateMachine(StateMachine):
    """The root state machine for the proxy inference engine."""

    def __init__(self):
        """
        Initialize the root state machine.
        """
        super().__init__(
            self._create_state_graph(),
            start_state="start",
            end_states=["end"],
            identifier="root",
        )

    def configure(
        self,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_delimiters: tuple[str, str] | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ):
        """
        Configure the root state machine.
        """
        self.state_graph = self._create_state_graph(
            response_format,
            tools,
            tool_delimiters,
            parallel_tool_calls,
            tool_choice,
        )


    def _create_state_graph(
        self,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_delimiters: tuple[str, str] | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> StateGraph:
        """
        Create and configure all states for the root state machine.
        """
        self.available_states: dict[str, SubState] = {}

        response_format = response_format or {"type": "text"}

        if response_format.get("type") == "json_schema":
            structured_output_state = StructuredOutputState(response_format, delimiters=None)
            self.available_states[structured_output_state.identifier] = structured_output_state
        elif tools and tool_choice != "none":
            tool_call_state = ToolCallState(
                tools,
                tool_delimiters,
                tool_choice,
                parallel_tool_calls,
            )
            self.available_states[tool_call_state.identifier] = tool_call_state

        if response_format.get("type") == "text" and tool_choice != "required":
            freeform_text = FreeformTextState()
            self.available_states[freeform_text.identifier] = freeform_text

        states = [state.state_machine for state in self.available_states.values()]
        root_state_machine: StateMachine = (
            AnyStateMachine(states)
            if len(states) > 1
            else states[0]
        )

        return { "start": [(root_state_machine, "end")] }
