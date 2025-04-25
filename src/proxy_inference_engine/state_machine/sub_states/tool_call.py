from typing import Any

from pse.types.json import json_schema_state_machine
from pse_core.state_machine import StateMachine

from proxy_inference_engine.state_machine.sub_state import SubState


class ToolCallState(SubState):
    """
    State for handling tool calls during agent execution.

    The ToolCallState enables agents to interact with their environment by providing
    a structured interface for invoking external tools. It creates a JSON schema-based
    state machine that ensures tool calls have the correct format, required parameters,
    and follow the expected structure.

    This state is a key component of the agent's ability to take action in the world,
    allowing it to perform operations like web searches, calculations, or API calls.
    """

    def __init__(
        self,
        tools: list[dict[str, Any]],
        delimiters: tuple[str, str] | None = ("```tool\n", "\n```"),
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ):
        """
        Initialize a new ToolCallState.

        Args:
            tools: List of Tool objects available for use in this state
            delimiters: Optional custom delimiters for the tool call state
            list_delimiters: Optional delimiters for the tool list in the prompt
        """
        super().__init__(identifier="tool_call")
        self.delimiters = delimiters
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls

        if tool_choice is None or (
            isinstance(tool_choice, str) and tool_choice != "none"
        ):
            self.tools = tools
        elif isinstance(tool_choice, dict) and "name" in tool_choice:
            requested_function_name = tool_choice["name"]
            self.tools = [
                tool for tool in tools if tool["name"] == requested_function_name
            ]
        else:
            self.tools = []

    @property
    def generation_kwargs(self) -> dict[str, Any]:
        return {"temperature": 0.0, "repetition_penalty": 1.0, "min_p": 0.02}

    @property
    def state_machine(self) -> StateMachine:
        """
        Create a JSON schema-based state machine for tool calls.

        This property dynamically generates a state machine that validates tool calls
        against their JSON schemas, ensuring all required parameters are provided and
        correctly formatted.

        Returns:
            A StateMachine instance configured for tool invocation validation
        """
        if self.parallel_tool_calls:
            schema = {"type": "array", "items": {"oneOf": self.tools}}
        else:
            schema = self.tools

        _, state_machine = json_schema_state_machine(schema, self.delimiters)
        state_machine.identifier = self.identifier
        return state_machine
