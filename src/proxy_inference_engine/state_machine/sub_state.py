from abc import abstractmethod
from typing import Any

from pse_core.state_machine import StateMachine


class SubState:
    """
    A sub state for the root state machine.
    """

    def __init__(self, identifier: str):
        self.identifier = identifier

    @property
    def generation_kwargs(self) -> dict[str, Any]:
        return {}

    @property
    @abstractmethod
    def state_machine(self) -> StateMachine:
        pass
