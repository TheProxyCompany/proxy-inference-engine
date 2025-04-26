from abc import abstractmethod
from typing import Any

from pse_core.state_machine import StateMachine


class SubState:
    """
    A sub state for the root state machine.
    """

    def __init__(self, identifier: str, generation_kwargs: dict[str, Any] | None = None):
        self.identifier = identifier
        self.specific_kwargs: dict[str, Any] = {}
        if generation_kwargs and generation_kwargs.get(self.identifier, None):
            self.specific_kwargs = generation_kwargs[self.identifier]


    @property
    def generation_kwargs(self) -> dict[str, Any]:
        return self.specific_kwargs

    @property
    @abstractmethod
    def state_machine(self) -> StateMachine:
        pass
