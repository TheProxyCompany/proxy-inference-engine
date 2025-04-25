from abc import abstractmethod

from pse_core.state_machine import StateMachine


class SubState:
    """
    A sub state for the root state machine.
    """

    def __init__(self, identifier: str):
        self.identifier = identifier

    @property
    @abstractmethod
    def state_machine(self) -> StateMachine:
        pass
