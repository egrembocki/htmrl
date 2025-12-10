"""Exposed interface for Agent layer components."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentInterface(Protocol):
    """Defines the interface for agent layer components."""

    def select_action(self, state: tuple) -> None:
        """Processes incoming data for the agent.

        Args:
            state (tuple): The current state of the environment.
        """
        ...

    def upate_policy(self) -> None:
        """Updated MLP network policy based on experience."""
        ...
