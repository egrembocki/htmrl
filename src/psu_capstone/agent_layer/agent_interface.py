"""Exposed interface for Agent layer components."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentInterface(Protocol):
    """Defines the interface for agent layer components."""

    def perceive(self, input_data: Any) -> None:
        """Processes incoming data for the agent.

        Args:
            input_data (Any): The data to be processed by the agent.
        """
        ...

    def act(self) -> Any:
        """Generates an action based on the agent's current state.

        Returns:
            Any: The action produced by the agent.
        """
        ...
