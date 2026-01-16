"""Exposed interfaces for environment layer components."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvInterface(Protocol):
    """Defines the interface for environment layer components."""

    def reset(self) -> None:
        """Resets the environment to its initial state."""
        ...

    def step(self, action: Any) -> None:
        """Takes a step in the environment using the provided action.

        Args:
            action (Any): The action to be taken in the environment."""

        ...

    def render(self) -> None:
        """Renders the current state of the environment."""
        ...

    def close(self) -> None:
        """Cleans up resources used by the environment."""
        ...

    def observe(self) -> tuple:
        """Returns the current observation from the environment."""
        ...

    def reward(self, action: tuple) -> tuple:
        """Calculate the reward back to the Agent"""

        ...
