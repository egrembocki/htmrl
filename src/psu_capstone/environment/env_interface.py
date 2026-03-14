"""Exposed interfaces for environment layer components."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvInterface(Protocol):
    """Defines the interface for environment layer components."""

    def step(self, action: Any) -> None:
        """Takes a step in the environment using the provided action.

        Args:
            action: The action to be taken in the environment.
        """
        ...

    def render(self) -> None:
        """Renders the current state of the environment."""
        ...

    def close(self) -> tuple[float, Any, Any]:
        """Closes the environment and returns the final observation."""
        ...

    def reset(self) -> None:
        """Resets the environment to its initial state."""
        ...
