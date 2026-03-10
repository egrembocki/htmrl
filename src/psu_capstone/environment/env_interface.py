"""Exposed interfaces for environment layer components."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvInterface(Protocol):
    """Defines the interface for environment layer components."""

    def setup(self) -> None:
        """Resets the environment to its initial state."""
        ...

    def step(self, action: Any) -> None:
        """Takes a step in the environment using the provided action.

        Args:
            action: The action to be taken in the environment.
        """
        ...

    def render(self) -> None:
        """Renders the current state of the environment."""
        ...

    def observe(self) -> tuple[float, Any, Any]:
        """Returns the current observation from the environment."""
        ...

    def update(self) -> None:
        """Updates the environment's state."""
        ...
