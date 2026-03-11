"""Environment module for the HTM brain to observe, take actions, and gain rewards."""

import copy
from collections.abc import Sequence
from typing import Any

from psu_capstone.environment.env_interface import EnvInterface


class Environment(EnvInterface):
    """Environment class for the HTM brain to observe, take actions, and gain rewards."""

    def __init__(self, observation_space: tuple[Any, ...], action_space: tuple[Any, ...]) -> None:

        # define the shape of the action space as a tuple of any values (e.g., (0, 1, 'action'))
        self._action_space: tuple[Any, ...] = action_space
        # define the shape of the observation space as a tuple of any values (e.g., (0.0, 0, 'state'))
        self._observation_space: tuple[Any, ...] = observation_space

        self._obs_space: tuple[Any, ...] = copy.deepcopy(
            observation_space
        )  # save a snapshot of the original observation space for resets
        self._current_state: tuple[Any, ...] = self._coerce_to_shape((), self._obs_space)
        self._next_state: tuple[Any, ...] = self._current_state
        self._reward: float = 0.0
        self._done: bool = False

    @property
    def observation_space(self) -> tuple[Any, ...]:
        """Returns the observation space of the environment."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value: tuple[Any, ...]) -> None:
        """Sets the observation space of the environment."""
        self._observation_space = value  # update the observation space

    @property
    def current_state(self) -> tuple[Any, ...]:
        """Returns the current state shaped like the observation space."""
        return self._current_state

    @current_state.setter
    def current_state(self, value: Sequence[Any]) -> None:
        """Sets current state while enforcing observation-space shape."""
        self._current_state = self._coerce_to_shape(value, self._observation_space)

    def _coerce_to_shape(self, values: Sequence[Any], shape: tuple[Any, ...]) -> tuple[Any, ...]:
        """Return values padded/truncated to match the observation-space tuple shape."""
        result: list[Any] = []

        for idx, template in enumerate(shape):
            value = values[idx] if idx < len(values) else template
            if isinstance(template, tuple):
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    nested_values: Sequence[Any] = value
                else:
                    nested_values = ()
                result.append(self._coerce_to_shape(nested_values, template))
            else:
                result.append(value)

        return tuple(result)

    def reset(self) -> None:
        """Resets the environment to its initial state."""
        self._observation_space = copy.deepcopy(
            self._obs_space
        )  # restore the original observation space
        self._current_state = self._coerce_to_shape((), self._observation_space)
        self._next_state = self._current_state
        self._reward = 0.0
        self._done = False

    def step(self, action: Any) -> None:
        """Takes a step in the environment using the provided action.

        Args:
            action: The action to be taken in the environment.
        """
        del action
        self.update()

    def update(self) -> None:
        self._current_state = self._coerce_to_shape(self._next_state, self._observation_space)

    def render(self) -> None:
        """Renders the current state of the environment."""
        pass

    def observe(self) -> tuple[float, Any, Any]:
        """Returns the current observation from the environment."""
        return (self._reward, self._current_state, self._done)
