"""Environment module for the HTM brain to observe, take actions, and gain rewards."""

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from psu_capstone.log import get_logger


class Environment:
    """Environment class for the HTM brain to observe, take actions, and gain rewards."""

    def __init__(self, observation_space: tuple[Any, ...], action_space: tuple[Any, ...]) -> None:

        self._logger = get_logger("Environment")
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
        self._episode_count: int = 0
        self._step_count: int = 0
        self._logger.info("Environment initialized.")

    @property
    def observation_space(self) -> tuple[Any, ...]:
        """Returns the observation space of the environment."""
        if self._observation_space is None:
            raise ValueError("Observation space has not been set.")
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

        if self._get_shape(value) != self._get_shape(self._observation_space):
            raise ValueError("Shape of the new state does not match the observation space.")
        self._current_state = self._coerce_to_shape(value, self._observation_space)

    def _compute_reward(self, action: Any) -> float:
        """Computes the reward based on the action taken and the new state of the environment."""
        return 0.0  # Placeholder implementation

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

    def _get_shape(self, seq: Sequence[Any]) -> tuple[int, ...]:
        """Get the shape of a nested sequence structure."""
        if not isinstance(seq, Sequence) or isinstance(seq, (str, bytes)):
            return ()
        return (len(seq),) + self._get_shape(seq[0]) if seq else (0,)

    def reset(self) -> None:
        """Resets the environment to its initial state."""

        self._observation_space = copy.deepcopy(
            self._obs_space
        )  # restore the original observation space

        self._current_state = self._coerce_to_shape((), self._observation_space)
        self._next_state = self._current_state
        self._reward = 0.0
        self._done = False
        self._episode_count += 1
        self._step_count = 0

    def step(self, action: Any) -> None:
        """Takes a step in the environment using the provided action.

        Args:
            action: The action to be taken in the environment.
        """

        # TODO: Calculate the reward and determine if the episode is done based on the action taken and the new state of the environment.
        # Then call update() to update the environment's state based on the action taken.
        self._step_count += 1
        self._next_state = (
            self._current_state
        )  # In a real implementation, this would be updated based on the action
        self._reward = self._compute_reward(
            action
        )  # In a real implementation, this would be calculated based on the action and the new state

        self.update()

    def update(self) -> None:
        """Updates the environment's state based on the action taken."""

        # TODO: Implement the logic to update the environment's state based on the action taken.
        # This is a placeholder implementation and should be replaced with actual logic.
        # TODO: This may need to be a while loop with a worker thread to allow for multiple updates until the episode is done.
        self._current_state = (
            self._next_state
        )  # In a real implementation, this would be updated based on the action
        self._done = (
            False  # In a real implementation, this would be set to True if the episode has ended
        )

    def render(self) -> None:
        """Renders the current state of the environment."""
        pass

    def observe(self) -> tuple[float, Any, Any]:
        """Returns the current observation from the environment."""
        return (self._reward, self._current_state, self._done)


@dataclass
class EnvironmentConfig:
    """Configuration class for the Environment."""

    observation_space: tuple[Any, ...]
    action_space: tuple[Any, ...]
    metrics_thresholds: dict[str, float] = {}
    metric_map: dict[str, tuple] = {}


if __name__ == "__main__":

    # Example usage of the Environment class
    env = Environment(observation_space=(0.0, 0, "state"), action_space=(0, 1, "action"))
    env.reset()
    print("Initial observation:", env.observe())
    env.step(1)
    print("Observation after taking action 1:", env.observe())
