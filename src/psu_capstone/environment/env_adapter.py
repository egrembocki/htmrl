"""Env Adapter for HTM-RL Brain-Agent to traditional Gymnasium environments."""

from typing import Any

import gymnasium as gym
import numpy as np


class EnvAdapter:
    """Adapter for a HTM-RL Brain Agent to Gymnasium interface.

    Args:
        gym_env: Gym environment id (for ``gym.make``) or a pre-built
            ``gym.Env`` instance (for example ``FinGym(...)``).
        **gym_kwargs: Optional kwargs forwarded to ``gym.make`` when
            ``gym_env`` is a string id.
    """

    def __init__(self, gym_env: str | gym.Env = "CartPole-v1", **gym_kwargs: Any) -> None:
        if isinstance(gym_env, str):
            self._env = gym.make(gym_env, **gym_kwargs)
        else:
            if gym_kwargs:
                raise ValueError("gym_kwargs can only be used when gym_env is a string id.")
            self._env = gym_env

        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._obs: Any | None = None

    def _to_serializable(self, value: Any) -> Any:
        """Convert numpy values to JSON-friendly Python types."""

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            return [self._to_serializable(item) for item in value]
        if isinstance(value, list):
            return [self._to_serializable(item) for item in value]
        if isinstance(value, dict):
            return {key: self._to_serializable(item) for key, item in value.items()}

        return value

    def _space_to_spec(self, space: gym.Space[Any]) -> dict[str, Any]:
        """Create a generic, structured description for any Gymnasium space."""

        if isinstance(space, gym.spaces.Discrete):
            return {
                "space_type": "Discrete",
                "n": space.n,
                "values": list(range(space.n)),
            }

        if isinstance(space, gym.spaces.MultiDiscrete):
            nvec = space.nvec.tolist()
            return {
                "space_type": "MultiDiscrete",
                "nvec": nvec,
                "values": [list(range(int(n))) for n in nvec],
            }

        if isinstance(space, gym.spaces.MultiBinary):
            return {
                "space_type": "MultiBinary",
                "n": self._to_serializable(space.n),
                "values": [0, 1],
            }

        if isinstance(space, gym.spaces.Box):
            return {
                "space_type": "Box",
                "shape": list(space.shape),
                "dtype": str(space.dtype),
                "low": self._to_serializable(space.low),
                "high": self._to_serializable(space.high),
            }

        if isinstance(space, gym.spaces.Tuple):
            return {
                "space_type": "Tuple",
                "spaces": [self._space_to_spec(subspace) for subspace in space.spaces],
            }

        if isinstance(space, gym.spaces.Dict):
            return {
                "space_type": "Dict",
                "spaces": {
                    key: self._space_to_spec(subspace) for key, subspace in space.spaces.items()
                },
            }

        return {
            "space_type": type(space).__name__,
            "sample": self._to_serializable(space.sample()),
        }

    def _flatten_value(self, value: Any, prefix: str) -> dict[str, Any]:
        """Flatten environment values into HTM-friendly key/value pairs."""

        serializable = self._to_serializable(value)

        if isinstance(serializable, dict):
            flattened: dict[str, Any] = {}
            for key, item in serializable.items():
                flattened.update(self._flatten_value(item, f"{prefix}_{key}"))
            return flattened

        if isinstance(serializable, list):
            flattened = {}
            for index, item in enumerate(serializable):
                flattened.update(self._flatten_value(item, f"{prefix}_{index}"))
            return flattened

        return {prefix: serializable}

    def _space_value_to_inputs(
        self,
        space: gym.Space[Any],
        value: Any,
        prefix: str,
    ) -> dict[str, Any]:
        """Convert an environment value into flat HTM input fields."""

        if isinstance(space, gym.spaces.Dict):
            flattened: dict[str, Any] = {}
            for key, subspace in space.spaces.items():
                flattened.update(self._space_value_to_inputs(subspace, value[key], str(key)))
            return flattened

        if isinstance(space, gym.spaces.Tuple):
            flattened = {}
            for index, subspace in enumerate(space.spaces):
                flattened.update(
                    self._space_value_to_inputs(subspace, value[index], f"{prefix}_{index}")
                )
            return flattened

        return self._flatten_value(value, prefix)

    def get_observation_spec(self) -> dict[str, Any]:
        """Get a generic description of the environment observation space."""

        return self._space_to_spec(self._observation_space)

    def get_action_spec(self) -> dict[str, Any]:
        """Get a generic description of the environment action space."""

        return self._space_to_spec(self._action_space)

    def observation_to_inputs(self, observation: Any) -> dict[str, Any]:
        """Convert a raw observation into flat HTM input fields."""

        return self._space_value_to_inputs(self._observation_space, observation, "observation")

    def action_to_inputs(self, action: Any) -> dict[str, Any]:
        """Convert an action value into flat HTM-style fields."""

        return self._space_value_to_inputs(self._action_space, action, "action")

    def get_observation(self) -> dict[str, Any]:
        """Get a generic description of the environment observation space."""

        return {"observations": self.get_observation_spec()}

    def get_action(self) -> dict[str, Any]:
        """Get a generic description of the environment action space."""

        return {"actions": self.get_action_spec()}

    def step(self, action: Any) -> dict[str, Any]:

        self._obs, self._reward, self._terminated, self._truncated, self._info = self._env.step(
            action
        )

        return {
            "obs": self._obs,
            "inputs": self.observation_to_inputs(self._obs),
            "reward": self._reward,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "info": self._info,
        }

    def reset(self) -> dict[str, Any]:

        obs, info = self._env.reset()
        self._obs = obs

        return {"obs": obs, "inputs": self.observation_to_inputs(obs), "info": info}
