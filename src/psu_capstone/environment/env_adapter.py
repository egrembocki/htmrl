"""Adapter between Gym environments and the Brain/Agent loop.

Think of this class as a translator sitting between two sides:

1. Gym side (normal RL calls):
    - reset() -> (obs, info)
    - step(action) -> (obs, reward, terminated, truncated, info)
2. Brain/Agent side (translated bridge dicts):
    - reset_bridge() -> {obs, inputs, info}
    - step_bridge() -> {obs, inputs, reward, terminated, truncated, info}

A bridge is just one dictionary that bundles:

- what Gym returned
- flattened inputs that are easy for the Brain to consume

Typical runtime flow:

1. caller creates EnvAdapter around a Gym env id or env instance
2. caller asks reset_bridge() for the first observation and HTM inputs
3. Brain consumes bridge["inputs"]
4. caller selects an action
5. caller asks step_bridge(action) for the next observation and HTM inputs
6. repeat until terminated or truncated is true

This keeps the adapter compatible with Gym while still giving the Brain the
simple key/value input map it expects.
"""

from typing import Any

import gymnasium as gym
import numpy as np


class EnvAdapter(gym.Wrapper):
    """Wrap any Gymnasium environment and translate data for HTM usage.

    This class does not change the environment rules. It only:

    - forwards normal Gym calls
    - exposes space metadata
    - flattens observations/actions into Brain-friendly fields

    Args:
        gym_env: Gym environment id (for gym.make) or any pre-built
            gym.Env instance (for example FinGym(...)).
        **gym_kwargs: Optional kwargs forwarded to gym.make when
            gym_env is a string id.
    """

    def __init__(self, gym_env: str | gym.Env = "CartPole-v1", **gym_kwargs: Any) -> None:
        if isinstance(gym_env, str):
            # If the caller gave an env id, create that env now.
            # env-id can be registered with gymnasium.envs.registration.register
            wrapped_env = gym.make(gym_env, **gym_kwargs)
        else:
            if gym_kwargs:
                raise ValueError("gym_kwargs can only be used when gym_env is a string id.")
            # If the caller already built an env, just wrap it.
            wrapped_env = gym_env

        super().__init__(wrapped_env)

        # Keep old private name for older code paths during migration.
        self._env = self.env

        # Keep old private names for compatibility with existing callers.
        self._observation_space = self.observation_space
        self._action_space = self.action_space
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
        """Return a plain metadata description of a Gym space.

        This does not return runtime values. It only describes shape/type
        details for debugging, introspection, or setup checks.
        """

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
        """Flatten nested values into a simple key/value map.

        Example:
            a nested value can turn into keys like observation_0 or
            action_price_open.
        """

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
        """Convert a value into flat Brain input fields.

        We walk Dict/Tuple spaces using the space definition so key names stay
        stable and predictable over time.
        """

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

        return self._space_to_spec(self.observation_space)

    def get_action_spec(self) -> dict[str, Any]:
        """Get a generic description of the environment action space."""

        return self._space_to_spec(self.action_space)

    def observation_to_inputs(self, observation: Any) -> dict[str, Any]:
        """Convert one observation into the Brain input map.

        Most callers use this through reset_bridge and step_bridge.
        """

        return self._space_value_to_inputs(self.observation_space, observation, "observation")

    def action_to_inputs(self, action: Any) -> dict[str, Any]:
        """Convert one action value into flat fields.

        Helpful when matching candidate actions against Brain predictions.
        """

        return self._space_value_to_inputs(self.action_space, action, "action")

    def get_observation(self) -> dict[str, Any]:
        """Get a generic description of the environment observation space."""

        return {"observations": self.get_observation_spec()}

    def get_action(self) -> dict[str, Any]:
        """Get a generic description of the environment action space."""

        return {"actions": self.get_action_spec()}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Run one normal Gym step.

        Use step_bridge if you also need Brain-ready flattened inputs.
        """

        self._obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs, float(reward), bool(terminated), bool(truncated), info

    def step_bridge(self, action: Any) -> dict[str, Any]:
        """Run one step and return the translated bridge dict.

        In plain terms: this is the handoff packet that bridges env-side data
        to Brain/Agent-side data.

        Normal Brain/Agent call order after choosing an action is:

        1. call step_bridge(action)
        2. read bridge["obs"] as the raw next observation
        3. read bridge["inputs"] as the Brain-ready input map
        4. inspect reward/done flags to decide whether the episode continues
        """

        obs, reward, terminated, truncated, info = self.step(action)

        return {
            "obs": obs,
            "inputs": self.observation_to_inputs(obs),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Run one normal Gym reset.

        Use reset_bridge if you immediately need Brain-ready inputs.
        """

        obs, info = self.env.reset(seed=seed, options=options)
        self._obs = obs
        return obs, info

    def reset_bridge(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset and return the translated bridge dict.

        Typical episode-start flow:

        1. call reset_bridge()
        2. feed bridge["inputs"] into the Brain
        3. choose an action
        4. continue with step_bridge(action)
        """

        obs, info = self.reset(seed=seed, options=options)

        return {"obs": obs, "inputs": self.observation_to_inputs(obs), "info": info}

    def render(self) -> Any:
        """Render using the wrapped environment's render behavior."""

        return self.env.render()

    def close(self) -> None:
        """Close the wrapped environment and release resources."""

        self.env.close()
