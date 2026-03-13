"""Agent orchestration for HTM-driven reinforcement learning.

The Agent owns the runtime control loop between Brain and environment.
It reads observations, asks Brain to process inputs, selects an action
with the configured policy, steps the environment, and updates policy
state from the resulting transition.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Literal

import numpy as np
from stable_baselines3 import PPO

from psu_capstone.agent_layer.brain import Brain
from psu_capstone.environment.env_adapter import EnvAdapter


class Agent:
    """Coordinate Brain inference, policy selection, and environment interaction."""

    def __init__(
        self,
        brain: Brain,
        adapter: EnvAdapter,
        episodes: int = 1000,
        policy_mode: Literal["q_table", "brain", "ppo"] = "q_table",
        ppo_policy: str = "MlpPolicy",
        ppo_kwargs: dict[str, Any] | None = None,
        ppo_deterministic: bool = True,
    ) -> None:
        self._learning_rate: float = 0.1
        self._discount_factor: float = 0.99
        self._epsilon: float = 1.0
        self._epsilon_decay: float = 0.01

        self._brain = brain
        self._adapter = adapter
        self._policy_mode: Literal["q_table", "brain", "ppo"] = policy_mode
        self._ppo_policy = ppo_policy
        self._ppo_kwargs = ppo_kwargs or {}
        self._ppo_model: PPO | None = None
        self._ppo_deterministic = ppo_deterministic

        action_spec = self._adapter.get_action_spec()
        action_count = action_spec.get("n")
        is_integer_like = isinstance(action_count, (int, np.integer))
        if self._policy_mode == "q_table" and (
            action_spec.get("space_type") != "Discrete" or not is_integer_like
        ):
            raise TypeError("q_table policy requires a Discrete action space.")

        self._action_count: int | None = int(action_count) if is_integer_like else None
        self._q_values: defaultdict[Any, np.ndarray] = defaultdict(self._new_q_row)

        self._training_episodes: int = episodes
        self._training_error: list[float] = []
        self._obs: Any | None = None
        self._inputs: dict[str, Any] = {}
        self._rng = random.Random()

        if self._policy_mode == "ppo":
            self._ppo_model = self._build_ppo_model()

    def _build_ppo_model(self) -> PPO:
        """Create a Stable-Baselines3 PPO model bound to adapter env."""

        if not hasattr(self._adapter, "_env"):
            raise ValueError("ppo policy mode requires an adapter with a Gym environment (_env).")

        env = self._adapter._env
        return PPO(self._ppo_policy, env, verbose=0, **self._ppo_kwargs)

    def _select_ppo_action(self, obs: Any) -> Any:
        """Select action via PPO model predict."""

        if self._ppo_model is None:
            self._ppo_model = self._build_ppo_model()

        action, _state = self._ppo_model.predict(obs, deterministic=self._ppo_deterministic)
        if isinstance(action, np.ndarray) and action.shape == ():
            return action.item()
        return action

    def _new_q_row(self) -> np.ndarray:
        """Create a zero-initialized action-value row for one state."""

        if self._action_count is None:
            return np.array([], dtype=float)
        return np.zeros(self._action_count, dtype=float)

    def _state_key(self, obs: Any) -> tuple[tuple[str, Any], ...]:
        """Convert observation into a hashable key for tabular lookup."""

        obs_inputs = self._adapter.observation_to_inputs(obs)
        return tuple(sorted(obs_inputs.items()))

    def reset_episode(self) -> dict[str, Any]:
        """Reset env and cache initial observation/input state."""

        result = self._adapter.reset_bridge()
        self._obs = result["obs"]
        self._inputs = result["inputs"]
        return result

    def _initialize_q_values(self, obs: Any) -> tuple[tuple[str, Any], ...]:
        """Ensure an observation has an initialized q-table row."""

        state_key = self._state_key(obs)
        _ = self._q_values[state_key]
        return state_key

    def _action_index(self, action: Any) -> int | None:
        """Resolve an action value to its q-row index when possible."""

        if isinstance(action, (int, np.integer)) and self._action_count is not None:
            index = int(action)
            if 0 <= index < self._action_count:
                return index

        candidate_actions = self._candidate_actions()
        if not candidate_actions:
            return None

        for index, candidate in enumerate(candidate_actions):
            if candidate == action:
                return index

        return None

    def _candidate_actions(self) -> list[Any]:
        """Enumerate candidate actions if action space is finite/flat."""

        action_spec = self._adapter.get_action_spec()
        values = action_spec.get("values")

        if isinstance(values, list) and all(not isinstance(item, list) for item in values):
            return values
        return []

    def _sample_action(self) -> Any:
        """Sample one random action from adapter action space."""

        action_space = self._adapter.action_space
        if action_space is None:
            raise RuntimeError("Adapter action_space is not initialized.")
        return action_space.sample()

    def _select_q_action(self, obs: Any) -> Any:
        """Select action using epsilon-greedy q-table policy."""

        state_key = self._initialize_q_values(obs)

        if self._epsilon > 0.1:
            self._epsilon -= self._epsilon_decay

        candidate_actions = self._candidate_actions()
        if not candidate_actions:
            return self._sample_action()

        if self._rng.random() < self._epsilon:
            return self._sample_action()

        q_row = self._q_values[state_key]
        if len(q_row) != len(candidate_actions):
            return self._sample_action()

        best_index = int(np.argmax(q_row))
        return candidate_actions[best_index]

    def _score_predicted_action(
        self,
        action_inputs: dict[str, Any],
        predictions: dict[str, Any],
    ) -> float:
        """Score alignment between candidate action and Brain predictions."""

        score = 0.0

        for key, value in action_inputs.items():
            if key not in predictions or predictions[key] is None:
                continue

            prediction = predictions[key]
            confidence = predictions.get(f"{key}.conf", 1.0)
            weight = float(confidence) if isinstance(confidence, (int, float)) else 1.0

            if isinstance(prediction, (int, float)) and isinstance(value, (int, float)):
                score -= abs(float(prediction) - float(value)) * weight
            else:
                score += weight if prediction == value else 0.0

        return score

    def _extract_action_from_brain_outputs(self, brain_outputs: dict[str, Any]) -> Any | None:
        """Extract direct action hint from Brain.step outputs when available."""

        for _name, payload in brain_outputs.items():
            if isinstance(payload, dict) and "action" in payload:
                return payload["action"]

        return None

    def _select_brain_action_from_outputs(
        self,
        obs: Any,
        brain_outputs: dict[str, Any] | None,
    ) -> Any:
        """Select Brain action hint, then prediction-scored action, then fallback."""

        if isinstance(brain_outputs, dict):
            action_hint = self._extract_action_from_brain_outputs(brain_outputs)
            if action_hint is not None:
                return action_hint

        candidate_actions = self._candidate_actions()
        if not candidate_actions:
            return self._sample_action()

        predictions = self._brain.prediction()
        if not isinstance(predictions, dict):
            return self._select_q_action(obs)

        action_predictions = {
            key: value for key, value in predictions.items() if key.startswith("action")
        }
        if not action_predictions:
            return self._select_q_action(obs)

        return max(
            candidate_actions,
            key=lambda action: self._score_predicted_action(
                self._adapter.action_to_inputs(action),
                action_predictions,
            ),
        )

    def select_action(self, obs: Any, brain_outputs: dict[str, Any] | None = None) -> Any:
        """Select action using currently configured policy mode."""

        if self._policy_mode == "brain":
            return self._select_brain_action_from_outputs(obs, brain_outputs)

        if self._policy_mode == "ppo":
            return self._select_ppo_action(obs)

        return self._select_q_action(obs)

    def step(self, learn: bool = True) -> dict[str, Any]:
        """Run one full agent-controlled timestep."""

        if self._obs is None:
            self.reset_episode()

        current_obs = self._obs
        current_inputs = self._inputs

        if current_obs is None:
            raise RuntimeError("Agent observation state was not initialized.")

        brain_outputs = self._brain.step(current_inputs, learn=learn)
        action = self.select_action(current_obs, brain_outputs=brain_outputs)
        result = self._adapter.step_bridge(action)

        next_obs = result["obs"]
        reward = result["reward"]

        self.update(
            current_obs,
            action,
            reward,
            next_obs,
            done=result["terminated"] or result["truncated"],
        )

        self._obs = next_obs
        self._inputs = result["inputs"]

        return {
            "obs": current_obs,
            "inputs": current_inputs,
            "action": action,
            "next_obs": next_obs,
            "next_inputs": result["inputs"],
            "reward": reward,
            "terminated": result["terminated"],
            "truncated": result["truncated"],
            "info": result["info"],
        }

    def update(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool = False,
    ) -> None:
        """Update policy state from one transition."""

        if self._policy_mode == "brain":
            self._brain.rl_policy_update()
            return

        if self._policy_mode == "ppo":
            return

        if self._action_count is None:
            return

        action_index = self._action_index(action)
        if action_index is None:
            return

        state_key = self._initialize_q_values(obs)
        next_state_key = self._initialize_q_values(next_obs)

        current_q = float(self._q_values[state_key][action_index])
        next_best_q = 0.0 if done else float(np.max(self._q_values[next_state_key]))

        td_target = float(reward) + (self._discount_factor * next_best_q)
        td_error = td_target - current_q

        self._q_values[state_key][action_index] = current_q + (self._learning_rate * td_error)
        self._training_error.append(abs(td_error))

    def train(self) -> None:
        """Run configured training episodes using Agent-owned stepping."""

        for _episode in range(self._training_episodes):
            self.reset_episode()
            done = False

            while not done:
                transition = self.step(learn=True)
                done = transition["terminated"] or transition["truncated"]
