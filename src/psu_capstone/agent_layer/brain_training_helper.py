"""Generic helpers for building and training Brains against Gym environments.

This module focuses on one task: set up a Brain for any Gymnasium environment,
whether the caller passes an env id string, a pre-built gym.Env instance, or
an existing EnvAdapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym

from psu_capstone.agent_layer.agent_runtime import (
    AgentRuntimeConfig,
    build_brain_for_adapter,
    run_local_session,
)
from psu_capstone.agent_layer.brain import Brain
from psu_capstone.environment.env_adapter import EnvAdapter


@dataclass(frozen=True)
class EnvTrainingConfig:
    """Configuration for training an Agent against any Gymnasium environment.

    Attributes:
        env_id: Gymnasium environment id passed to ``gym.make`` via ``EnvAdapter``.
        episodes: Number of episodes to run.
        max_steps_per_episode: Hard cap on timesteps per episode.
        input_size: SDR width for each input field.
        cells_per_column: Number of cells per HTM column.
        resolution: RDSE resolution used for continuous observations.
        seed: Base seed used when constructing encoder parameters.
        render_mode: Optional Gymnasium render mode forwarded at env creation.
        policy_mode: Active action-selection strategy used by ``Agent``.
    """

    env_id: str = "CartPole-v1"
    episodes: int = 200
    max_steps_per_episode: int = 150
    input_size: int = 256
    cells_per_column: int = 8
    resolution: float = 0.001
    seed: int = 5
    render_mode: str | None = "human"
    policy_mode: Literal["q_table", "brain", "ppo"] = "brain"


def as_env_brain_config(
    config: EnvTrainingConfig,
) -> AgentRuntimeConfig:
    """Translate a training config into the unified runtime config."""

    seed = config.seed

    return AgentRuntimeConfig(
        env_id=config.env_id,
        episodes=config.episodes,
        max_steps_per_episode=config.max_steps_per_episode,
        input_size=config.input_size,
        cells_per_column=config.cells_per_column,
        resolution=config.resolution,
        seed=seed,
        policy_mode=config.policy_mode,
        render_mode=config.render_mode,
    )


def setup_env_adapter(
    gym_env: str | gym.Env[Any, Any] | EnvAdapter,
    config: EnvTrainingConfig | None = None,
) -> EnvAdapter:
    """Build or normalize an EnvAdapter from any supported Gym input.

    Args:
        gym_env: Gym env id string, pre-built ``gym.Env``, or existing ``EnvAdapter``.
        config: Optional training configuration used for render mode when
            creating an adapter from an env id string.

    Returns:
        Normalized ``EnvAdapter`` instance.
    """

    if isinstance(gym_env, EnvAdapter):
        return gym_env

    cfg = config or EnvTrainingConfig()
    if isinstance(gym_env, str):
        adapter_kwargs: dict[str, Any] = {}
        if cfg.render_mode is not None:
            adapter_kwargs["render_mode"] = cfg.render_mode
        return EnvAdapter(gym_env, **adapter_kwargs)

    return EnvAdapter(gym_env)


def setup_env_brain(
    gym_env: str | gym.Env[Any, Any] | EnvAdapter,
    config: EnvTrainingConfig | None = None,
) -> tuple[Brain, EnvAdapter]:
    """Set up a Brain for any Gym environment shape.

    Args:
        gym_env: Gym env id string, pre-built ``gym.Env``, or existing ``EnvAdapter``.
        config: Optional training/encoder configuration.

    Returns:
        Tuple of ``(brain, adapter)`` ready to use with ``Agent``.
    """

    cfg = config or EnvTrainingConfig()
    adapter = setup_env_adapter(gym_env, cfg)
    brain = build_brain_for_adapter(adapter, as_env_brain_config(cfg))
    return brain, adapter


def build_env_brain(
    gym_env: str | gym.Env[Any, Any] | EnvAdapter,
    config: EnvTrainingConfig | None = None,
) -> Brain:
    """Build only the Brain for any Gym environment input."""

    brain, _ = setup_env_brain(gym_env, config)
    return brain


def train_env_policy(
    config: EnvTrainingConfig | None = None,
) -> dict[str, Any]:
    """Run episodes for any Gymnasium environment using the configured policy mode.

    Args:
        config: Optional environment training configuration. Defaults to
            ``EnvTrainingConfig()`` if not provided.

    Returns:
        Dictionary of training metrics and per-episode summaries.
    """

    cfg = config or EnvTrainingConfig()
    return run_local_session(as_env_brain_config(cfg))


def main() -> None:
    """Run a default training session and print metrics."""

    metrics = train_env_policy(EnvTrainingConfig(render_mode=None))
    print("Generic gym brain-policy training summary")
    print(f"Episodes: {metrics['episodes']}")
    print(f"Mean reward: {metrics['mean_reward']:.3f}")
    print(f"Best reward: {metrics['best_reward']:.3f}")


if __name__ == "__main__":
    main()
