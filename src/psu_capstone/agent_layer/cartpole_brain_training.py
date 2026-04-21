"""CartPole compatibility wrappers over generic Trainer runtime flows.

This module preserves older CartPole entrypoints while delegating to the
generic training helpers driven by ``Trainer.build_brain_for_env``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from psu_capstone.agent_layer.brain_training_helper import EnvTrainingConfig, train_env_policy
from psu_capstone.agent_layer.pullin.pullin_brain import Brain
from psu_capstone.agent_layer.train import Trainer
from psu_capstone.environment.env_adapter import EnvAdapter


@dataclass(frozen=True)
class CartPoleTrainingConfig:
    """Configuration for CartPole brain-policy training.

    Kept for backward compatibility. Internally converted to generic helper
    configuration so all environment training paths share one implementation.
    """

    episodes: int = 200
    max_steps_per_episode: int = 150
    input_size: int = 128
    cells_per_column: int = 8
    resolution: float = 0.001
    rdse_seed: int = 5


def build_cartpole_brain(adapter: EnvAdapter, config: CartPoleTrainingConfig) -> Brain:
    """Build a CartPole Brain via the generic Trainer environment builder."""

    trainer = Trainer(Brain({}))
    return trainer.build_brain_for_env(adapter, config)


def _to_env_config(config: CartPoleTrainingConfig) -> EnvTrainingConfig:
    return EnvTrainingConfig(
        env_id="CartPole-v1",
        episodes=config.episodes,
        max_steps_per_episode=config.max_steps_per_episode,
        input_size=config.input_size,
        cells_per_column=config.cells_per_column,
        resolution=config.resolution,
        seed=config.rdse_seed,
        render_mode=None,
    )


def train_cartpole_brain_policy(
    config: CartPoleTrainingConfig | None = None,
) -> dict[str, Any]:
    """Run CartPole brain-policy training through generic env helpers."""

    cfg = config or CartPoleTrainingConfig()
    return train_env_policy(_to_env_config(cfg))


def main() -> None:
    """Run a default CartPole brain-policy training session and print metrics."""

    metrics = train_cartpole_brain_policy()
    print("CartPole brain-policy training summary")
    print(f"Episodes: {metrics['episodes']}")
    print(f"Mean reward: {metrics['mean_reward']:.3f}")
    print(f"Best reward: {metrics['best_reward']:.3f}")


if __name__ == "__main__":
    main()
