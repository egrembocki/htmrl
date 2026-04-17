# Test Suite: TS 21 (CartPole Brain Policy Training)
"""Smoke tests for generic env brain-policy training loop."""

from __future__ import annotations

from psu_capstone.agent_layer.brain_training_helper import (
    EnvTrainingConfig,
    train_env_policy,
)


# Test Type: system test
def test_train_env_policy_returns_metrics() -> None:
    """Training helper should run and return structured metrics."""

    metrics = train_env_policy(
        EnvTrainingConfig(
            env_id="CartPole-v1",
            episodes=2,
            max_steps_per_episode=10,
            input_size=64,
            cells_per_column=4,
            resolution=0.05,
            seed=11,
            render_mode=None,
        )
    )

    assert metrics["episodes"] == 2
    assert len(metrics["episode_rewards"]) == 2
    assert len(metrics["episode_steps"]) == 2
    assert all(step <= 10 for step in metrics["episode_steps"])
    assert isinstance(metrics["mean_reward"], float)
    assert isinstance(metrics["best_reward"], float)
