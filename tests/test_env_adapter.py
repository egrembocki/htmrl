"""Tests for EnvAdapter construction and FinGym integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from psu_capstone.environment.env_adapter import EnvAdapter
from psu_capstone.environment.fin_gym import FinGym


def test_env_adapter_accepts_instantiated_fingym() -> None:
    """EnvAdapter should wrap a pre-built FinGym instance directly."""

    frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "target": [10.0, 11.0, 12.0],
        }
    )
    env = FinGym(data_source=frame, target_column="target")

    adapter = EnvAdapter(env)
    reset_payload = adapter.reset()

    assert isinstance(reset_payload["obs"], np.ndarray)
    assert isinstance(reset_payload["inputs"], dict)

    step_payload = adapter.step(1)
    assert isinstance(step_payload["reward"], float)
    assert "terminated" in step_payload
    assert "truncated" in step_payload


def test_env_adapter_accepts_make_kwargs() -> None:
    """EnvAdapter should forward kwargs when constructing Gym env from id."""

    adapter = EnvAdapter("CartPole-v1", render_mode="rgb_array")

    reset_payload = adapter.reset()
    assert isinstance(reset_payload["inputs"], dict)


def test_env_adapter_rejects_kwargs_with_env_instance() -> None:
    """Passing gym kwargs with an env instance should raise ValueError."""

    frame = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "target": [10.0, 11.0, 12.0],
        }
    )
    env = FinGym(data_source=frame, target_column="target")

    try:
        _ = EnvAdapter(env, render_mode="human")
        assert False, "Expected ValueError when kwargs are passed with env instance"
    except ValueError as exc:
        assert "gym_kwargs" in str(exc)
