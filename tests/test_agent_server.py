"""Tests for AgentWebSocketServer visualization payload helpers."""

import pytest

pytest.importorskip("websockets")

from psu_capstone.agent_layer.agent_server import AgentWebSocketServer


class _StubAdapter:
    max_steps_per_episode = 100


class _StubAgent:
    def __init__(self) -> None:
        self._training_episodes = 10
        self._adapter = _StubAdapter()


def test_build_trading_visualization_returns_none_for_non_trading_obs() -> None:
    server = AgentWebSocketServer(agent=_StubAgent())

    payload = server._build_trading_visualization(
        obs={"x": 1.0},
        action=1,
        reward=0.1,
        total_reward=0.3,
        brain_outputs={},
        step=2,
    )

    assert payload is None


def test_build_trading_visualization_computes_good_buy_alignment() -> None:
    server = AgentWebSocketServer(agent=_StubAgent())
    obs = {
        "open": 100.0,
        "high": 104.0,
        "low": 99.5,
        "close": 103.0,
        "volume": 250000.0,
    }
    brain_outputs = {"action_output": {"action": 1, "confidence": 0.8}}

    payload = server._build_trading_visualization(
        obs=obs,
        action=1,
        reward=1.5,
        total_reward=3.5,
        brain_outputs=brain_outputs,
        step=5,
    )

    assert payload is not None
    assert payload["action"]["label"] == "buy"
    assert payload["alignment_score"] == 1.0
    assert payload["quality"] == "good"
    assert payload["brain_confidence"] == 0.8
    assert payload["ohlcv"]["close"] == 103.0
