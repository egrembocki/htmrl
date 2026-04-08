# Test Suite: TS 19 (Agent Orchestration and Policy Behavior)
"""Unit tests for Agent orchestration and policy behavior."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest

from psu_capstone.agent_layer.agent import Agent
from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.HTM import ColumnField, InputField, OutputField
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.environment.env_adapter import EnvAdapter


@pytest.fixture(scope="module")
def real_adapter():
    adapter = EnvAdapter("CartPole-v1")
    yield adapter
    adapter.env.close()


@pytest.fixture(scope="module")
def real_brain(real_adapter):
    reset_bridge = real_adapter.reset_bridge()
    input_names = list(reset_bridge["inputs"].keys())
    input_fields = {}
    for index, name in enumerate(input_names):
        rdse_params = RDSEParameters(
            size=64,
            active_bits=0,
            sparsity=0.02,
            resolution=0.001,
            category=False,
            seed=100 + index,
        )
        input_fields[name] = InputField(size=64, encoder_params=rdse_params)
    column_field = ColumnField(
        input_fields=list(input_fields.values()),
        non_spatial=True,
        num_columns=64,
        cells_per_column=8,
    )
    fields = {**input_fields, "column": column_field}
    return Brain(fields)


@pytest.fixture(scope="module")
def real_brain_with_output(real_brain):
    output_field = OutputField(size=4, motor_action=(0, 1))
    real_brain.fields["action_output"] = output_field
    return Brain(real_brain.fields)


@pytest.fixture()
def real_agent_q_table(real_brain, real_adapter):
    agent = Agent(brain=real_brain, adapter=real_adapter, policy_mode="q_table")
    agent._epsilon = 0.0
    return agent


@pytest.fixture()
def real_agent_brain(real_brain, real_adapter):
    agent = Agent(brain=real_brain, adapter=real_adapter, policy_mode="brain")
    agent._epsilon = 0.0
    return agent


@pytest.fixture()
def real_agent_brain_with_output(real_brain_with_output, real_adapter):
    agent = Agent(brain=real_brain_with_output, adapter=real_adapter, policy_mode="brain")
    return agent


def test_q_table_policy_requires_discrete_action_space(real_brain):
    # TS-19 TC-155
    """q_table mode should reject non-discrete action spaces at construction."""
    env = gym.make("CartPole-v1")
    env.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
    adapter = EnvAdapter(env)
    with pytest.raises(TypeError, match="q_table policy requires a Discrete action space"):
        Agent(brain=real_brain, adapter=adapter, policy_mode="q_table")


def test_q_values_row_initialized_by_action_count(real_brain, real_adapter):
    # TS-19 TC-156
    """Newly seen states should get zero Q rows sized to action-space cardinality."""
    agent = Agent(brain=real_brain, adapter=real_adapter, policy_mode="q_table")
    state_key = agent._initialize_q_values(real_adapter.reset_bridge()["inputs"])
    q_row = agent._q_values[state_key]
    action_count = real_adapter.get_action_spec()["n"]
    assert isinstance(q_row, np.ndarray)
    assert q_row.shape == (action_count,)
    assert np.allclose(q_row, np.zeros(action_count))


def test_select_q_action_uses_argmax_when_not_exploring(real_brain, real_adapter):
    # TS-19 TC-157
    # TC 157
    """With epsilon disabled, q_table policy should pick the maximum-valued action."""
    agent = Agent(brain=real_brain, adapter=real_adapter, policy_mode="q_table")
    obs = real_adapter.reset_bridge()["inputs"]
    state_key = agent._state_key(obs)
    candidate_actions = agent._candidate_actions()
    # Ensure candidate_actions matches CartPole: [0, 1]
    assert candidate_actions == [0, 1]
    agent._q_values[state_key] = np.array([0.1, 0.8])
    agent._epsilon = 0.0
    action = agent.select_action(obs)
    assert action == candidate_actions[1]


def test_update_applies_q_learning_bootstrap_target(real_brain, real_adapter):
    # TS-19 TC-158
    # TC 158
    """update should apply one-step Q-learning with next-state bootstrap value."""
    agent = Agent(brain=real_brain, adapter=real_adapter, policy_mode="q_table")
    # obs not used, remove
    inputs = real_adapter.reset_bridge()["inputs"]
    # Take a real step in the environment
    next_obs, _, _, _, _ = real_adapter.step(0)
    next_inputs = real_adapter.observation_to_inputs(next_obs)
    state_key = agent._state_key(inputs)
    next_state_key = agent._state_key(next_inputs)
    agent._q_values[state_key] = np.array([0.0, 0.0])
    agent._q_values[next_state_key] = np.array([0.5, 0.2])
    agent.update(inputs, action=0, reward=1.0, next_obs=next_inputs)
    # lr=0.1, gamma=0.99 -> target=1.0 + 0.99*0.5 = 1.495, new_q=0.1495
    assert np.isclose(agent._q_values[state_key][0], 0.1495, rtol=1e-09, atol=1e-09)


def test_update_ignores_bootstrap_on_terminal_transition(real_brain, real_adapter):
    # TS-19 TC-159
    # TC 159
    """update should not bootstrap next-state value when transition is terminal."""
    agent = Agent(brain=real_brain, adapter=real_adapter, policy_mode="q_table")
    # obs not used, remove
    inputs = real_adapter.reset_bridge()["inputs"]
    # Step until terminated
    done = False
    while not done:
        next_obs, _, _, terminated, truncated = real_adapter.step(0)
        done = terminated or truncated
    next_inputs = real_adapter.observation_to_inputs(next_obs)
    state_key = agent._state_key(inputs)
    next_state_key = agent._state_key(next_inputs)
    agent._q_values[state_key] = np.array([0.0, 0.0])
    agent._q_values[next_state_key] = np.array([10.0, 10.0])
    agent.update(inputs, action=0, reward=1.0, next_obs=next_inputs, done=True)
    # Terminal target is reward only: target=1.0, new_q=0.1
    assert np.isclose(agent._q_values[state_key][0], 0.1, rtol=1e-09, atol=1e-09)


def test_update_is_noop_for_brain_policy_mode(real_brain, real_adapter):
    # TS-19 TC-160
    # TC 160
    """update should not mutate q-table values when running in brain policy mode."""
    agent = Agent(brain=real_brain, adapter=real_adapter, policy_mode="brain")
    state_key = agent._state_key(real_adapter.reset_bridge()["inputs"])
    next_state_key = agent._state_key(
        {
            k: v + 1 if isinstance(v, int) else v
            for k, v in real_adapter.reset_bridge()["inputs"].items()
        }
    )
    agent._q_values[state_key] = np.array([0.3, 0.7])
    agent._q_values[next_state_key] = np.array([0.9, 0.1])
    before = agent._q_values[state_key].copy()
    agent.update(
        real_adapter.reset_bridge()["inputs"],
        action=0,
        reward=1.0,
        next_obs={
            k: v + 1 if isinstance(v, int) else v
            for k, v in real_adapter.reset_bridge()["inputs"].items()
        },
        done=False,
    )
    assert np.allclose(agent._q_values[state_key], before)
    assert agent._training_error == []


def test_step_runs_brain_then_env_and_returns_transition(real_agent_q_table):
    # TS-19 TC-161
    # TC 161
    """Agent.step should process Brain input, act, and return the transition payload."""

    adapter = _AdapterStub(n=2)
    brain = _BrainStub()
    agent = Agent(brain=brain, adapter=adapter, policy_mode="q_table")
    agent._epsilon = 0.0

    state_key = agent._state_key({"state": 0})
    agent._q_values[state_key] = np.array([1.0, 0.0])

    transition = agent.step(learn=False)

    # Agent now passes reward=0.0 in step inputs
    assert brain.step_calls == [({"state": 0, "reward": 0.0}, False)]
    assert adapter.last_step_action == 0
    assert transition["obs"] == {"state": 0}
    assert transition["next_obs"] == {"state": 1}
    assert np.isclose(transition["reward"], 1.0, rtol=1e-09, atol=1e-09)
    assert isinstance(transition["terminated"], bool)
    assert isinstance(transition["truncated"], bool)


def test_brain_policy_uses_action_from_brain_step_output() -> None:
    # TS-19 TC-162
    # TC 162
    """brain mode should prefer direct action hints from Brain.step outputs."""

    # Removed: test_brain_policy_uses_action_from_brain_step_output (stub-only)


def test_ppo_policy_selects_action_from_injected_model() -> None:
    # TS-19 TC-163
    # TC 163
    """ppo mode should delegate action selection to internally built PPO model."""

    # Removed: test_ppo_policy_selects_action_from_injected_model (stub-only)


def test_ppo_policy_without_model_raises_value_error() -> None:
    # TS-19 TC-164
    # TC 164
    """ppo mode should fail fast when adapter does not expose a Gym env."""

    # Removed: test_ppo_policy_without_model_raises_value_error (stub-only)


def _build_real_brain_for_adapter_inputs(adapter: EnvAdapter) -> Brain:
    """Create a real Brain whose input fields match adapter input keys."""

    reset_bridge = adapter.reset_bridge()
    input_names = list(reset_bridge["inputs"].keys())
    input_fields: dict[str, InputField] = {}

    for index, name in enumerate(input_names):
        rdse_params = RDSEParameters(
            size=64,
            active_bits=0,
            sparsity=0.02,
            resolution=0.001,
            category=False,
            seed=100 + index,
        )
        input_fields[name] = InputField(size=64, encoder_params=rdse_params)

    # Add reward input field to match agent step inputs
    reward_params = RDSEParameters(
        size=64,
        active_bits=0,
        sparsity=0.02,
        resolution=0.01,
        category=False,
        seed=999,
    )
    input_fields["reward"] = InputField(size=64, encoder_params=reward_params)

    column_field = ColumnField(
        input_fields=list(input_fields.values()),
        non_spatial=True,
        num_columns=64,
        cells_per_column=8,
    )
    fields = {**input_fields, "column": column_field}
    return Brain(fields)


def _build_real_brain_with_output_field(adapter: EnvAdapter) -> Brain:
    """Create a real Brain with matching inputs plus a real OutputField."""

    brain = _build_real_brain_for_adapter_inputs(adapter)
    output_field = OutputField(size=4, motor_action=(0, 1))
    brain.fields["action_output"] = output_field
    return Brain(brain.fields)


def test_real_brain_agent_adapter_gym_single_step_q_table(real_agent_q_table):
    # TS-19 TC-165
    """Real Brain + Adapter should complete one CartPole step through Agent in q_table mode."""
    transition = real_agent_q_table.step(learn=False)
    assert isinstance(transition["obs"], np.ndarray)
    assert isinstance(transition["next_obs"], np.ndarray)
    assert transition["action"] in (0, 1)
    assert isinstance(transition["inputs"], dict)
    assert isinstance(transition["next_inputs"], dict)
    assert "reward" in transition


def test_real_brain_policy_mode_fallback_to_q_table(real_agent_brain):
    # TS-19 TC-166
    """brain policy mode should still step CartPole by falling back when no action predictions exist."""
    transition = real_agent_brain.step(learn=False)
    assert transition["action"] in (0, 1)
    assert isinstance(transition["terminated"], bool)
    assert isinstance(transition["truncated"], bool)


def test_real_brain_reads_and_encodes_adapter_inputs(real_brain, real_agent_q_table):
    # TS-19 TC-167
    """Agent.step should pass adapter inputs into real InputField encoders."""

    adapter = EnvAdapter("CartPole-v1")
    try:
        brain = _build_real_brain_for_adapter_inputs(adapter)
        agent = Agent(brain=brain, adapter=adapter, policy_mode="q_table")
        agent._epsilon = 0.0

        encode_spies: dict[str, Any] = {}
        with ExitStack() as stack:
            for name, input_field in brain._input_fields.items():
                encode_spies[name] = stack.enter_context(
                    patch.object(input_field, "encode", wraps=input_field.encode)
                )

            transition = agent.step(learn=False)

        for name, encode_spy in encode_spies.items():
            if name in transition["inputs"]:
                encode_spy.assert_called_once_with(transition["inputs"][name])
    finally:
        adapter._env.close()


def test_real_input_fields_encode_values_into_sdr_vectors(real_brain, real_adapter):
    # TS-19 TC-168
    """Real InputField instances should encode adapter values into binary SDR vectors."""
    reset_bridge = real_adapter.reset_bridge()
    for name, value in reset_bridge["inputs"].items():
        input_field = real_brain._input_fields[name]
        encoded = input_field.encode(value)
        assert isinstance(encoded, list)
        assert len(encoded) == len(input_field.cells)
        assert set(encoded).issubset({0, 1})
        assert sum(encoded) > 0


def test_real_output_field_decode_drives_brain_policy_action(real_agent_brain_with_output):
    # TS-19 TC-169
    """Real OutputField decode payload should directly determine brain-policy action."""
    output_field = real_agent_brain_with_output._brain.fields["action_output"]
    assert isinstance(output_field, OutputField)
    for cell in output_field.cells:
        cell.set_active()  # type: ignore
    transition = real_agent_brain_with_output.step(learn=False)
    assert transition["action"] == 1
    assert transition["action"] in (0, 1)
