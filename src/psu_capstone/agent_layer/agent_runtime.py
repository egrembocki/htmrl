"""Unified runtime flow for Agent, Brain, server, and local env execution.

This module is the single orchestration point for:
- choosing the correct adapter for a requested environment
- building a Brain that matches that environment's observation/action schema
- constructing the Agent
- running a websocket server for frontend-driven sessions
- running a local Gym session for render-based testing
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Literal, TypedDict, cast

from psu_capstone.agent_layer.agent import Agent
from psu_capstone.agent_layer.agent_server import AgentWebSocketServer
from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.train import Trainer
from psu_capstone.encoder_layer.category_encoder import CategoryParameters
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.environment.env_adapter import EnvAdapter
from psu_capstone.environment.frontend_env_adapter import FrontendEnvAdapter
from psu_capstone.log import get_logger

PolicyMode = Literal["q_table", "brain", "ppo"]


class FrontendEnvSpec(TypedDict):
    """Frontend-managed environment schema used by the websocket runtime."""

    observation_labels: list[str]
    action_count: int
    initial_observation: dict[str, float]
    max_steps_per_episode: int


FRONTEND_ENV_SPECS: dict[str, FrontendEnvSpec] = {
    "CartPole-v1": {
        "observation_labels": [
            "Cart Position",
            "Cart Velocity",
            "Pole Angle",
            "Pole Angular Velocity",
        ],
        "action_count": 2,
        "initial_observation": {
            "Cart Position": 0.0,
            "Cart Velocity": 0.0,
            "Pole Angle": 0.0,
            "Pole Angular Velocity": 0.0,
        },
        "max_steps_per_episode": 200,
    },
    "FrozenLake": {
        "observation_labels": ["Agent X", "Agent Y", "Goal X", "Goal Y"],
        "action_count": 4,
        "initial_observation": {
            "Agent X": 0.0,
            "Agent Y": 0.0,
            "Goal X": 4.0,
            "Goal Y": 4.0,
        },
        "max_steps_per_episode": 100,
    },
    "FrozenLake-v1": {
        "observation_labels": ["Agent X", "Agent Y", "Goal X", "Goal Y"],
        "action_count": 4,
        "initial_observation": {
            "Agent X": 0.0,
            "Agent Y": 0.0,
            "Goal X": 4.0,
            "Goal Y": 4.0,
        },
        "max_steps_per_episode": 100,
    },
    "MountainCar-v0": {
        "observation_labels": ["Position", "Velocity"],
        "action_count": 3,
        "initial_observation": {
            "Position": -0.5,
            "Velocity": 0.0,
        },
        "max_steps_per_episode": 200,
    },
    "Pendulum-v1": {
        "observation_labels": ["cos(theta)", "sin(theta)", "theta_dot"],
        "action_count": 3,
        "initial_observation": {
            "cos(theta)": 1.0,
            "sin(theta)": 0.0,
            "theta_dot": 0.0,
        },
        "max_steps_per_episode": 200,
    },
    "GridWorld": {
        "observation_labels": ["Agent X", "Agent Y", "Goal X", "Goal Y"],
        "action_count": 4,
        "initial_observation": {
            "Agent X": 0.0,
            "Agent Y": 0.0,
            "Goal X": 4.0,
            "Goal Y": 4.0,
        },
        "max_steps_per_episode": 100,
    },
}


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Configuration shared by server and local execution paths.

    Attributes:
        env_id: Requested environment identifier.
        policy_mode: Agent policy mode.
        episodes: Number of episodes to run in local/train loops.
        max_steps_per_episode: Hard cap on steps per episode.
        input_size: SDR width for each input field.
        cells_per_column: Number of HTM cells per column.
        resolution: RDSE resolution used for continuous inputs.
        seed: Base seed for encoder parameter generation.
        render_mode: Optional Gymnasium render mode for local runs.
        host: WebSocket bind host for server mode.
        port: WebSocket bind port for server mode.
        ppo_pretrain_timesteps: PPO warm-up timesteps before serving/running.
    """

    env_id: str = "CartPole-v1"
    policy_mode: PolicyMode = "brain"
    episodes: int = 200
    max_steps_per_episode: int = 200
    input_size: int = 256
    cells_per_column: int = 8
    resolution: float = 0.001
    seed: int = 5
    render_mode: str | None = None
    host: str = "localhost"
    port: int = 8765
    ppo_pretrain_timesteps: int = 50_000


@dataclass
class AgentRuntime:
    """Bundle of built runtime objects used by server and local flows."""

    config: AgentRuntimeConfig
    adapter: Any
    brain: Brain
    agent: Agent
    is_frontend_env: bool

    def close(self) -> None:
        """Close the underlying environment if one exists."""

        env = getattr(self.adapter, "_env", None)
        if env is not None and hasattr(env, "close"):
            env.close()


def build_adapter(config: AgentRuntimeConfig, *, allow_frontend_env: bool) -> tuple[Any, bool]:
    """Build the correct adapter for the requested environment id."""

    use_frontend_adapter = (
        allow_frontend_env and config.env_id in FRONTEND_ENV_SPECS and config.policy_mode != "ppo"
    )

    if use_frontend_adapter:
        spec = cast(FrontendEnvSpec, FRONTEND_ENV_SPECS[config.env_id])
        adapter = FrontendEnvAdapter(
            env_name=config.env_id,
            observation_labels=list(spec["observation_labels"]),
            action_count=spec["action_count"],
            initial_observation=dict(spec["initial_observation"]),
        )
        return adapter, True

    adapter_kwargs: dict[str, Any] = {}
    if config.render_mode is not None:
        adapter_kwargs["render_mode"] = config.render_mode

    try:
        return EnvAdapter(config.env_id, **adapter_kwargs), False
    except Exception as exc:
        if config.env_id in FRONTEND_ENV_SPECS and not allow_frontend_env:
            raise ValueError(
                f"Environment '{config.env_id}' is frontend-driven in this mode. Use server mode for frontend aliases or pass a valid Gym env id."
            ) from exc
        if config.env_id in FRONTEND_ENV_SPECS and config.policy_mode == "ppo":
            raise ValueError(
                f"PPO mode requires a Gym-backed environment. '{config.env_id}' is configured as a frontend env and could not be created via Gym."
            ) from exc
        raise


def build_brain_for_adapter(adapter: Any, config: AgentRuntimeConfig) -> Brain:
    """Build a Brain that matches the adapter's observation/action schema."""

    reset_bridge = adapter.reset_bridge()
    input_names = list(reset_bridge["inputs"].keys())

    obs_spec = adapter.get_observation_spec()
    action_spec = adapter.get_action_spec()
    obs_space_type = obs_spec.get("space_type", "Box")

    trainer = Trainer(Brain({}))

    for index, name in enumerate(input_names):
        if obs_space_type == "Discrete":
            n_categories = int(obs_spec.get("n", 16))
            params: RDSEParameters | CategoryParameters = CategoryParameters(
                size=config.input_size,
                w=3,
                category_list=[str(i) for i in range(n_categories)],
            )
        else:
            params = RDSEParameters(
                size=config.input_size,
                active_bits=0,
                sparsity=0.02,
                resolution=config.resolution,
                category=False,
                seed=config.seed + index,
            )
        trainer.add_input_field(f"{name}_input", config.input_size, params)

    reward_params = RDSEParameters(
        size=config.input_size,
        active_bits=0,
        sparsity=0.02,
        resolution=0.01,
        category=False,
        seed=config.seed + len(input_names),
    )
    trainer.add_input_field("reward_input", config.input_size, reward_params)

    action_space_type = action_spec.get("space_type", "Discrete")
    if action_space_type == "Discrete":
        n_actions = int(action_spec.get("n", 2))
        motor_action: tuple[Any, ...] = tuple(range(n_actions))
    else:
        n_actions = 2
        motor_action = (0, 1)

    trainer.add_output_field("action_output", n_actions, motor_action)
    trainer.add_column_field(
        "column_column",
        num_columns=config.input_size,
        cells_per_column=config.cells_per_column,
    )

    trainer_brain = trainer.main_brain
    remapped_fields: dict[str, Any] = {
        name: trainer_brain.fields[f"{name}_input"] for name in input_names
    }
    remapped_fields["reward"] = trainer_brain.fields["reward_input"]
    remapped_fields["action_output"] = trainer_brain.fields["action_output"]
    remapped_fields["column"] = trainer_brain.fields["column_column"]

    return Brain(remapped_fields)


def build_runtime(config: AgentRuntimeConfig, *, allow_frontend_env: bool) -> AgentRuntime:
    """Build the adapter, Brain, and Agent for a requested environment."""

    adapter, is_frontend_env = build_adapter(config, allow_frontend_env=allow_frontend_env)
    brain = build_brain_for_adapter(adapter, config)
    agent = Agent(
        brain=brain,
        adapter=adapter,
        episodes=config.episodes,
        policy_mode=config.policy_mode,
        # Keep confidence-threshold fallback behavior active in brain mode:
        # brain (>= threshold) -> ppo -> q_table.
        force_brain_mode=False,
    )

    if config.policy_mode == "ppo":
        if not hasattr(adapter, "_env"):
            raise ValueError(
                f"PPO mode requires a Python Gym environment. Environment '{config.env_id}' is frontend-driven."
            )
        logger = get_logger(None)
        logger.info("Pre-training PPO for %d timesteps...", config.ppo_pretrain_timesteps)
        agent.train_ppo(total_timesteps=config.ppo_pretrain_timesteps)
        logger.info("PPO pre-training complete.")

    return AgentRuntime(
        config=config,
        adapter=adapter,
        brain=brain,
        agent=agent,
        is_frontend_env=is_frontend_env,
    )


def build_server(runtime: AgentRuntime) -> AgentWebSocketServer:
    """Create the websocket server for a built runtime."""

    return AgentWebSocketServer(
        runtime.agent,
        host=runtime.config.host,
        port=runtime.config.port,
    )


async def run_server(config: AgentRuntimeConfig) -> None:
    """Build a runtime and start the websocket server."""

    logger = get_logger(None)
    runtime = build_runtime(config, allow_frontend_env=True)
    logger.info("Starting WebSocket server on ws://%s:%s", config.host, config.port)
    logger.info(
        "Server is running. Connect your web client to ws://%s:%s", config.host, config.port
    )
    logger.info("Press Ctrl+C to stop the server.")
    try:
        await build_server(runtime).start()
    finally:
        runtime.close()


def run_local_session(config: AgentRuntimeConfig) -> dict[str, Any]:
    """Run local env stepping for Gym-based testing with the same build flow.

    This path is intended for local testing of the env you pass in. When the
    env supports ``render_mode='human'``, Gymnasium will usually drive the
    pygame/window rendering internally.
    """

    runtime = build_runtime(config, allow_frontend_env=False)
    logger = get_logger(None)
    episode_rewards: list[float] = []
    episode_steps: list[int] = []

    try:
        env = getattr(runtime.adapter, "_env", None)

        for episode_index in range(config.episodes):
            logger.info("Starting local episode %d/%d", episode_index + 1, config.episodes)
            runtime.agent.reset_episode()
            done = False
            total_reward = 0.0
            steps = 0

            if env is not None and config.render_mode is not None and hasattr(env, "render"):
                env.render()

            while not done and steps < config.max_steps_per_episode:
                transition = runtime.agent.step(learn=True)
                total_reward += float(transition["reward"])
                done = bool(transition["terminated"] or transition["truncated"])
                steps += 1

                if env is not None and config.render_mode is not None and hasattr(env, "render"):
                    env.render()

            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            logger.info(
                "Episode %d finished: reward=%s steps=%s",
                episode_index + 1,
                total_reward,
                steps,
            )

        return {
            "env_id": config.env_id,
            "episodes": config.episodes,
            "max_steps_per_episode": config.max_steps_per_episode,
            "episode_rewards": episode_rewards,
            "episode_steps": episode_steps,
            "mean_reward": float(fmean(episode_rewards)) if episode_rewards else 0.0,
            "best_reward": float(max(episode_rewards)) if episode_rewards else 0.0,
        }
    finally:
        runtime.close()
