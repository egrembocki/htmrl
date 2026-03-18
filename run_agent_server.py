#!/usr/bin/env python3
"""Run the agent WebSocket server for real-time visualization."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from psu_capstone.agent_layer.agent import Agent  # noqa: E402
from psu_capstone.agent_layer.agent_server import AgentWebSocketServer  # noqa: E402
from psu_capstone.environment.env_adapter import EnvAdapter  # noqa: E402
from psu_capstone.log import get_logger  # noqa: E402


def build_agent(env_id: str, ppo_timesteps: int = 50_000) -> Agent:
    """Build and (for PPO) pre-train an agent for the specified environment."""
    try:
        from psu_capstone.agent_layer.cartpole_brain_training import (
            CartPoleTrainingConfig,
            build_cartpole_brain,
        )

        config = CartPoleTrainingConfig(episodes=1000, max_steps_per_episode=150)
        adapter = EnvAdapter(env_id)
        brain = build_cartpole_brain(adapter, config)
        agent = Agent(
            brain=brain,
            adapter=adapter,
            episodes=config.episodes,
            policy_mode="ppo",
        )
        if ppo_timesteps > 0:
            logger = get_logger(None)
            logger.info("Pre-training PPO for %d timesteps...", ppo_timesteps)
            agent.train_ppo(total_timesteps=ppo_timesteps)
            logger.info("PPO pre-training complete.")
        return agent
    except ImportError:
        raise ImportError(
            "CartPole brain training support not available. "
            "Ensure psu_capstone is properly installed."
        )


async def main(args: argparse.Namespace) -> None:
    """Start the agent WebSocket server."""
    logger = get_logger(None)

    try:
        logger.info(f"Building {args.env} agent...")
        agent = build_agent(args.env, ppo_timesteps=args.ppo_timesteps)

        logger.info(f"Starting WebSocket server on ws://{args.host}:{args.port}")
        server = AgentWebSocketServer(
            agent,
            host=args.host,
            port=args.port,
            switch_after_episodes=args.switch_episodes,
            switch_min_reward=args.switch_reward,
        )

        logger.info(f"Server is running. Connect your web client to ws://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop the server.")

        await server.start()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


def main_sync(args: argparse.Namespace) -> None:
    """Wrapper to run the async main function."""
    asyncio.run(main(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an agent WebSocket server for real-time visualization."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gym environment ID to use (default: CartPole-v1)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="WebSocket server bind address (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server bind port (default: 8765)",
    )
    parser.add_argument(
        "--ppo-timesteps",
        type=int,
        default=50_000,
        help="Timesteps to pre-train PPO before serving (default: 50000, 0 to skip)",
    )
    parser.add_argument(
        "--switch-episodes",
        type=int,
        default=100,
        help="Switch from PPO to brain after this many episodes meet the reward threshold (default: 100)",
    )
    parser.add_argument(
        "--switch-reward",
        type=float,
        default=100.0,
        help="Mean reward threshold required to trigger policy switch to brain (default: 100.0)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()
    main_sync(args)
