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


def build_agent(env_id: str) -> Agent:
    """Build an agent for the specified environment."""
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
            policy_mode="brain",
        )
        return agent
    except ImportError:
        raise ImportError(
            "CartPole brain training support not available. "
            "Ensure psu_capstone is properly installed."
        )


async def main(args: argparse.Namespace) -> None:
    """Start the agent WebSocket server."""
    logger = get_logger(None, level=logging.INFO)

    try:
        logger.info(f"Building {args.env} agent...")
        agent = build_agent(args.env)

        logger.info(f"Starting WebSocket server on ws://{args.host}:{args.port}")
        server = AgentWebSocketServer(agent, host=args.host, port=args.port)

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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    args = parser.parse_args()
    main_sync(args)
