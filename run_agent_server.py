#!/usr/bin/env python3
"""Run the agent WebSocket server for real-time visualization.

Branch behavior summary for developers:
- The startup ``--policy`` flag chooses the active policy mode.
- ``brain`` is the default to match current experimentation workflow.
- If ``ppo`` is selected, a fixed warm-up training pass is run before serving.
- No runtime policy switching is performed by the server.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Literal, cast

sys.path.insert(0, str(Path(__file__).parent / "src"))

from psu_capstone.agent_layer.agent import Agent  # noqa: E402
from psu_capstone.agent_layer.agent_server import AgentWebSocketServer  # noqa: E402
from psu_capstone.environment.env_adapter import EnvAdapter  # noqa: E402
from psu_capstone.log import get_logger  # noqa: E402

PPO_PRETRAIN_TIMESTEPS = 50_000


def build_agent(
    env_id: str,
    policy_mode: Literal["q_table", "brain", "ppo"] = "brain",
) -> Agent:
    """Build an agent for the specified environment.

    Plain-language behavior:
    - Build the Brain + EnvAdapter stack once.
    - Initialize Agent using the requested policy mode.
    - Only if mode is ``ppo``, pre-train PPO once before opening the server.
    """
    try:
        from psu_capstone.agent_layer.cartpole_brain_training import (
            CartPoleTrainingConfig,
            build_cartpole_brain,
        )

        config = CartPoleTrainingConfig(
            episodes=1000,
            max_steps_per_episode=150,
            policy_mode=policy_mode,
        )
        adapter = EnvAdapter(env_id)
        brain = build_cartpole_brain(adapter, config)

        agent = Agent(
            brain=brain,
            adapter=adapter,
            episodes=config.episodes,
            policy_mode=policy_mode,
        )

        if policy_mode == "ppo":
            # Keep PPO warm-up explicit and mode-scoped so brain/q_table startup
            # remains immediate and predictable.
            logger = get_logger(None)
            logger.info("Pre-training PPO for %d timesteps...", PPO_PRETRAIN_TIMESTEPS)
            agent.train_ppo(total_timesteps=PPO_PRETRAIN_TIMESTEPS)
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
        policy_mode = cast(Literal["q_table", "brain", "ppo"], args.policy)
        agent = build_agent(args.env, policy_mode=policy_mode)

        logger.info(f"Starting WebSocket server on ws://{args.host}:{args.port}")
        server = AgentWebSocketServer(
            agent,
            host=args.host,
            port=args.port,
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
        "--policy",
        type=str,
        default="brain",
        choices=["ppo", "brain", "q_table"],
        help="Starting policy mode (default: brain)",
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
