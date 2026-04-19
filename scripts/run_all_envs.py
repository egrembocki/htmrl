#!/usr/bin/env python3
"""
run_all_envs.py

Easily run all supported RL environments locally and graph performance after training.

Usage:
    python run_all_envs.py [--episodes N] [--no-render] [--step-delay S] [--graph]

Options:
    --episodes N   Number of episodes to run (default: 5)
    --no-render    Disable environment window (headless mode)
    --step-delay S Sleep S seconds after each step (default: 0.08 when rendering, else 0)
    --graph        Plot episode rewards after all runs

Environments tested:
    - CartPole-v1
    - FrozenLake-v1
    - MountainCar-v0
    - Pendulum-v1
    - LunarLander-v3
    - TradingEnv
    - (add more as needed)

Requires: matplotlib, pandas, gymnasium, numpy
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# List of environments to test
envs = [
    "CartPole-v1",
    "FrozenLake-v1",
    "MountainCar-v0",
    "Pendulum-v1",
    "LunarLander-v3",
    "TradingEnv",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_AGENT_SERVER = REPO_ROOT / "run_agent_server.py"
REWARD_FILE_TEMPLATE = "episode_rewards_{env}.json"


def reward_file_path(env):
    return REPO_ROOT / REWARD_FILE_TEMPLATE.format(env=env)


def run_env(env, episodes, render, step_delay, no_spatial=False, no_temporal=False, policy="brain"):
    print(f"\n=== Running {env} for {episodes} episodes | policy={policy} ===")
    cmd = [
        sys.executable,
        str(RUN_AGENT_SERVER),
        "--env",
        env,
        "--mode",
        "local",
        "--policy",
        policy,
        "--log-level",
        "ERROR",
        "--episodes",
        str(episodes),
        "--reward-file",
        str(reward_file_path(env)),
        "--step-delay",
        str(step_delay),
    ]
    if no_spatial:
        cmd.append("--no-spatial")
    if no_temporal:
        cmd.append("--no-temporal")
    if render:
        cmd += ["--render-mode", "human"]
    else:
        cmd += ["--render-mode", "none"]
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] {env} failed with exit code {result.returncode}")
    else:
        print(f"[SUCCESS] {env} completed.")


def plot_rewards(envs, episodes):
    plt.figure(figsize=(10, 6))
    for env in envs:
        reward_file = reward_file_path(env)
        if not reward_file.exists():
            print(f"[WARN] No reward file for {env}, skipping plot.")
            continue
        with reward_file.open("r") as f:
            payload = json.load(f)
        rewards = payload.get("episode_rewards", payload)
        policy = payload.get("policy_mode", "unknown") if isinstance(payload, dict) else "unknown"
        plt.plot(rewards, label=f"{env} ({policy})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Episode Rewards over {episodes} Episodes")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run all RL environments and graph results.")
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes per environment"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        dest="render",
        help="Show environment window (human render mode)",
    )
    parser.add_argument(
        "--no-render",
        action="store_false",
        dest="render",
        help="Disable environment window (headless mode)",
    )
    parser.set_defaults(render=True)
    parser.add_argument(
        "--step-delay",
        type=float,
        default=None,
        help="Delay in seconds after each step (default: 0.08 when rendering, otherwise 0)",
    )
    parser.add_argument("--graph", action="store_true", help="Plot episode rewards after all runs")
    parser.add_argument(
        "--policy",
        type=str,
        default="brain",
        choices=["brain", "ppo", "q_table"],
        help="Agent policy mode for all environments (default: brain)",
    )
    parser.add_argument(
        "--no-spatial",
        action="store_true",
        default=False,
        help="Disable Spatial Pooling in the HTM brain for all environments",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        default=False,
        help="Disable Temporal Memory in the HTM brain for all environments",
    )
    args = parser.parse_args()

    step_delay = args.step_delay if args.step_delay is not None else (0.08 if args.render else 0.0)

    for env in envs:
        run_env(
            env,
            args.episodes,
            args.render,
            step_delay,
            args.no_spatial,
            args.no_temporal,
            args.policy,
        )

    if args.graph:
        plot_rewards(envs, args.episodes)


if __name__ == "__main__":
    main()
