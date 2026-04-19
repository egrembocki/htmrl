#!/usr/bin/env python3
"""
run_all_envs.py

Easily run all supported RL environments locally and graph performance after training.

Usage:
    python run_all_envs.py [--episodes N] [--render] [--step-delay S] [--graph]

Options:
    --episodes N   Number of episodes to run (default: 200)
    --render       Show environment window (human render mode)
    --step-delay S Sleep S seconds after each step (default: 0.08 when --render, else 0)
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

REWARD_FILE_TEMPLATE = "episode_rewards_{env}.json"


def run_env(env, episodes, render, step_delay):
    print(f"\n=== Running {env} for {episodes} episodes ===")
    cmd = [
        sys.executable,
        "run_agent_server.py",
        "--env",
        env,
        "--mode",
        "local",
        "--policy",
        "brain",
        "--log-level",
        "ERROR",
        "--episodes",
        str(episodes),
        "--reward-file",
        REWARD_FILE_TEMPLATE.format(env=env),
        "--step-delay",
        str(step_delay),
    ]
    if render:
        cmd += ["--render-mode", "human"]
    else:
        cmd += ["--render-mode", "none"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {env} failed with exit code {result.returncode}")
    else:
        print(f"[SUCCESS] {env} completed.")


def plot_rewards(envs, episodes):
    plt.figure(figsize=(10, 6))
    for env in envs:
        reward_file = REWARD_FILE_TEMPLATE.format(env=env)
        if not os.path.exists(reward_file):
            print(f"[WARN] No reward file for {env}, skipping plot.")
            continue
        with open(reward_file, "r") as f:
            payload = json.load(f)
        rewards = payload.get("episode_rewards", payload)
        plt.plot(rewards, label=env)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Episode Rewards over {episodes} Episodes")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run all RL environments and graph results.")
    parser.add_argument(
        "--episodes", type=int, default=200, help="Number of episodes per environment"
    )
    parser.add_argument(
        "--render", action="store_true", help="Show environment window (human render mode)"
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=None,
        help="Delay in seconds after each step (default: 0.08 with --render, otherwise 0)",
    )
    parser.add_argument("--graph", action="store_true", help="Plot episode rewards after all runs")
    args = parser.parse_args()

    step_delay = args.step_delay if args.step_delay is not None else (0.08 if args.render else 0.0)

    for env in envs:
        run_env(env, args.episodes, args.render, step_delay)

    if args.graph:
        plot_rewards(envs, args.episodes)


if __name__ == "__main__":
    main()
