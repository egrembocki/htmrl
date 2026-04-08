#!/home/millscb/repos/psu-capstone/.venv/bin/python3
"""
run_all_envs.py

Easily run all supported RL environments locally and graph performance after training.

Usage:
    python run_all_envs.py [--episodes N] [--render] [--graph] [--env ENV]

Options:
    --episodes N   Number of episodes to run (default: 200)
    --render       Show environment window (human render mode)
    --graph        Plot episode rewards after all runs
    --env ENV      Run only the specified environment (default: run all)

Environments tested:
    - CartPole-v1
    - FrozenLake-v1
    - MountainCar-v0
    - Pendulum-v1
    - LunarLander-v3
    - TradingEnv

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


def run_env(env, episodes, render):
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
        "--episodes",
        str(episodes),
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
    print(
        f"\n[INFO] Graphing not implemented yet - would plot rewards for {envs} over {episodes} episodes"
    )
    # TODO: Implement reward logging in run_agent_server.py and graphing here
    pass


def main():
    parser = argparse.ArgumentParser(description="Run all RL environments and graph results.")
    parser.add_argument(
        "--episodes", type=int, default=200, help="Number of episodes per environment"
    )
    parser.add_argument(
        "--render", action="store_true", help="Show environment window (human render mode)"
    )
    parser.add_argument("--graph", action="store_true", help="Plot episode rewards after all runs")
    parser.add_argument(
        "--env", type=str, help="Run only the specified environment (default: run all)"
    )
    args = parser.parse_args()

    # Determine which environments to run
    if args.env:
        if args.env not in envs:
            print(f"[ERROR] Environment '{args.env}' not in supported list: {envs}")
            sys.exit(1)
        envs_to_run = [args.env]
    else:
        envs_to_run = envs

    for env in envs_to_run:
        run_env(env, args.episodes, args.render)

    if args.graph:
        plot_rewards(envs_to_run, args.episodes)


if __name__ == "__main__":
    main()
