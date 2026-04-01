import json

import matplotlib.pyplot as plt


def plot_rewards_from_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    rewards = data["episode_rewards"]
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Rewards Over Time")
    plt.show()


if __name__ == "__main__":
    plot_rewards_from_file("episode_rewards.json")
