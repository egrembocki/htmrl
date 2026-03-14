"""Agent for HTMRL environment.

This module defines the Agent class which orchestrates the interaction between
Spatial Poolers and Temporal Memory components within the Hierarchical Temporal
Memory (HTM) Reinforcement Learning framework.
"""

from typing import Any

import numpy as np


class Agent:
    """Agent for HTMRL environment.

    The Agent manages a collection of Spatial Poolers and Temporal Memory units
    to process sensory input, learn temporal patterns, and make decisions within
    the environment.

    Args:
        poolers: A list of initialized SpatialPooler instances.
        memory: A list of initialized TemporalMemory instances.
    """

    def __init__(self):

        self.learning_rate = 0.1
        self.epsilon = 0.1
        self.decay = 0.99
        self.discount_factor = 0.9

    def select_action(self, state: tuple) -> Any:
        """Select an action based on the current state observation.

        This method processes the input state through the agent's HTM hierarchy
        to determine the most appropriate action.

        Args:
            state: The current state of the environment, typically represented
                as a tuple of sensory values.

        Returns:
            The action selected by the agent to be executed in the environment.
        """
        pass

    def update(self, state: tuple, action: Any, reward: float, next_state: tuple) -> None:
        """Update the agent's internal model and policy based on experience.

        This method focuse on updating the MLP model policy based on the agent's
        experience tuple (state, action, reward, next_state).

        Args:
            state: The state observed before taking the action.
            action: The action taken by the agent.
            reward: The reward received from the environment.
            next_state: The state observed after taking the action.
        """
        pass

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict[str, Any]]:  # type: ignore
        """Execute the given action and return the resulting experience tuple.

        This method should interact with the environment to perform the action,
        observe the next state, receive the reward, and determine if the episode
        has terminated.

        Args:
            action: The action to execute in the environment.

        Returns:
            A tuple of (next_state, reward, done, info) where:
                - next_state: The new state observation after taking the action.
                - reward: The reward received from the environment.
                - done: A boolean indicating if the episode has ended.
                - info: A dictionary with any additional information from the environment.
        """

        return ()  # type: ignore
