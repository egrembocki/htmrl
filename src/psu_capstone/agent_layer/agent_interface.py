"""Protocol definitions for agent layer components.

This module defines the interface contract that all agents must satisfy
for decision-making and learning in the HTM reinforcement learning system.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np

from psu_capstone.agent_layer.legacy_htm.spatial_pooler import SpatialPooler
from psu_capstone.sdr_layer.sdr_interface import SDRInterface


@runtime_checkable
class AgentInterface(Protocol):
    """Protocol defining the interface requirements for HTM-RL agents.

    Agents implementing this protocol can select actions based on state
    observations and update their internal policy based on experience
    in reinforcement learning environments.
    """

    def select_action(self, state: tuple) -> Any:
        """Select an action based on the current state observation.

        Processes the current state through the agent's decision-making
        mechanism (typically HTM + policy network) to determine the best action.

        Args:
            state: Current environment state as a tuple of observations.

        Returns:
            The action to take in the environment.
        """
        ...

    def step(self, state: tuple, action: Any, reward: float, next_state: tuple, done: bool) -> None:
        """Update the agent's internal policy based on experience.

        This method is called after taking an action in the environment and
        receiving feedback. It allows the agent to learn from the transition
        (state, action, reward, next_state) and update its decision-making
        mechanism accordingly.

        Args:
            state: The previous state before taking the action.
            action: The action that was taken.
            reward: The reward received after taking the action.
            next_state: The new state after taking the action.
            done: Whether the episode has ended after this transition.
        """
        ...
