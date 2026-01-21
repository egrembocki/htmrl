"""Exposed interface for Agent layer components."""

from typing import Any, Protocol, runtime_checkable

import numpy as np

from psu_capstone.agent_layer.htm.spatial_pooler import SpatialPooler
from psu_capstone.sdr_layer.sdr_interface import SDRInterface


@runtime_checkable
class AgentInterface(Protocol):
    """Defines the interface requiremnts to be an Agent in the HTM-RL system."""

    def select_action(self, state: tuple) -> Any:
        """Select an action based on the current state.

        Args:
            state (tuple): The current state of the environment.
        """
        ...

    def update_policy(self, state: tuple, action: Any, reward: float, next_state: tuple) -> None:
        """Update the agent's internal model and policy based on experience.

        Args:
            state: The previous state of the environment.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_state: The new state of the environment after the action.
        """
        ...
