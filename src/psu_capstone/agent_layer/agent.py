"""Agent for HTMRL environment."""

from src.psu_capstone.agent_layer.htm.spatial_pooler import SpatialPooler
from src.psu_capstone.agent_layer.htm.temporal_memory import TemporalMemory


class Agent:
    """Agent for HTMRL environment."""

    __poolers: list[SpatialPooler]
    __memory: list[TemporalMemory]

    def __init__(self, poolers: list[SpatialPooler], memory: list[TemporalMemory]):
        """Initialize the agent."""
        self.__poolers = poolers
        self.__memory = memory

    def select_action(self, state: tuple):
        """Select an action based on the current state.

        Args:
            state: The current state of the environment.

        Returns:
            The action selected by the agent.
        """
        pass

    def update_policy(self, state, action, reward, next_state):
        """Update the agent's policy based on experience.

        Args:
            experience: The experience tuple (state, action, reward, next_state).
        """
        pass
