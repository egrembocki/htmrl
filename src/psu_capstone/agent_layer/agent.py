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

    def set_memory(self, memory: list[TemporalMemory]):
        """Set the agent's temporal memory modules.

        Args:
            memory: A list of TemporalMemory instances.
        """
        self.__memory = memory

    def set_poolers(self, poolers: list[SpatialPooler]):
        """Set the agent's spatial pooler modules.

        Args:
            poolers: A list of SpatialPooler instances.
        """
        self.__poolers = poolers

    def create_pooler(
        self, input_space_size: int, column_count: int, synapse_per_column: int, seed: int
    ) -> None:
        """Create and add a spatial pooler to the agent.

        Args:
            input_space_size: Size of the input space.
            column_count: Number of columns in the spatial pooler.
            synapse_per_column: Number of synapses per column.
            seed: Random seed for initialization.
        """
        pooler = SpatialPooler(
            input_space_size=input_space_size,
            column_count=column_count,
            synapse_per_column=synapse_per_column,
            seed=seed,
        )
        self.__poolers.append(pooler)

    def create_memory(self, column_count: int, cells_per_column: int, seed: int) -> None:
        """Create and add a temporal memory module to the agent.

        Args:
            column_count: Number of columns in the temporal memory.
            cells_per_column: Number of cells per column.
            seed: Random seed for initialization.
        """
        memory = TemporalMemory(
            column_count=column_count, cells_per_column=cells_per_column, seed=seed
        )
        self.__memory.append(memory)
