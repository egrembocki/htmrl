"""Agent for HTMRL environment.

This module defines the Agent class which orchestrates the interaction between
Spatial Poolers and Temporal Memory components within the Hierarchical Temporal
Memory (HTM) Reinforcement Learning framework.
"""

from typing import Any

from src.psu_capstone.agent_layer.htm.spatial_pooler import SpatialPooler
from src.psu_capstone.agent_layer.htm.temporal_memory import TemporalMemory


class Agent:
    """Agent for HTMRL environment.

    The Agent manages a collection of Spatial Poolers and Temporal Memory units
    to process sensory input, learn temporal patterns, and make decisions within
    the environment.

    Attributes:
        __poolers (list[SpatialPooler]): A list of SpatialPooler instances used for
            encoding input patterns.
        __memory (list[TemporalMemory]): A list of TemporalMemory instances used for
            learning sequences and temporal context.
    """

    def __init__(self, poolers: list[SpatialPooler], memory: list[TemporalMemory]):
        """Initialize the agent with specific HTM components.

        Args:
            poolers: A list of initialized SpatialPooler instances.
            memory: A list of initialized TemporalMemory instances.
        """
        self.__poolers = poolers
        self.__memory = memory
        self._interface: Any
        self._sdr_size: int = 0
        self._cells_per_column: int = 0
        self._sparisty: float = 0.0
        self._list[Any] = []

    def select_action(self, state: tuple):
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

    def update_policy(self, state, action, reward, next_state):
        """Update the agent's internal model and policy based on experience.

        This method allows the agent to learn from the consequences of its actions
        by updating its Spatial Pooler and Temporal Memory connections.

        Args:
            state: The state observed before taking the action.
            action: The action taken by the agent.
            reward: The reward received from the environment.
            next_state: The state observed after taking the action.
        """
        pass

    def set_memory(self, memory: list[TemporalMemory]):
        """Set the agent's temporal memory modules directly.

        Args:
            memory: A list of TemporalMemory instances to replace the current memory modules.
        """
        self.__memory = memory

    def set_poolers(self, poolers: list[SpatialPooler]):
        """Set the agent's spatial pooler modules directly.

        Args:
            poolers: A list of SpatialPooler instances to replace the current poolers.
        """
        self.__poolers = poolers

    def create_pooler(
        self, input_space_size: int, column_count: int, synapse_per_column: int, seed: int
    ) -> None:
        """Create a new Spatial Pooler and append it to the agent's poolers.

        This factory method simplifies the addition of new spatial pooling layers
        to the agent.

        Args:
            input_space_size: The total size of the input bit space.
            column_count: The number of columns (outputs) in the spatial pooler.
            synapse_per_column: The number of potential synapses per column.
            seed: Random seed for reproducible initialization.
        """
        pooler = SpatialPooler(
            input_space_size=input_space_size,
            column_count=column_count,
            synapse_per_column=synapse_per_column,
            seed=seed,
        )
        self.__poolers.append(pooler)

    def create_memory(self, column_count: int, cells_per_column: int, seed: int) -> None:
        """Create a new Temporal Memory module and append it to the agent's memory.

        This factory method simplifies the addition of new temporal memory layers
        to the agent.

        Args:
            column_count: The number of columns in the temporal memory (should match
                the output of the corresponding Spatial Pooler).
            cells_per_column: The number of cells per column for temporal context.
            seed: Random seed for reproducible initialization.
        """
        memory = TemporalMemory(
            column_count=column_count, cells_per_column=cells_per_column, seed=seed
        )
        self.__memory.append(memory)

    def set_interface(self, interface: Any) -> None:
        """Set the agent's interface for environment interaction.

        Args:
            interface: An object that defines how the agent interacts with the environment.
        """
        self._interface = interface

    def set_list(self, lst: list[Any]) -> None:
        """Set an internal list for the agent.

        Args:
            lst: A list of any type to be stored internally by the agent.
        """
        self._list = lst
