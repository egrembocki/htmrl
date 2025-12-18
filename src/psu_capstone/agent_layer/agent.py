"""Agent for HTMRL environment.

This module defines the Agent class which orchestrates the interaction between
Spatial Poolers and Temporal Memory components within the Hierarchical Temporal
Memory (HTM) Reinforcement Learning framework.
"""

from typing import Any

from psu_capstone.agent_layer.htm.spatial_pooler import SpatialPooler
from psu_capstone.agent_layer.htm.temporal_memory import TemporalMemory


class Agent:
    """Agent for HTMRL environment.

    The Agent manages a collection of Spatial Poolers and Temporal Memory units
    to process sensory input, learn temporal patterns, and make decisions within
    the environment.

    Attributes:
        _poolers (list[SpatialPooler]): A list of SpatialPooler instances used for
            encoding input patterns.
        _memory (list[TemporalMemory]): A list of TemporalMemory instances used for
            learning sequences and temporal context.
    """

    def __init__(self, poolers: list[SpatialPooler], memory: list[TemporalMemory]):
        """Initialize the agent with specific HTM components.

        Args:
            poolers: A list of initialized SpatialPooler instances.
            memory: A list of initialized TemporalMemory instances.
        """
        self._poolers = poolers
        self._memory = memory

        self._buffer_list: list[Any] = []  # place to store batched sdrs to send to poolers
        self._models: list[Any] = []  # place to store models for agent
        self._interface: list[Any] = []  # place to store interface handlers
        self._encoder_handler: list[Any] = []  # place to store encoder handlers

    @property
    def poolers(self) -> list[SpatialPooler]:
        """Get the agent's spatial poolers.

        Returns:
            list[SpatialPooler]: The list of SpatialPooler instances.
        """
        return self._poolers

    @poolers.setter
    def poolers(self, poolers: list[SpatialPooler]):
        """Set the agent's spatial poolers.

        Args:
            poolers: A list of SpatialPooler instances to set.
        """
        self._poolers = poolers

    @property
    def memory(self) -> list[TemporalMemory]:
        """Get the agent's temporal memory modules.

        Returns:
            list[TemporalMemory]: The list of TemporalMemory instances.
        """
        return self._memory

    @memory.setter
    def memory(self, memory: list[TemporalMemory]):
        """Set the agent's temporal memory modules.

        Args:
            memory: A list of TemporalMemory instances to set.
        """
        self._memory = memory

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

    def append_buffer(self, sdr: Any) -> None:
        """Load an SDR into the agent's buffer list.

        Args:
            sdr: The SDR to be added to the buffer.
        """
        self._buffer_list.append(sdr)

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

        self._poolers.append(pooler)

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
        self._memory.append(memory)

    def create_model(self, model_type: str, **kwargs) -> None:
        """Create a new model for the agent.

        This method allows the agent to instantiate various models (e.g., predictive
        models, classifiers) based on the specified type and parameters.

        Args:
            model_type: A string indicating the type of model to create.
            **kwargs: Additional parameters required for model initialization.
        """
        # Implementation would depend on the specific models supported.

        # build model with parameters

        # append to self._models

        self._models.append(None)  # Placeholder for actual model instance

    def create_encoder(self, encoder_type: str, **kwargs) -> Any:
        """Create a new encoder for the agent.

        This method allows the agent to instantiate various encoders based on the specified type and parameters.

        Args:
            encoder_type: A string indicating the type of encoder to create.
            **kwargs: Additional parameters required for encoder initialization.

        Returns:
            Any: The created encoder instance.


        """
        # Implementation would depend on the specific encoders supported.

        # build encoder with parameters

        # append to self._encoders

        self._encoders.append(None)  # Placeholder for actual encoder instance
