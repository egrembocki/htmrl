"""Exposed interface for Agent layer components."""

from typing import Protocol, runtime_checkable

from src.psu_capstone.agent_layer.htm.spatial_pooler import SpaitialPooler
from src.psu_capstone.sdr_layer.sdr_interface import SDRInterface as SDR


@runtime_checkable
class AgentInterface(Protocol):
    """Defines the interface for agent layer components."""

    def select_action(self, state: tuple) -> None:
        """Processes incoming data for the agent.

        Args:
            state (tuple): The current state of the environment.
        """
        ...

    def upate_policy(self) -> None:
        """Updated MLP network policy based on experience."""
        ...

    def set_poolers(self, poolers: list) -> None:
        """Sets the spatial poolers for the agent.

        Args:
            poolers (list): List of spatial pooler instances.
        """
        ...

    def set_memory(self, memory: list) -> None:
        """Sets the temporal memory modules for the agent.

        Args:
            memory (list): List of temporal memory instances.
        """
        ...

    def input_pooler(self, pooler: SpaitialPooler, sdr: list[SDR]) -> None:
        """Inputs a spatial pooler into the agent.

        Args:
            pooler (SpatialPooler): The spatial pooler instance to input.
        """
        ...
