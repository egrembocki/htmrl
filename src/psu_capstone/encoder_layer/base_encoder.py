"""Base encoder abstractions for creating sparse distributed representations (SDRs).

This module defines :class:`BaseEncoder`, the shared contract for encoder implementations.
Encoders map input values to sparse distributed representations with the following goals:

1. **Semantic similarity**: similar inputs produce overlapping SDRs.
2. **Stability**: an input value maps to the same SDR over time.
3. **Sparsity**: SDRs preserve a consistent sparsity level to support robustness.

Reference: https://arxiv.org/pdf/1602.05925.pdf
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Any, Generic, TypeVar

import pandas as pd

from psu_capstone.agent_layer.agent_interface import AgentInterface

T = TypeVar("T")


class BaseEncoder(ABC, Generic[T]):
    """Base class for all encoders.

    Subclasses should implement :meth:`encode` to transform input values into SDRs.
    """

    # class variables

    __interface: AgentInterface | None = None

    def __init__(self, size: int | None = None):
        """Initialize an encoder with optional size.

        Args:
            size: Explicit SDR size.
        """
        self._size: int = 0 if size is None else int(size)

    @property
    def interface(self) -> AgentInterface | None:
        """Return the agent interface associated with this encoder, if any."""
        return self.__interface

    @interface.setter
    def interface(self, value: AgentInterface | None) -> None:
        """Set the agent interface associated with this encoder."""
        self.__interface = value

    @property
    def size(self) -> int:
        """Return the SDR size used by this encoder."""
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        """Set the SDR size for this encoder."""
        self._size = value

    def reset(self):
        """Reset the encoder state, clearing size."""
        self._size = 0

    @abstractmethod
    def encode(self, input_value: T) -> list[int]:
        """Encode a value into a sparse representation.

        Args:
            input_value: The value to encode.

        Returns:
            The indices of active bits in the SDR.
        """
        raise NotImplementedError("Subclasses must implement this method")


@dataclass
class ParentDataclass:
    encoder_class = BaseEncoder
