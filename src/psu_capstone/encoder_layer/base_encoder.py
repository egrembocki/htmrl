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
from typing import Generic, TypeVar, override

T = TypeVar("T")


class BaseEncoder(ABC, Generic[T]):
    """Base class for all encoders.

    Subclasses should implement :meth:`encode` to transform input values into SDRs.
    """

    def __init__(self, size: int | None = None):
        """Initializes the BaseEncoder with given size."""

        self._size: int = size if size is not None else 0
        self._parameters: ParentDataClass | None = None

    @property
    def size(self) -> int:
        assert self._size >= 0, "size must be a non-negative integer"
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        if value < 0:
            raise ValueError("size must be a non-negative integer")
        elif value == 0:
            raise ValueError("size must be greater than zero")
        self._size = value

    def reset(self):
        """Reset the encoder state, clearing size."""
        self._size = 0
        self.__buffered_data = None
        self.__buffer_bounds = None

    @abstractmethod
    def encode(self, input_value: T) -> list[int]:
        """Encodes the input value into a binary vector."""
        raise NotImplementedError("Subclasses must implement this method")


@dataclass
class ParentDataClass:
    """Parent Class to mark all Parameter Classes for encoders."""

    encoder_class = BaseEncoder
    """Class variable to specify the associated encoder class. Subclasses should override this."""

    size: int = 2048
    """Size of the output SDR. Must be a positive integer."""
