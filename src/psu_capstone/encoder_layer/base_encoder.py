"""Base class for all encoders -- from NuPic Numenta Cpp ported to python.
/**
 * Base class for all encoders.
 * An encoder converts a value to a sparse distributed representation.
 *
 * Subclasses must implement method encode and Serializable interface.
 * Subclasses can optionally implement method reset.
 *
 * There are several critical properties which all encoders must have:
 *
 * 1) Semantic similarity:  Similar inputs should have high overlap.  Overlap
 * decreases smoothly as inputs become less similar.  Dissimilar inputs have
 * very low overlap so that the output representations are not easily confused.
 *
 * 2) Stability:  The representation for an input does not change during the
 * lifetime of the encoder.
 *
 * 3) Sparsity: The output SDR should have a similar sparsity for all inputs and
 * have enough active bits to handle noise and subsampling.
 *
 * Reference: https://arxiv.org/pdf/1602.05925.pdf
 */

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, override

T = TypeVar("T")


class BaseEncoder(ABC, Generic[T]):
    """Base class for all encoders"""

    def __init__(self, size: int | None = None):
        """Initializes the BaseEncoder with given size."""

        self._size: int = size if size is not None else 0

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
        """Resets the encoder to its initial state if applicable."""

        self._dimensions = []
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
