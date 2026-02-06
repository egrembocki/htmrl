"""Base encoder abstractions for creating sparse distributed representations (SDRs).

This module defines :class:`BaseEncoder`, the shared contract for encoder implementations.
Encoders map input values to sparse distributed representations with the following goals:

1. **Semantic similarity**: similar inputs produce overlapping SDRs.
2. **Stability**: an input value maps to the same SDR over time.
3. **Sparsity**: SDRs preserve a consistent sparsity level to support robustness.

Reference: https://arxiv.org/pdf/1602.05925.pdf
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from math import prod
from typing import Any, Generic, TypeVar

from psu_capstone.agent_layer.agent_interface import AgentInterface

T = TypeVar("T")


class BaseEncoder(ABC, Generic[T]):
    """Base class for all encoders.

    Subclasses should implement :meth:`encode` to transform input values into SDRs.
    """

    # class variables

    __interface: AgentInterface | None = None

    __buffered_data: list[dict[str, Any]] | None = None

    __buffer_bounds: tuple[int, int] | None = None

    def __init__(self, dimensions: list[int] | None = None, size: int | None = None):
        """Initialize an encoder with optional dimensions or size.

        Args:
            dimensions: Shape of the SDR when the encoder outputs multi-dimensional SDRs.
            size: Explicit SDR size. If omitted, it is derived from ``dimensions``.
        """

        self._dimensions: list[int] = dimensions if dimensions is not None else []
        self._size: int = size if size is not None else prod(int(dim) for dim in self._dimensions)

    @property
    def interface(self) -> AgentInterface | None:
        """Return the agent interface associated with this encoder, if any."""
        return self.__interface

    @interface.setter
    def interface(self, value: AgentInterface | None) -> None:
        """Set the agent interface associated with this encoder."""
        self.__interface = value

    @property
    def dimensions(self) -> list[int]:
        """Return the SDR dimensions configured for this encoder."""
        return self._dimensions

    @property
    def size(self) -> int:
        """Return the SDR size used by this encoder."""
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        """Set the SDR size for this encoder."""
        self._size = value

    @property
    def buffered_data(self) -> list[dict[str, Any]] | None:
        """Return buffered input records, if any."""
        return self.__buffered_data

    def reset(self):
        """Reset the encoder state, clearing dimensions, size, and buffers."""

        self._dimensions = []
        self._size = 0
        self.__buffered_data = None
        self.__buffer_bounds = None

    def buffer_data(
        self, input_data: list[Mapping[str, Any]], start: int = 0, stop: int | None = None
    ) -> list[dict[str, Any]]:
        """Buffer input data for encoder processing.

        Args:
            input_data (list[Mapping[str, Any]]): Normalized record data to buffer.
            start (int): Inclusive row index where buffering begins.
            stop (int | None): Exclusive row index where buffering ends; defaults to the record length.
        """
        if not isinstance(input_data, list) or not all(
            isinstance(item, Mapping) for item in input_data
        ):
            raise TypeError("input_data must be a list of record mappings")

        total_len = len(input_data)
        if total_len == 0:
            raise ValueError("input_data must contain at least one row")
        if start < 0 or start >= total_len:
            raise ValueError("start must be within the range of input_data")

        stop = total_len if stop is None else stop
        if stop <= start:
            raise ValueError("stop must be greater than start")
        if stop > total_len:
            raise ValueError("stop must not exceed the length of input_data")

        self.__buffer_bounds = (start, stop)
        self.__buffered_data = list(input_data)
        return self.__buffered_data

    @abstractmethod
    def encode(self, input_value: T) -> list[int]:
        """Encode a value into a sparse representation.

        Args:
            input_value: The value to encode.

        Returns:
            The indices of active bits in the SDR.
        """
        raise NotImplementedError("Subclasses must implement this method")
