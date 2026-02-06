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
from collections.abc import Mapping, Sequence
from math import prod
from typing import Any, Generic, TypeVar

from psu_capstone.agent_layer.agent_interface import AgentInterface

T = TypeVar("T")


class BaseEncoder(ABC, Generic[T]):
    """Base class for all encoders"""

    # class variables

    __interface: AgentInterface | None = None

    __buffered_data: list[dict[str, Any]] | None = None

    __buffer_bounds: tuple[int, int] | None = None

    def __init__(self, dimensions: list[int] | None = None, size: int | None = None):
        """Initializes the BaseEncoder with given dimensions."""

        self._dimensions: list[int] = dimensions if dimensions is not None else []
        self._size: int = size if size is not None else prod(int(dim) for dim in self._dimensions)

    @property
    def interface(self) -> AgentInterface | None:
        """Gets the AgentInterface associated with this encoder."""
        return self.__interface

    @interface.setter
    def interface(self, value: AgentInterface | None) -> None:
        """Sets the AgentInterface associated with this encoder."""
        self.__interface = value

    @property
    def dimensions(self) -> list[int]:
        return self._dimensions

    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        self._size = value

    @property
    def buffered_data(self) -> list[dict[str, Any]] | None:
        """Gets the buffered data for processing by the encoder."""
        return self.__buffered_data

    def reset(self):
        """Resets the encoder to its initial state if applicable."""

        self._dimensions = []
        self._size = 0
        self.__buffered_data = None
        self.__buffer_bounds = None

    def buffer_data(
        self, input_data: Any, start: int = 0, stop: int | None = None
    ) -> list[dict[str, Any]]:
        """Buffers the input data for processing by the encoder.

        Args:
            input_data (Any): The input data to be buffered.
            start (int): Inclusive row index where buffering begins.
            stop (int | None): Exclusive row index where buffering ends; defaults to the record length.
        """
        records = self._to_records(input_data)
        total_len = len(records)
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
        self.__buffered_data = records
        return self.__buffered_data

    def _to_records(self, input_data: Any) -> list[dict[str, Any]]:
        """Normalize input data into a list of record dictionaries."""
        if (
            isinstance(input_data, list)
            and input_data
            and all(isinstance(item, Mapping) for item in input_data)
        ):
            return [dict(item) for item in input_data]

        if isinstance(input_data, Mapping):
            values = list(input_data.values())
            if values and all(isinstance(value, list) for value in values):
                max_len = max(len(value) for value in values)
                records = []
                for idx in range(max_len):
                    records.append(
                        {
                            key: (col_values[idx] if idx < len(col_values) else None)
                            for key, col_values in input_data.items()
                        }
                    )
                return records
            return [dict(input_data)]

        if isinstance(input_data, Sequence) and not isinstance(input_data, (str, bytes, bytearray)):
            if input_data and all(
                isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
                for item in input_data
            ):
                num_cols = max(len(item) for item in input_data)
                columns = [f"col_{idx}" for idx in range(num_cols)]
                records = []
                for row in input_data:
                    row_values = list(row)
                    records.append(
                        {
                            columns[idx]: (row_values[idx] if idx < len(row_values) else None)
                            for idx in range(len(columns))
                        }
                    )
                return records
            return [{"value": item} for item in list(input_data)]

        return [{"value": input_data}]

    @abstractmethod
    def encode(self, input_value: T) -> list[int]:
        """Encodes the input value into the provided output SDR by reference."""
        raise NotImplementedError("Subclasses must implement this method")
