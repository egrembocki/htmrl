"""Protocol definitions for encoder layer components.

This module defines the interface contract that all encoder implementations
must satisfy, using Python's Protocol for structural subtyping.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EncoderInterface(Protocol):
    """Protocol defining the interface for encoder layer components.

    Encoders implementing this protocol convert input values into sparse
    distributed representations. The protocol uses structural subtyping,
    so any class with matching method signatures automatically satisfies it.
    """

    def buffer_data(self, input_data: Any, start: int = 0, stop: int | None = None) -> Any:
        """Buffer a dataset for batch processing.

        Stores input data internally for later encoding operations. This allows
        efficient batch processing of multiple values.

        Args:
            input_data: The dataset to be buffered (typically a DataFrame or array).
            start: Inclusive row index where buffering begins.
            stop: Exclusive row index where buffering ends. Defaults to end of data.

        Returns:
            The buffered data structure.
        """
        ...

    def encode(self, input_value: Any) -> list[int]:
        """Encode a single input value into an SDR.

        Transforms the input value into a sparse distributed representation
        expressed as a list of active bit indices.

        Args:
            input_value: The value to encode.

        Returns:
            List of active bit indices representing the encoded value.
        """
        ...
