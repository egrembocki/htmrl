"""Exposed interfaces for encoder layer components."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EncoderInterface(Protocol):
    """Defines the interface for encoder layer components."""

    def encode(self, input_value: Any) -> list[int]:
        """Encodes a single input value and returns the encoded output.

        Args:
            input_value (Any): The input value to be encoded."""

        ...
