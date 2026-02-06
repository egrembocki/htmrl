"""Exposed interfaces for encoder layer components."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EncoderInterface(Protocol):
    """Defines the interface for encoder layer components."""

    def buffer_data(
        self,
        input_data: list[Mapping[str, Any]],
        start: int = 0,
        stop: int | None = None,
    ) -> list[dict[str, Any]]:
        """Hold a buffered dataset to be processed by encode().

        Args:
            input_data (list[Mapping[str, Any]]): The normalized record dataset to be encoded."""

        ...

    def encode(self, input_value: Any) -> list[int]:
        """Encodes a single input value and returns the encoded output.

        Args:
            input_value (Any): The input value to be encoded."""

        ...
