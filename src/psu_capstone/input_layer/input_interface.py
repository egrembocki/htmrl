"""Input layer interface contract"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InputInterface(Protocol):
    """Interface for input handlers."""

    @property
    def data(self) -> list[Mapping[str, Any]]:
        """Return the normalized record list currently held by the handler."""

        ...

    def input_data(
        self,
        input_source: Any,
        required_columns: list[str] | None = None,
    ) -> list[Mapping[str, Any]]:
        """Process input data from various sources into standardized records."""
        ...

    def to_encoder_sequence(
        self,
        input_source: Any,
        required_columns: list[str] | None = None,
        column: str | None = None,
    ) -> list[Any]:
        """Return a list of values suitable for encoder encode() calls."""
        ...
