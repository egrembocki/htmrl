"""Input layer interface contract"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InputInterface(Protocol):
    """Interface for input handlers."""

    @property
    def data(self) -> dict[Any, list[Any]]:
        """Return the normalized column data currently held by the handler."""

        ...

    def input_data(
        self,
        input_source: Any,
        required_columns: list[str] | None = None,
    ) -> dict[Any, list[Any]]:
        """Process input data from various sources into standardized column data."""
        ...
