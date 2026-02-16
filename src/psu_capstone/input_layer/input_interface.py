"""Input layer interface contract"""

from typing import Any, Protocol, runtime_checkable

from psu_capstone.agent_layer.HTM import Field, InputField


@runtime_checkable
class InputInterface(Protocol):
    """Interface for input handlers."""

    @property
    def data(self) -> dict[Any, list[Any]]:
        """Return the normalized column data currently held by the handler."""

        ...

    @property
    def fields(self) -> list[Field]:
        """Return the primary input fields used for processing."""
        ...

    def input_data(
        self,
        input_source: Any,
        required_columns: list[str] | None = None,
    ) -> dict[Any, list[Any]]:
        """Process input data from various sources into standardized column data."""
        ...
