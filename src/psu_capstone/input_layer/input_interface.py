"""Input layer interface contract"""

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class InputInterface(Protocol):
    """Interface for input handlers."""

    @property
    def data(self) -> pd.DataFrame:
        """Return the normalized DataFrame currently held by the handler."""

        ...

    def input_data(
        self,
        input_source: Any,
        required_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Process input data from various sources into a standardized DataFrame."""
        ...
