"""cell: defines the Cell class used in HTM Model."""

# htm_core/cell.py
from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .segment import Segment


class Cell:
    """Single cell within a column.

    Holds a list of distal segments used for Temporal Memory.
    """

    def __init__(self) -> None:
        # Filled by Temporal Memory: list of Segment objects
        self.segments: List["Segment"] = []

    def __repr__(self) -> str:
        return f"Cell(id={id(self)})"
