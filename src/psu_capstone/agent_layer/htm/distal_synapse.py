# htm_core/distal_synapse.py
from __future__ import annotations

from .cell import Cell


class DistalSynapse:
    """Distal synapse referencing a source cell (Temporal Memory)."""

    def __init__(self, source_cell: Cell, permanence: float) -> None:
        self.source_cell: Cell = source_cell
        self.permanence: float = permanence
