# htm_core/column.py
from __future__ import annotations

import numpy as np

from psu_capstone.agent_layer.htm.cell import Cell
from psu_capstone.agent_layer.htm.constants import CONNECTED_PERM, MIN_OVERLAP
from psu_capstone.agent_layer.htm.synapse import Synapse


class Column:
    """Column in the HTM region.

    Holds proximal synapses plus a list of cells for Temporal Memory.
    """

    def __init__(self, potential_synapses: list[Synapse], position: tuple[int, int]) -> None:
        self.position: tuple[int, int] = position
        self.potential_synapses: list[Synapse] = potential_synapses

        # Spatial pooler stats
        self.boost: float = 1.0
        self.active_duty_cycle: float = 0.0
        self.overlap_duty_cycle: float = 0.0
        self.min_duty_cycle: float = 0.01

        # Connected proximal synapses
        self.connected_synapses: list[Synapse] = [
            s for s in potential_synapses if s.permanence > CONNECTED_PERM
        ]

        # Overlap score from last compute
        self.overlap: float = 0.0

        # Cells (populated externally by Temporal Memory)
        self.cells: list[Cell] = []

    def compute_overlap(self, input_vector: np.ndarray) -> None:
        """Compute overlap with current binary input vector and apply boost."""
        overlap_raw = sum(1 for s in self.connected_synapses if input_vector[s.source_input])
        if overlap_raw >= MIN_OVERLAP:
            self.overlap = float(overlap_raw * self.boost)
        else:
            self.overlap = 0.0

        print(f"Column at position {self.position} has overlap: {self.overlap}")


if __name__ == "__main__":
    col = Column([], (0, 0))
    assert isinstance(col, Column)
