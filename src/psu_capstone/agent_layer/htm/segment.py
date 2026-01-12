# htm_core/segment.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psu_capstone.agent_layer.htm.cell import Cell

from psu_capstone.agent_layer.htm.constants import CONNECTED_PERM
from psu_capstone.agent_layer.htm.distal_synapse import DistalSynapse


class Segment:
    """Distal segment composed of synapses to previously active cells."""

    def __init__(self, synapses: list[DistalSynapse] | None = None) -> None:
        self.synapses: list[DistalSynapse] = synapses if synapses is not None else []
        self.sequence_segment: bool = False  # True if learned in predictive context

    def active_synapses(self, active_cells: set[Cell]) -> list[DistalSynapse]:
        """Return connected synapses whose source cell is active."""
        return [
            syn
            for syn in self.synapses
            if syn.source_cell in active_cells and syn.permanence > CONNECTED_PERM
        ]

    def matching_synapses(self, prev_active_cells: set[Cell]) -> list[DistalSynapse]:
        """Return synapses whose source cell was previously active (ignores permanence threshold)."""
        return [syn for syn in self.synapses if syn.source_cell in prev_active_cells]


# smoke test

if __name__ == "__main__":
    seg = Segment()
    assert isinstance(seg, Segment)
