"""HTM Distal Segment: aggregates distal synapses for sequence prediction.

In HTM Temporal Memory (TM), each Cell can have multiple distal segments.
A segment groups distal synapses (links to other recently active cells). When
enough synapses on a segment are simultaneously active and connected, the
segment is considered active, putting the owning cell into a predictive state.

Concepts:
- Distal Synapse: Connects from a source cell to this segment; permanence
  determines whether the synapse is connected (above CONNECTED_PERM).
- Active Synapses: Synapses whose source cells are currently active and connected.
- Matching Synapses: Synapses whose source cells were active in the previous
  timestep (used for learning, regardless of permanence).
- Sequence Segment: A segment learned while the cell was predictive; these
  typically encode transitions and are preferred for sequence disambiguation.
"""

# htm_core/segment.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psu_capstone.agent_layer.htm.cell import Cell

from psu_capstone.agent_layer.htm.constants import CONNECTED_PERM
from psu_capstone.agent_layer.htm.distal_synapse import DistalSynapse


class Segment:
    """Distal segment composed of synapses to previously active cells.

    Holds and evaluates a set of distal synapses to determine predictive
    activation. Learning logic (growing/pruning synapses, adjusting permanence)
    typically occurs elsewhere but uses segment queries to decide updates.

    Attributes:
    - synapses: List of DistalSynapse objects owned by this segment.
    - sequence_segment: True if learned while the cell was predictive, which
      usually indicates a sequence-specific transition.
    """

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
