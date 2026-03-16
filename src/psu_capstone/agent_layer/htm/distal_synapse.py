"""HTM Distal Synapse: connects a source cell to a segment in Temporal Memory.

In HTM Temporal Memory (TM), distal synapses link from a source cell (that was
recently active) to a segment on another cell. Collections of distal synapses
form a distal segment. When enough synapses on a segment are active/connected,
the segment activates and places its owning cell into a predictive state.

Key points:
- Source cell: The cell this synapse monitors for recent activity.
- Permanence: Synaptic strength; values above a threshold count as connected.
- Segment aggregation: Prediction emerges from the combined activity of multiple
  distal synapses on a segment, not from a single synapse alone.
"""

# htm_core/distal_synapse.py
from __future__ import annotations

from psu_capstone.agent_layer.htm.cell import Cell


class DistalSynapse:
    """Distal synapse referencing a source cell (Temporal Memory).

    Represents one learned connection from a source cell to a distal segment on
    another cell. The Temporal Memory algorithm adjusts permanence during
    learning:
    - Increase permanence when the source cell activity correctly predicts the
      target cell's future activation.
    - Decrease permanence on incorrect predictions to prune weak associations.

    Notes:
    - Connectivity is determined by comparing permanence against a threshold
      (defined elsewhere, typically CONNECTED_PERM).
    - This class is intentionally lightweight; evaluation of active/connected
      states and learning rules are performed at the segment level.


        Create a distal synapse with a source cell and initial permanence.

        Parameters:
        - source_cell: The Cell that provides activity to drive this synapse.
        - permanence: The synaptic strength; higher values indicate more stable
          connections. Thresholding determines if the synapse is considered
          connected during inference.

        The owning distal segment will evaluate this synapse against current
        network state to compute segment activation and drive predictions.
        """

    def __init__(self, source_cell: Cell, permanence: float) -> None:
        self.source_cell: Cell = source_cell
        self.permanence: float = permanence


if __name__ == "__main__":
    syn = DistalSynapse(Cell(), 0.5)
    assert isinstance(syn, DistalSynapse)
