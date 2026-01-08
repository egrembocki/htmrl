"""HTM Column: proximal synapses (Spatial Pooler) and cells (Temporal Memory).

Concepts:
- Spatial Pooler (SP): Columns connect to the input space via proximal synapses.
  Each synapse has a permanence; synapses with permanence > CONNECTED_PERM are
  considered connected. Column overlap measures how strongly a column matches
  the current input, and boost helps under-active columns compete.
- Temporal Memory (TM): Each column hosts multiple cells. Cells learn temporal
  context through distal segments, enabling sequence prediction. This module
  only maintains the list of cells; TM logic resides in Cell/Segment/Synapse
  and higher-level controllers.

Key fields:
- potential_synapses: All candidate proximal synapses to the input space.
- connected_synapses: Subset of potential_synapses with permanence above the
  connection threshold.
- boost/active_duty_cycle/overlap_duty_cycle: SP homeostasis metrics to
  encourage fair participation.
- cells: The per-column set of cells used by TM.
"""

# htm_core/column.py
from __future__ import annotations

from typing import Any

import numpy as np

from psu_capstone.agent_layer.htm.cell import Cell
from psu_capstone.agent_layer.htm.constants import CONNECTED_PERM, MIN_OVERLAP
from psu_capstone.agent_layer.htm.synapse import Synapse


class Column:
    """Column in an HTM region, participating in SP and hosting TM cells.

    Responsibilities:
    - Spatial Pooler:
      - Maintain proximal synapses and compute input overlap.
      - Apply boosting based on duty cycles to ensure column participation.
    - Temporal Memory:
      - Hold a list of cells; TM attaches distal segments to those cells and
        drives learning/prediction. This class does not implement TM updates.

    Notes:
    - position identifies the column in a 2D lattice.
    - connected_synapses is derived from potential_synapses using CONNECTED_PERM.
    """

    def __init__(self, potential_synapses: list[Synapse], position: tuple[int, int]) -> None:
        """Construct an HTM Column with given proximal synapses and position."""

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
        """Compute boosted overlap with a binary input vector.

        Process:
        - Count connected proximal synapses whose source_input index is active
          in input_vector.
        - If the raw overlap meets MIN_OVERLAP, scale by boost; otherwise set 0.
        - Store the result in self.overlap for downstream SP competition.

        Parameters:
        - input_vector: A 1D numpy array of binary activations representing the
          current input space; indices must align with synapse.source_input.

        Side effects:
        - Updates self.overlap.
        - Prints a simple trace message with the column's position and overlap.
        """
        overlap_raw = sum(1 for s in self.connected_synapses if input_vector[s.source_input])
        if overlap_raw >= MIN_OVERLAP:
            self.overlap = float(overlap_raw * self.boost)
        else:
            self.overlap = 0.0

        print(f"Column at position {self.position} has overlap: {self.overlap}")
