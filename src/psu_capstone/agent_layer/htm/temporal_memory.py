# htm_core/temporal_memory.py
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .cell import Cell
from .column import Column
from .constants import (
    INITIAL_DISTAL_PERM,
    NEW_SYNAPSE_MAX,
    PERMANENCE_DEC,
    PERMANENCE_INC,
    SEGMENT_ACTIVATION_THRESHOLD,
)
from .distal_synapse import DistalSynapse
from .segment import Segment


class TemporalMemory:
    """Temporal Memory: learns transitions between column activations."""

    def __init__(
        self,
        columns: Sequence[Column],
        cells_per_column: int,
    ) -> None:
        self.columns: List[Column] = list(columns)
        self.cells_per_column: int = cells_per_column

        # Attach cells to each column
        for c in self.columns:
            c.cells = [Cell() for _ in range(cells_per_column)]

        # Time-indexed TM state
        self.active_cells: Dict[int, Set[Cell]] = {}
        self.winner_cells: Dict[int, Set[Cell]] = {}
        self.predictive_cells: Dict[int, Set[Cell]] = {}
        self.learning_segments: Dict[int, Set[Segment]] = {}
        self.negative_segments: Dict[int, Set[Segment]] = {}

        self.current_t: int = 0

        # Optional column -> field mapping if the SP builds one
        self.column_field_map: Dict[Column, str | None] = {}

    # ---------- Core step API ----------

    def step(self, active_columns: Sequence[Column]) -> Dict[str, np.ndarray]:
        """Advance TM one time step given the active columns.

        Returns:
            dict with binary vectors for active_cells, predictive_cells, learning_cells
        """
        t = self.current_t
        active_columns = [c for c in active_columns if isinstance(c, Column)]

        self._compute_active_state(active_columns)
        self._compute_predictive_state()
        self._learn()

        self.current_t += 1

        active_cells_vec = self.cells_to_binary(self.active_cells.get(t, set()))
        predictive_cells_vec = self.cells_to_binary(self.predictive_cells.get(t, set()))
        learning_cells_vec = self.cells_to_binary(self.winner_cells.get(t, set()))

        return {
            "active_cells": active_cells_vec,
            "predictive_cells": predictive_cells_vec,
            "learning_cells": learning_cells_vec,
        }

    # ---------- Core TM logic ----------

    def _compute_active_state(self, active_columns: Sequence[Column]) -> None:
        t = self.current_t
        prev_predictive = self.predictive_cells.get(t - 1, set())
        active_cells_t: Set[Cell] = set()
        winner_cells_t: Set[Cell] = set()
        learning_segments_t: Set[Segment] = set()

        for column in active_columns:
            predictive_cells_prev = [cell for cell in column.cells if cell in prev_predictive]
            if predictive_cells_prev:
                # Correctly predicted column
                for cell in predictive_cells_prev:
                    active_cells_t.add(cell)
                    winner_cells_t.add(cell)
                    for seg in self._active_segments_of(cell, t - 1):
                        learning_segments_t.add(seg)
            else:
                # Bursting: all cells active
                for cell in column.cells:
                    active_cells_t.add(cell)
                best_cell, best_segment = self._best_matching_cell(column, t - 1)
                if best_segment is None:
                    if best_cell is None:
                        best_cell = column.cells[0]
                    best_segment = Segment()
                    best_cell.segments.append(best_segment)
                if best_cell is not None:
                    winner_cells_t.add(best_cell)
                learning_segments_t.add(best_segment)

        self.active_cells[t] = active_cells_t
        self.winner_cells[t] = winner_cells_t
        self.learning_segments[t] = learning_segments_t
        print(f"[TM] Active state at t={t}: {len(active_cells_t)} cells active.")

    def _compute_predictive_state(self) -> None:
        t = self.current_t
        active_cells_t = self.active_cells.get(t, set())
        predictive_cells_t: Set[Cell] = set()
        for column in self.columns:
            for cell in column.cells:
                for seg in cell.segments:
                    if len(seg.active_synapses(active_cells_t)) >= SEGMENT_ACTIVATION_THRESHOLD:
                        predictive_cells_t.add(cell)
                        break
        self.predictive_cells[t] = predictive_cells_t
        print(f"[TM] Predictive state at t={t}: {len(predictive_cells_t)} cells predictive.")

    def _learn(self) -> None:
        t = self.current_t
        prev_predictive = self.predictive_cells.get(t - 1, set())
        active_columns = {
            c
            for c in self.columns
            if any(cell in self.active_cells.get(t, set()) for cell in c.cells)
        }
        negative_segments: Set[Segment] = set()

        # Identify segments that predicted but whose columns did not become active
        for column in self.columns:
            if column not in active_columns:
                for cell in column.cells:
                    if cell in prev_predictive:
                        for seg in self._active_segments_of(cell, t - 1):
                            negative_segments.add(seg)
        self.negative_segments[t] = negative_segments

        # Positive reinforcement
        for seg in self.learning_segments.get(t, set()):
            self._reinforce_segment(seg)

        # Negative reinforcement
        for seg in negative_segments:
            self._punish_segment(seg)

        print(
            f"[TM] Learning at t={t}: +{len(self.learning_segments.get(t, set()))} / "
            f"-{len(negative_segments)} segments."
        )

    # ---------- Helpers (belong with TM) ----------

    def cells_to_binary(self, cells: Set[Cell]) -> np.ndarray:
        """Return binary vector over all cells (flattened across columns)."""
        total_cells = len(self.columns) * self.cells_per_column
        vec = np.zeros(total_cells, dtype=int)
        for col_idx, col in enumerate(self.columns):
            base = col_idx * self.cells_per_column
            for local_idx, cell in enumerate(col.cells):
                if cell in cells:
                    vec[base + local_idx] = 1
        return vec

    def get_predictive_columns_mask(self, t: Optional[int] = None) -> np.ndarray:
        """Return binary vector of predictive columns for time t."""
        if not self.predictive_cells:
            return np.zeros(len(self.columns), dtype=int)
        max_t = max(self.predictive_cells.keys())
        if t is None:
            query_t = max_t
        elif t == -1:
            query_t = max_t - 1
        else:
            query_t = t
        if query_t < 0:
            return np.zeros(len(self.columns), dtype=int)
        pred_cells = self.predictive_cells.get(query_t, set())
        cols = {col for col in self.columns if any(cell in pred_cells for cell in col.cells)}
        mask = np.zeros(len(self.columns), dtype=int)
        for idx, col in enumerate(self.columns):
            if col in cols:
                mask[idx] = 1
        return mask

    def reset_state(self) -> None:
        """Reset transient TM state; learned segments remain."""
        self.active_cells.clear()
        self.winner_cells.clear()
        self.predictive_cells.clear()
        self.learning_segments.clear()
        self.negative_segments.clear()
        self.current_t = 0

    # ---------- Lower-level TM helpers ----------

    def _best_matching_cell(
        self, column: Column, prev_t: int
    ) -> Tuple[Optional[Cell], Optional[Segment]]:
        prev_active_cells = self.active_cells.get(prev_t, set())
        best_cell: Optional[Cell] = None
        best_segment: Optional[Segment] = None
        best_match = -1

        for cell in column.cells:
            if not cell.segments:
                # Prefer unused cell if no better match yet
                if best_match == -1:
                    best_cell = cell
                    best_segment = None
                    best_match = 0
                continue
            for seg in cell.segments:
                match_count = len(seg.matching_synapses(prev_active_cells))
                if match_count > best_match:
                    best_match = match_count
                    best_cell = cell
                    best_segment = seg
        return best_cell, best_segment

    def _active_segments_of(self, cell: Cell, t: int) -> List[Segment]:
        prev_active_cells = self.active_cells.get(t, set())
        active_list: List[Segment] = []
        for seg in cell.segments:
            if len(seg.active_synapses(prev_active_cells)) >= SEGMENT_ACTIVATION_THRESHOLD:
                active_list.append(seg)
        return active_list

    def _reinforce_segment(self, segment: Segment) -> None:
        t = self.current_t
        prev_active_cells = self.active_cells.get(t - 1, set())
        # Strengthen existing active synapses, weaken others
        for syn in segment.synapses:
            if syn.source_cell in prev_active_cells:
                syn.permanence = min(1.0, syn.permanence + PERMANENCE_INC)
            else:
                syn.permanence = max(0.0, syn.permanence - PERMANENCE_DEC)
        # Grow new synapses
        existing_sources = {syn.source_cell for syn in segment.synapses}
        candidates = [c for c in prev_active_cells if c not in existing_sources]
        random.shuffle(candidates)
        for cell_src in candidates[:NEW_SYNAPSE_MAX]:
            segment.synapses.append(DistalSynapse(cell_src, INITIAL_DISTAL_PERM))
        segment.sequence_segment = True

    def _punish_segment(self, segment: Segment) -> None:
        for syn in segment.synapses:
            syn.permanence = max(0.0, syn.permanence - PERMANENCE_DEC)
