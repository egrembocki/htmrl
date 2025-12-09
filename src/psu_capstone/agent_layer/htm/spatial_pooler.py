"""Spatial Pooler: maps input SDRs to active columns."""

# htm_core/spatial_pooler.py
from __future__ import annotations

from typing import (
    Dict,
    List,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from .column import Column
from .constants import (
    CONNECTED_PERM,
    DESIRED_LOCAL_ACTIVITY,
    PERMANENCE_DEC,
    PERMANENCE_INC,
)
from .synapse import Synapse


class SpatialPooler:
    """Spatial Pooler: maps input SDRs to active columns."""

    _input_field = Union[np.ndarray, Sequence[int]]
    _input_composite = Union[
        np.ndarray,
        Sequence[int],
        Sequence[_input_field],
        Dict[str, _input_field],
    ]

    def __init__(
        self,
        input_space_size: int,
        column_count: int,
        initial_synapses_per_column: int,
        random_seed: int = 0,
    ) -> None:
        self.input_space_size: int = int(input_space_size)
        self.column_count: int = column_count
        self.random_seed: int = random_seed

        self.columns: List[Column] = self._initialize_region(
            input_space_size,
            column_count,
            initial_synapses_per_column,
            self.random_seed,
        )

        # Multi-field metadata for dict inputs
        self.field_ranges: Dict[str, Tuple[int, int]] = {}
        self.field_order: List[str] = []
        self.column_field_map: Dict[Column, str | None] = {}

    def _initialize_region(
        self,
        input_space_size: int,
        column_count: int,
        initial_synapses_per_column: int,
        random_seed: int,
    ) -> List[Column]:
        columns: List[Column] = []
        grid_size = int(column_count**0.5)  # assume square grid
        rng = np.random.default_rng(random_seed)

        for i in range(column_count):
            x = i % grid_size
            y = i // grid_size
            position = (x, y)
            potential_synapses = [
                Synapse(
                    int(rng.integers(input_space_size)),
                    float(rng.uniform(0.4, 0.6)),
                )
                for _ in range(initial_synapses_per_column)
            ]
            columns.append(Column(potential_synapses, position))

        print(f"[SP] Initialized {len(columns)} columns with positions and potential synapses.")
        return columns

    # ---------- Input combination & field metadata ----------

    def combine_input_fields(self, input_vector: _input_composite) -> np.ndarray:
        """Prepare / combine input fields into a single binary numpy array."""
        if isinstance(input_vector, dict):
            start = 0
            arrays: List[np.ndarray] = []
            self.field_ranges = {}
            self.field_order = []
            for name, arr in input_vector.items():
                a = np.asarray(arr, dtype=int)
                end = start + a.shape[0]
                self.field_ranges[name] = (start, end)
                self.field_order.append(name)
                arrays.append(a)
                start = end
            combined = np.concatenate(arrays) if arrays else np.array([], dtype=int)
            self.column_field_map = {}
        elif isinstance(input_vector, (list, tuple)):
            arrays = [np.asarray(v, dtype=int) for v in input_vector]
            combined = np.concatenate(arrays) if arrays else np.array([], dtype=int)
        else:
            combined = np.asarray(input_vector, dtype=int)

        if combined.shape[0] != self.input_space_size:
            raise ValueError(
                f"Combined input length {combined.shape[0]} != configured input_space_size {self.input_space_size}."
            )

        # Clean up bad metadata, if any
        if self.field_ranges and any(len(rg) != 2 for rg in self.field_ranges.values()):
            self.field_ranges = {}
            self.field_order = []
            self.column_field_map = {}

        if self.field_ranges and not self.column_field_map:
            self._assign_column_fields()

        return combined

    def _columns_from_raw_input(self, combined: np.ndarray) -> List[Column]:
        """Return columns that receive at least one active (1) bit via a connected synapse."""
        cols: List[Column] = []
        active_indices = np.nonzero(combined > 0)[0]
        active_set = {int(i) for i in active_indices}
        for col in self.columns:
            if any(s.source_input in active_set for s in col.connected_synapses):
                cols.append(col)
        return cols

    def _assign_column_fields(self) -> None:
        """Assign each column a dominant field based on connected synapse source indices."""
        if not self.field_ranges:
            return
        inv_order = {name: i for i, name in enumerate(self.field_order)}
        for col in self.columns:
            counts: Dict[str, int] = {}
            for syn in col.connected_synapses:
                idx = syn.source_input
                for name, (s, e) in self.field_ranges.items():
                    if s <= idx < e:
                        counts[name] = counts.get(name, 0) + 1
                        break
            if counts:
                best = sorted(
                    counts.items(),
                    key=lambda kv: (-kv[1], inv_order[kv[0]]),
                )[
                    0
                ][0]
                self.column_field_map[col] = best
            else:
                self.column_field_map[col] = None

    # ---------- Core SP computation ----------

    def compute_active_columns(
        self,
        input_vector: _input_composite,
        inhibition_radius: float,
    ) -> tuple[np.ndarray, List[Column]]:
        """Compute active columns given an input SDR.

        Returns:
            (mask, active_columns_list)
        """
        combined = self.combine_input_fields(input_vector)
        for c in self.columns:
            c.compute_overlap(combined)
        active_columns = self._inhibition(self.columns, inhibition_radius)
        mask = self.columns_to_binary(active_columns)
        print(f"[SP] Computed active columns. Total active columns: {int(mask.sum())}")
        return mask, active_columns

    # ---------- Helpers (belong with SP) ----------

    def columns_to_binary(self, columns: Sequence[Column]) -> np.ndarray:
        mask = np.zeros(len(self.columns), dtype=int)
        col_index = {c: i for i, c in enumerate(self.columns)}
        for c in columns:
            idx = col_index.get(c)
            if idx is not None:
                mask[idx] = 1
        return mask

    def _inhibition(self, columns: Sequence[Column], inhibition_radius: float) -> List[Column]:
        active_columns: List[Column] = []
        for c in columns:
            neighbors = [
                c2
                for c2 in columns
                if c is not c2
                and self._euclidean_distance(c.position, c2.position) <= inhibition_radius
            ]
            min_local_activity = self._kth_score(neighbors, DESIRED_LOCAL_ACTIVITY)
            if c.overlap > 0 and c.overlap >= min_local_activity:
                active_columns.append(c)
        print(f"[SP] After inhibition, active columns: {[c.position for c in active_columns]}")
        return active_columns

    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))

    def _kth_score(self, neighbors: Sequence[Column], k: int) -> float:
        if not neighbors:
            return 0.0
        ordered = sorted(neighbors, key=lambda x: x.overlap, reverse=True)
        if k <= 0:
            return 0.0
        if k > len(ordered):
            return float(ordered[-1].overlap) if ordered else 0.0
        return float(ordered[k - 1].overlap)

    # ---------- Spatial learning ----------

    def learning_phase(self, active_columns: Sequence[Column], input_vector: np.ndarray) -> None:
        """Spatial Pooler permanence adaptation for currently active columns."""
        for c in active_columns:
            for s in c.potential_synapses:
                if input_vector[s.source_input]:
                    s.permanence = min(1.0, s.permanence + PERMANENCE_INC)
                else:
                    s.permanence = max(0.0, s.permanence - PERMANENCE_DEC)
            c.connected_synapses = [
                s for s in c.potential_synapses if s.permanence > CONNECTED_PERM
            ]
        print(f"[SP] Learning phase updated synapses for {len(active_columns)} active columns.")
        _ = self.average_receptive_field_size()

    def average_receptive_field_size(self) -> float:
        total_receptive_field_size = 0
        count = 0
        for c in self.columns:
            connected_positions = [s.source_input for s in c.connected_synapses]
            if connected_positions:
                receptive_field_size = max(connected_positions) - min(connected_positions)
                total_receptive_field_size += receptive_field_size
                count += 1
        return total_receptive_field_size / count if count > 0 else 0.0
