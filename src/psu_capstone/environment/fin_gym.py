"""Tabular-data financial Gym environment.

This environment is designed for spreadsheet-like datasets with many rows and
columns. Each timestep corresponds to one row in the table and observations are
numeric feature vectors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd


class FinGym(gym.Env):
    """Gymnasium environment for row-wise tabular/financial datasets.

    This environment treats a table like a time-indexed sequence where each
    row is one timestep. The observation is a numeric vector built from selected
    feature columns. Rewards can optionally be derived from the delta of a
    target column between consecutive rows.

    Args:
        data_source: Table source as DataFrame, file path, or mapping of
            columns to value sequences.
        feature_columns: Optional ordered list of columns used as observation
            features. If omitted, all columns except ``target_column`` are used.
        target_column: Optional numeric column used to compute reward from
            row-to-row deltas.
        max_rows: Optional cap on number of rows read from the dataset.

        Notes:
                - Actions are ``0=hold``, ``1=long``, ``2=short``.
                - If ``target_column`` is not provided, rewards are always ``0.0``.
                - Non-numeric feature values are coerced to numeric and missing values
                    are filled with ``0.0`` for stable observation vectors.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_source: pd.DataFrame | str | Path | Mapping[str, Sequence[Any]],
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        max_rows: int | None = None,
    ) -> None:
        super().__init__()

        self._raw_frame = self._load_frame(data_source)
        if max_rows is not None:
            if max_rows <= 1:
                raise ValueError("max_rows must be greater than 1.")
            self._raw_frame = self._raw_frame.iloc[:max_rows].copy()

        if len(self._raw_frame) < 2:
            raise ValueError("FinGym requires at least 2 rows to step through data.")

        self._target_column = target_column
        self._feature_columns = self._resolve_feature_columns(feature_columns, target_column)

        # Normalize all selected feature columns into a dense float matrix that
        # can be consumed directly by Gym/agents.
        features = self._raw_frame[self._feature_columns].apply(pd.to_numeric, errors="coerce")
        features = features.fillna(0.0).astype(np.float32)
        self._feature_matrix = features.to_numpy(dtype=np.float32)

        self._target_values: np.ndarray | None = None
        if self._target_column is not None:
            if self._target_column not in self._raw_frame.columns:
                raise ValueError(f"Unknown target column: {self._target_column}")
            target_series = pd.to_numeric(
                self._raw_frame[self._target_column], errors="coerce"
            ).fillna(0.0)
            self._target_values = target_series.to_numpy(dtype=np.float32)

        self._row_count = self._feature_matrix.shape[0]
        self._current_row = 0

        # 0=hold, 1=long, 2=short.
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self._feature_columns),),
            dtype=np.float32,
        )

    @property
    def feature_columns(self) -> list[str]:
        """Return the ordered feature column names used for observations."""

        return list(self._feature_columns)

    @property
    def target_column(self) -> str | None:
        """Return the optional target column used for reward calculation."""

        return self._target_column

    def _load_frame(
        self,
        data_source: pd.DataFrame | str | Path | Mapping[str, Sequence[Any]],
    ) -> pd.DataFrame:
        """Load tabular data from common spreadsheet-like sources.

        Supported file inputs: ``.csv``, ``.xlsx``, ``.xls``, ``.parquet``,
        ``.json``.
        """

        if isinstance(data_source, pd.DataFrame):
            return data_source.copy()

        if isinstance(data_source, Mapping):
            return pd.DataFrame(data_source)

        if isinstance(data_source, (str, Path)):
            path = Path(data_source)
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return pd.read_csv(path)
            if suffix in {".xlsx", ".xls"}:
                return pd.read_excel(path)
            if suffix == ".parquet":
                return pd.read_parquet(path)
            if suffix == ".json":
                return pd.read_json(path)
            raise ValueError(f"Unsupported file format: {suffix}")

        raise TypeError("data_source must be a DataFrame, mapping, or supported file path")

    def _resolve_feature_columns(
        self,
        feature_columns: list[str] | None,
        target_column: str | None,
    ) -> list[str]:
        """Resolve and validate feature columns used for observations.

        If ``feature_columns`` is omitted, all columns except the target are
        used as features.
        """

        if feature_columns is None:
            resolved = [column for column in self._raw_frame.columns if column != target_column]
        else:
            resolved = feature_columns

        if not resolved:
            raise ValueError("No feature columns available for observations.")

        missing = [column for column in resolved if column not in self._raw_frame.columns]
        if missing:
            raise ValueError(f"Unknown feature columns: {missing}")

        return list(resolved)

    def _current_observation(self) -> np.ndarray:
        """Return current row features as an observation vector."""

        return self._feature_matrix[self._current_row].copy()

    def _reward_from_action(self, action: int) -> float:
        """Compute reward from target delta for hold/long/short actions.

        Reward is based on ``target[t+1] - target[t]``:
            - long (1): positive delta is rewarded
            - short (2): negative delta is rewarded
            - hold (0): zero reward
        """

        if self._target_values is None or self._current_row >= self._row_count - 1:
            return 0.0

        current_value = self._target_values[self._current_row]
        next_value = self._target_values[self._current_row + 1]
        delta = float(next_value - current_value)

        if action == 1:
            return delta
        if action == 2:
            return -delta

        return 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to the first row of the tabular dataset.

        Returns:
            ``(observation, info)`` where observation is the first row feature
            vector and info includes row index and active column metadata.
        """

        super().reset(seed=seed)
        self._current_row = 0
        observation = self._current_observation()
        info = {
            "row_index": self._current_row,
            "feature_columns": self.feature_columns,
            "target_column": self._target_column,
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one row and return Gymnasium step tuple.

        The environment advances until the final valid transition
        ``row_count - 2 -> row_count - 1`` has been consumed.
        """

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        reward = self._reward_from_action(action)

        # Terminal is raised when there is no next transition to evaluate.
        terminated = self._current_row >= self._row_count - 2
        truncated = False

        if not terminated:
            self._current_row += 1

        observation = self._current_observation()
        info = {
            "row_index": self._current_row,
            "reward_source": self._target_column,
        }

        return observation, reward, terminated, truncated, info

    def render(self) -> dict[str, Any]:
        """Return the current row payload for debugging/inspection.

        This returns structured data rather than printing so callers can log,
        assert, or visualize the current state as needed.
        """

        row = self._raw_frame.iloc[self._current_row]
        return {
            "row_index": self._current_row,
            "row": row.to_dict(),
        }

    def close(self) -> None:
        """Close environment resources."""
        super().close()
        return None
