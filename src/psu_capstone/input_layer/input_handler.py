"""Simplified pandas-backed input handler.

The handler accepts ``Any`` input, detects file-path inputs first, loads/normalizes data
with pandas, validates core constraints, and exposes records as ``list[dict[str, Any]]``.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from psu_capstone.input_layer.input_interface import InputInterface
from psu_capstone.log import logger


class InputHandler:
    """Load, clean, validate, and normalize raw input payloads for the encoding pipeline."""

    __instance: ClassVar[InputHandler | None] = None
    __interface: ClassVar[Any | None] = None

    _SUPPORTED_EXTENSIONS: ClassVar[set[str]] = {
        ".csv",
        ".json",
        ".xlsx",
        ".xls",
        ".parquet",
        ".txt",
    }

    def __new__(cls) -> InputHandler:
        if getattr(cls, "_InputHandler__instance", None) is None:
            cls.__instance = super(InputHandler, cls).__new__(cls)
        return cls.__instance  # type: ignore

    def __init__(self, data: Any | None = None) -> None:
        self._data: list[dict[str, Any]] = []
        self._columns: list[str] = []
        self._repeating_temporal_columns: list[str] = []
        self._contains_multidimensional_data: bool = False
        if data is not None:
            self.input_data(data)

    @classmethod
    def get_instance(cls) -> InputHandler:
        if getattr(cls, "_InputHandler__instance", None) is None:
            cls.__instance = InputHandler()
        return cls.__instance  # type: ignore

    @property
    def data(self) -> list[dict[str, Any]]:
        return self._data

    @data.setter
    def data(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    @property
    def interface(self) -> Any:
        return type(self).__interface

    @interface.setter
    def interface(self, interface: Any) -> None:
        if not isinstance(interface, InputInterface):
            raise TypeError(f"Not all methods from Interface have been implemented in {interface}")
        type(self).__interface = interface

    def input_data(
        self, input_source: Any, required_columns: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Ingest data from files or in-memory payloads and return normalized records."""

        self._data = []
        required = required_columns or []
        if self._is_file_path(input_source):
            df = self._load_file_to_dataframe(input_source)
        else:
            df = self._coerce_non_file_to_dataframe(input_source)

        df = self._process_dataframe(df, required)
        self._columns = list(df.columns)
        self._data = df.to_dict(orient="records")
        return self._data

    def to_encoder_sequence(
        self,
        input_source: Any,
        required_columns: list[str] | None = None,
        column: str | None = None,
    ) -> list[Any]:
        records = self.input_data(input_source, required_columns=required_columns)
        if not records:
            raise ValueError("No records available to build encoder sequence.")

        if column is None:
            if len(records[0]) != 1:
                raise ValueError(
                    "Column must be specified when multiple columns are present in the input."
                )
            column = next(iter(records[0].keys()))

        sequence = [row.get(column) for row in records]
        filtered = [value for value in sequence if value is not None]
        if not filtered:
            raise ValueError("Encoder sequence contains no valid values after filtering.")
        return filtered

    def to_numpy(self, data: list[dict[str, Any]]) -> np.ndarray:
        if not data:
            raise ValueError("Cannot convert empty record list to numpy array.")
        df = pd.DataFrame(data)
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        if numeric_df.isna().all(axis=None):
            raise ValueError("Data validation failed; cannot convert to numpy ndarray.")
        return numeric_df.to_numpy(dtype=np.float64)

    def _is_file_path(self, input_source: Any) -> bool:
        if isinstance(input_source, os.PathLike):
            raise TypeError("Path-like objects must be converted to strings before ingestion.")
        if isinstance(input_source, str):
            path = Path(input_source)
            suffix = path.suffix.lower()
            if path.exists() and path.is_file():
                return True
            if suffix in self._SUPPORTED_EXTENSIONS:
                raise FileNotFoundError(f"No file found at {path}")
        return False

    def _load_file_to_dataframe(self, input_source: str | os.PathLike[str]) -> pd.DataFrame:
        path = Path(input_source)
        ext = path.suffix.lower()
        logger.info("Loading data from %s", path)

        if ext == ".csv":
            return pd.read_csv(path, dtype=str)
        if ext == ".json":
            return pd.read_json(path)
        if ext == ".xlsx":
            return pd.read_excel(path)
        if ext == ".xls":
            raise ValueError("Unsupported file type: .xls (requires optional xls reader).")
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext == ".txt":
            return pd.DataFrame(
                {"value": path.read_text(encoding="utf-8").splitlines(keepends=True)}
            )
        raise ValueError(f"Unsupported file type: {ext}")

    def _coerce_non_file_to_dataframe(self, input_source: Any) -> pd.DataFrame:
        if isinstance(input_source, bytes):
            input_source = bytearray(input_source)

        if isinstance(input_source, pd.DataFrame):
            return input_source.copy()

        if isinstance(input_source, np.ndarray):
            self._contains_multidimensional_data = input_source.ndim > 1
            return pd.DataFrame(input_source)

        if isinstance(input_source, Mapping):
            return pd.DataFrame(input_source)

        if isinstance(input_source, bytearray):
            if b"," in input_source or b"\n" in input_source:
                try:
                    text_payload = input_source.decode("utf-8")
                except UnicodeDecodeError:
                    text_payload = input_source.decode("latin-1")
                try:
                    return pd.read_csv(StringIO(text_payload), dtype=str)
                except Exception:
                    pass
            return pd.DataFrame({"value": list(input_source)})

        if isinstance(input_source, Sequence) and not isinstance(
            input_source, (str, bytes, bytearray)
        ):
            if not input_source:
                return pd.DataFrame()
            first = input_source[0]
            if isinstance(first, Mapping):
                return pd.DataFrame(list(input_source))
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray)):
                self._contains_multidimensional_data = True
                return pd.DataFrame([list(row) for row in input_source])
            return pd.DataFrame({"value": list(input_source)})

        if isinstance(input_source, str):
            return pd.DataFrame({"value": [input_source]})

        raise TypeError(f"Unsupported data type for conversion to sequence: {type(input_source)}")

    def _process_dataframe(self, df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
        if df.empty:
            self._repeating_temporal_columns = []
            return df

        if not df.columns.is_unique:
            duplicates = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(f"Duplicate columns detected: {duplicates}")

        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            for col in missing:
                df[col] = None
            df = df[required_columns]

        df = self._normalize_datetime_columns(df)
        df = self._normalize_column_types(df)
        df = self._fill_missing_values(df)
        self._repeating_temporal_columns = self._detect_repeating_temporal_values(df)
        return df

    def _normalize_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col in result.columns:
            series = result[col]
            non_null = series.dropna()
            if non_null.empty:
                continue

            if pd.api.types.is_datetime64_any_dtype(series):
                result[col] = series.dt.strftime("%Y-%m-%dT%H:%M:%S")
                continue

            if not pd.api.types.is_object_dtype(series):
                continue

            sample = non_null.iloc[0]
            if not isinstance(sample, (str, datetime.date, datetime.datetime)):
                continue

            if isinstance(sample, str):
                looks_temporal = any(token in sample for token in ("-", ":", "T", "/"))
                if not looks_temporal:
                    continue

            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().sum() == non_null.shape[0]:
                result[col] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")
        return result

    def _normalize_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col in result.columns:
            non_null = result[col].dropna()
            if non_null.empty:
                continue
            primitive_types = {type(value) for value in non_null}
            has_non_primitive = any(
                t not in {str, bool, int, float, np.int64, np.float64} for t in primitive_types
            )
            if has_non_primitive:
                raise ValueError(f"Corrupt data detected in column '{col}': unsupported value type")

            if primitive_types.issubset({int, float, np.int64, np.float64, bool}):
                result[col] = pd.to_numeric(result[col], errors="coerce")
            elif len(primitive_types) > 1:
                raise ValueError(
                    f"Mixed primitive types detected in column '{col}': {sorted(t.__name__ for t in primitive_types)}"
                )
        return result

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean_value = result[col].mean(skipna=True)
            if pd.notna(mean_value):
                result[col] = result[col].fillna(mean_value)
        return result

    def _detect_repeating_temporal_values(self, df: pd.DataFrame) -> list[str]:
        repeating_columns: list[str] = []
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().all() and series.duplicated().any():
                repeating_columns.append(str(col))
        return repeating_columns

    def _validate_data(self, required_columns: list[str] | None = None) -> tuple[str, bool]:
        required = required_columns or []
        if not isinstance(self._data, list):
            return ("Data is not a list of records.", False)
        if not self._data:
            return ("Record list is empty", False)
        if not all(isinstance(row, Mapping) for row in self._data):
            return ("Data is valid.", True)

        if self._columns and len(self._columns) != len(set(self._columns)):
            return ("Record list has duplicate column names.", False)

        df = pd.DataFrame(self._data)
        if not df.columns.is_unique:
            return ("Record list has duplicate column names.", False)

        missing_required = [col for col in required if col not in df.columns]
        if missing_required:
            return (f"Missing required columns: {missing_required}", False)

        all_none_cols = [col for col in df.columns if df[col].isna().all()]
        if all_none_cols:
            return (f"Columns with all None values: {all_none_cols}", False)

        return ("Data is valid.", True)
