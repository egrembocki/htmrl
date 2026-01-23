"""InputHandler for HTM pipeline. Build a Singleton handler that normalizes varied payloads into Encoder-ready DataFrames."""

from __future__ import annotations

import datetime
import os
from collections.abc import Sequence
from typing import Any, Callable, ClassVar

import numpy as np
import pandas as pd

from psu_capstone.input_layer.input_interface import InputInterface
from psu_capstone.log import logger


class InputHandler:
    """Build an input data manager."""

    __instance: ClassVar[InputHandler | None] = None
    """Singleton instance"""

    __interface: ClassVar[Any]  # may get removed in future updates -- sonarqube(python:S4487)
    """Interface for future extension"""

    _DATAFRAME_READERS: ClassVar[dict[str, Callable[[str], pd.DataFrame]]] = {
        ".csv": pd.read_csv,
        ".xls": pd.read_excel,
        ".xlsx": pd.read_excel,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
    }
    """Mapping of file extensions to pandas DataFrame reader functions."""

    _TEXT_EXTENSION: ClassVar[str] = ".txt"
    """File extension for text files."""

    def __new__(cls) -> "InputHandler":
        """Constructor ::  Singleton pattern implementation."""

        if cls.__instance is None:
            cls.__instance = super(InputHandler, cls).__new__(cls)

        return cls.__instance

    def __init__(self, data: Any = None) -> None:
        """Constructor :: called after new()."""

        self._data: pd.DataFrame | np.ndarray
        """The processed input data -> encoder-ready DataFrame."""

        self._appended_required_columns: set[str] = set()
        """Set of required columns."""

        self._required_columns_context: list[str] = []
        """Most-recent required_columns list passed to input_data (used for timestamp behavior)"""

    @classmethod
    def get_instance(cls) -> "InputHandler":
        """Static access method to get the singleton instance."""

        if cls.__instance is None:
            cls.__instance = InputHandler()
        return cls.__instance

    @property
    def data(self) -> pd.DataFrame | np.ndarray:
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame | np.ndarray) -> None:
        self._data = data

    @property
    def interface(self) -> Any:
        return type(self).__interface

    @interface.setter
    def interface(self, interface: Any) -> None:
        type(self).__interface = interface

    def input_data(
        self, input_source: Any, required_columns: list[str] | None = None
    ) -> pd.DataFrame | np.ndarray:
        """Public :: exposed method to the user. Inputing data into the handler starts here

        raises:
            TypeError: If input_source is a path-like object.
            FileNotFoundError: If a file path is provided but the file does not exist.
            ValueError: If data validation fails.
        """

        self._appended_required_columns.clear()

        # Check if input is a file path
        if isinstance(input_source, os.PathLike):
            raise TypeError("Path-like objects must be converted to strings before ingestion.")

        if isinstance(input_source, str):
            file_extension = os.path.splitext(input_source)[1].lower()
            if os.path.exists(input_source):

                # Load data from file :: raw_data -> normalized_frame
                raw_data = self._load_from_file(input_source)
                normalized_frame = self._raw_to_sequence(self._to_dataframe(raw_data))

                self._data = normalized_frame

                if required_columns:
                    self._apply_required_columns(required_columns)
                self._validate_data(required_columns)
                return self._data

            if file_extension in self._DATAFRAME_READERS or file_extension == self._TEXT_EXTENSION:
                raise FileNotFoundError(f"No file found at {input_source}")

        normalized_frame = self._raw_to_sequence(input_source)
        self._data = normalized_frame
        if required_columns:
            self._apply_required_columns(required_columns)
        self._validate_data(required_columns)
        return self._data

    # return a np.ndarray from a pd.DataFrame
    def to_numpy(self, data: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Convert a pandas DataFrame to a numpy ndarray.

        Args:
            data (pd.DataFrame | np.ndarray): The input DataFrame or ndarray.
        Returns:
            np.ndarray: The converted numpy ndarray.
        """

        validate_data = self._validate_data(dataframe=data)  # type: ignore
        return (
            data.to_numpy(
                copy=False,
                dtype=np.float64,
            )
            if isinstance(data, pd.DataFrame)
            else data
        )

    def _load_from_file(self, filepath: str) -> Any:
        """Read supported files via pandas readers or wrap text files in a DataFrame."""

        try:

            logger.info(f"Loading data from {filepath}")

            file_extension = os.path.splitext(filepath)[1].lower()

            if file_extension in self._DATAFRAME_READERS:
                return self._DATAFRAME_READERS[file_extension](filepath)
            if file_extension == self._TEXT_EXTENSION:
                with open(filepath, "r", encoding="utf-8") as file:
                    return pd.DataFrame({"value": file.readlines()})
            raise ValueError(f"Unsupported file type: {file_extension}")

        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            raise

    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Coerce supported containers to a DataFrame and back-fill missing numeric values."""

        logger.info("converting data to dataframe")

        if isinstance(data, pd.DataFrame):
            logger.info("already dataframe")
            dataframe = data
        elif isinstance(data, (Sequence, bytearray, np.ndarray, dict)):
            dataframe = pd.DataFrame(data)
        else:
            raise TypeError(
                "Unsupported data type for conversion to DataFrame. Supported types: "
                "DataFrame, list, dict, bytearray, numpy ndarray."
            )

        self._fill_missing_values(dataframe)
        return dataframe

    def _raw_to_sequence(self, data: Any) -> pd.DataFrame:
        """Turn raw payloads into sequence-friendly DataFrames while flagging temporal values."""

        logger.info("converting to sequence")

        dataframe, is_nested = self._coerce_dataframe_for_sequence(data)
        normalized_df, contains_sequential = self._normalize_dataframe_entries(dataframe)

        if is_nested:
            logger.info("Detected multi-column input from nested iterables.")

        # If we did not detect temporal data, only inject a timestamp column when the caller
        # explicitly requested one (required_columns includes "timestamp").
        if not contains_sequential:
            if (
                isinstance(normalized_df, pd.DataFrame)
                and "timestamp" in self._required_columns_context
            ):
                normalized_df = self._prepend_timestamp_column(normalized_df)
                logger.info("No temporal data detected; prepended timestamp column.")
            else:
                logger.info("No temporal data detected; continuing without timestamp column")

        return normalized_df

    def _coerce_dataframe_for_sequence(self, data: Any) -> tuple[pd.DataFrame, bool]:
        """Build a DataFrame from the payload and report whether it was multi-column."""

        if isinstance(data, pd.DataFrame):
            df = data
            return df, df.shape[1] > 1

        iterable, is_nested = self._coerce_iterable_for_sequence(data)
        if is_nested:
            df = pd.DataFrame(iterable)
            if df.shape[1] == 0:
                df = pd.DataFrame({"value": [None] * len(iterable)})
                return df, False
            return df, True

        return pd.DataFrame({"value": iterable}), False

    def _coerce_iterable_for_sequence(self, data: Any) -> tuple[list[Any], bool]:
        """Standardize iterables into lists and detect nested structures."""

        if isinstance(data, pd.Series):
            iterable = data.tolist()
        elif isinstance(data, np.ndarray):
            iterable = data.tolist()
        elif isinstance(data, (bytearray, bytes)):
            iterable = list(data)
        elif isinstance(data, list):
            iterable = data[:]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
            iterable = list(data)
        elif isinstance(data, str):
            iterable = [data]
        elif isinstance(data, dict):
            iterable = list(data.values())
        else:
            raise TypeError(
                "Unsupported data type for conversion to sequence. "
                "Supported types: list, tuple, bytearray, bytes, numpy ndarray, or string."
            )

        is_nested = any(
            isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
            for item in iterable
        )
        return iterable, is_nested

    def _normalize_dataframe_entries(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Normalize each value and report whether datetime-like content was encountered."""

        contains_sequential = self._validate_sequence(dataframe)

        def _normalize(value: Any) -> Any:
            nonlocal contains_sequential
            normalized, is_date = self._normalize_datetime_entry(value)
            contains_sequential = contains_sequential or is_date
            return normalized

        if isinstance(dataframe, pd.Series):
            normalized_df = dataframe.map(_normalize)
        else:
            normalized_df = dataframe.apply(_normalize)
        return normalized_df, contains_sequential

    def _prepend_timestamp_column(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Insert a leading timestamp column using the current time for each row."""

        timestamp_values = [datetime.datetime.now().isoformat() for _ in range(len(dataframe))]
        timestamp_series = pd.Series(
            timestamp_values,
            index=dataframe.index,
            name="timestamp",
            dtype="object",
        )
        dataframe = dataframe
        dataframe.insert(0, "timestamp", timestamp_series)
        return dataframe

    def _normalize_datetime_entry(self, value: object) -> tuple[object, bool]:
        """Return ISO-8601 strings for date-like values and note if a conversion occurred."""

        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime().isoformat(), True
        elif isinstance(value, datetime.datetime):
            return value.isoformat(), True
        elif isinstance(value, datetime.date):
            # Avoid double conversion if already datetime
            return datetime.datetime.combine(value, datetime.time()).isoformat(), True
        elif isinstance(value, str):
            try:
                parsed = datetime.datetime.fromisoformat(value)
                return parsed.isoformat(), True
            except ValueError:
                return value, False
        else:
            return value, False

    def _fill_missing_values(self, data: Any) -> None:
        """Apply lightweight mean imputation for numeric containers when practical."""

        logger.info("filling missing values...")

        if isinstance(data, pd.DataFrame):
            data.fillna(data.mean(numeric_only=True), inplace=True)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    mean_value = (
                        pd.Series(value).mean() if pd.Series(value).dtype.kind in "biufc" else None
                    )
                    data[key] = [mean_value if v is None else v for v in value]

        elif isinstance(data, list):
            series = pd.Series(data)
            mean_value = series.mean() if series.dtype.kind in "biufc" else None
            data[:] = [mean_value if v is None else v for v in data]
        elif isinstance(data, pd.Series):
            mean_value = data.mean() if data.dtype.kind in "biufc" else None
            data.fillna(mean_value, inplace=True)
        elif isinstance(data, np.ndarray):
            if np.issubdtype(data.dtype, np.number):
                mean_value = np.nanmean(data)
                inds = np.nonzero(np.isnan(data))
                data[inds] = mean_value
        elif isinstance(data, tuple):
            logger.info("tuples are immutable; no missing value handling applied")
        else:
            logger.info("no missing value handling for type: %s", type(data))

    def _validate_data(self, required_columns: list[str] | None = None) -> bool:
        """Verify structure, NaN patterns, duplicates, and caller-specified schemas."""

        logger.info("validating data...")
        # Check type
        if not isinstance(self._data, pd.DataFrame):
            raise ValueError("data is not a pandas DataFrame.")

        # Check empty
        if self._data.empty:
            raise ValueError("DataFrame is empty")

        # Check for all-NaN columns
        nan_cols = [
            col
            for col in self._data.columns[self._data.isna().all()].tolist()
            if col not in self._appended_required_columns
        ]
        if nan_cols:
            raise ValueError(f"Columns with all NaN values: {nan_cols}")

        if self._appended_required_columns:
            logger.info(
                "Appended required columns with placeholder values: %s",
                sorted(self._appended_required_columns),
            )

        # Check for duplicate columns
        if self._data.columns.duplicated().any():
            # logger.warning("DataFrame has duplicate column names.")
            raise ValueError("DataFrame has duplicate column names.")

        # Check for duplicate rows
        if self._data.duplicated().any():
            logger.info("DataFrame has duplicate rows.")

        # Check for non-numeric columns
        # This appears to think float64 is not numeric, there may be other data types that are numerical that will be marked as not.
        non_numeric = self._data.select_dtypes(exclude=["number"])
        if not non_numeric.empty:
            logger.info(f"Non-numeric columns detected: {non_numeric.columns.tolist()}")

        # Check for required columns (customize as needed)
        if required_columns:
            missing = [col for col in required_columns if col not in self._data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        logger.info("data has been validated")
        return True

    def _apply_required_columns(self, required_columns: list[str] | None) -> None:
        """Align columns to the requested order and append NA placeholders for missing names."""

        if not required_columns:
            return

        # Required column lists must be unique; duplicates would create invalid DataFrames.
        if len(set(required_columns)) != len(required_columns):
            seen: set[str] = set()
            duplicates = [c for c in required_columns if (c in seen) or seen.add(c)]
            raise ValueError(f"Duplicate entries in required_columns: {sorted(set(duplicates))}")

        # When a required schema is provided, trim any surplus columns first.
        # This prevents source files with extra/duplicate columns from
        # leaking into the validated DataFrame.
        if self._data.shape[1] > len(required_columns):
            self._data = self._data.iloc[:, : len(required_columns)].copy()

        existing_cols = list(self._data.columns)
        rename_count = min(len(required_columns), len(existing_cols))
        if rename_count:
            renamed = required_columns[:rename_count] + existing_cols[rename_count:]
            self._data.columns = renamed
            existing_cols = list(self._data.columns)
        for i in range(len(existing_cols), len(required_columns)):
            column = required_columns[i]
            if column not in self._data:
                self._data[column] = pd.NA
                self._appended_required_columns.add(column)

    def _validate_sequence(self, data: Any) -> bool:
        """Placeholder sequence check that only ensures the payload is iterable."""

        # New Algorithm to peek at data set and find a periodic sequence in it

        logger.info("validating sequence...")

        # Treat common containers as acceptable "sequence-like" inputs for our pipeline.
        # A DataFrame is a valid sequence of rows, and numpy arrays are also iterable.
        if isinstance(data, (pd.DataFrame, pd.Series, list, tuple, np.ndarray)):
            return True

        logger.warning("Input data is not a sequence.")
        return False

    def _run_sample_case(
        self, handler: "InputHandler", label: str, payload: Any, required: list[str] | None = None
    ) -> None:
        """Exercise the ingestion path for smoke tests and print quick diagnostics."""

        print(f"\n=== {label} ===")
        frame = handler.input_data(payload, required)
        print(frame)
        print(frame.dtypes)
        print(frame.shape)


if __name__ == "__main__":

    handler = InputHandler.get_instance()

    assert isinstance(handler, InputHandler)
    print("Singleton test passed.")
    assert isinstance(handler.data, pd.DataFrame)
    print("Initial data type test passed.")
    assert isinstance(handler, InputInterface)
    print("Interface conformance test passed.")

    sample_matrix = [
        ("List input", [1, 2, 3], ["value"]),
        ("Tuple input", ((1, 10), (2, 20)), ["first", "second"]),
        ("Sequence input (range)", range(3), ["value"]),
        ("NumPy ndarray input", np.array([[0.1, 0.2], [0.3, 0.4]]), ["x", "y"]),
        ("Dict input", {"feat": [5, 6], "target": [7, 8]}, ["feat", "target"]),
        ("String input", "2024-01-01T12:00:00", None),
        (
            "DataFrame input",
            pd.DataFrame({"feat": [9, 10], "target": [11, 12]}),
            ["feat", "target"],
        ),
        ("Series input", pd.Series([13, 14], name="value"), ["value"]),
    ]
    for label, payload, required in sample_matrix:
        handler._run_sample_case(handler, label, payload, required)

    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easyData.xlsx")
    handler._run_sample_case(handler, "File path input", DATA_PATH, None)
    print("File path test passed.")
