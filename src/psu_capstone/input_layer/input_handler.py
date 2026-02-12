"""InputHandler for HTM pipeline. Build a Singleton handler that normalizes varied payloads into Encoder-ready DataFrames."""

from __future__ import annotations

import datetime
import os
from collections.abc import Sequence
from typing import Any, Callable, ClassVar, cast

import numpy as np
import pandas as pd

from psu_capstone.log import logger


class InputHandler:
    """Build an input data manager."""

    __instance: ClassVar[InputHandler | None] = None
    """Singleton instance"""

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
        """Constructor :: called after new().

        Args:
            data (Any, optional): Initial data to load into the handler. Defaults to None.

        """

        self._data: Any = None if data is None else self.input_data(data)
        """The processed input data -> encoder-safe."""

        self._appended_required_columns: set[str] = (
            set()
        )  # no repeat columns allowed in required_columns
        """Set of required columns."""

        # This context is necessary to determine whether to inject a timestamp column when no temporal data is detected.
        self._required_columns_context: list[str] = []
        """Most-recent required_columns list passed to input_data (used for timestamp behavior)"""

    @classmethod
    def get_instance(cls) -> "InputHandler":
        """Static access method to get the singleton instance."""

        if cls.__instance is None:
            cls.__instance = InputHandler()
        return cls.__instance

    @property
    def data(self) -> Any:
        if self._data is None:
            logger.warning("Data has not been set yet; returning None.")
        elif isinstance(self._data, pd.DataFrame):
            logger.info(
                f"Data is a DataFrame with shape {self._data.shape} and columns {self._data.columns.tolist()}"
            )
            self._data = cast(Any, self.to_numpy(self._data))
            logger.info(
                f"Data converted to numpy ndarray with shape {self._data.shape} and dtype {self._data.dtype}"
            )
        elif isinstance(self._data, np.ndarray):
            logger.info(
                f"Data is a numpy ndarray with shape {self._data.shape} and dtype {self._data.dtype}"
            )
            self._data = cast(Any, self._data)
        elif isinstance(self._data, (list, tuple)):
            logger.info(f"Data is a {type(self._data).__name__} with length {len(self._data)}")
            self._data = cast(Any, self._data)
        elif isinstance(self._data, dict):
            logger.info(f"Data is a dict with keys {list(self._data.keys())}")
            self._keys = list(self._data.keys())
            self._data = list(self._data.values())
            self._data = cast(Any, self._data)
        elif isinstance(self._data, str):
            logger.info(f"Data is a string with length {len(self._data)}")
            self._data = cast(Any, self._data)
        elif isinstance(self._data, (bytearray, bytes)):
            logger.info(f"Data is a {type(self._data).__name__} with length byte {len(self._data)}")
            self._data = cast(Any, self._data)
        elif isinstance(self._data, pd.Series):
            logger.info(
                f"Data is a Series with length {len(self._data)} and name {self._data.name}"
            )
            self._data = list(self._data)
            logger.info(f"Data converted to list with length {len(self._data)}")
            self._data = cast(Any, self._data)
        elif isinstance(self._data, (int, float, bool)):
            logger.info(
                f"Data is a scalar of type {type(self._data).__name__} with value {self._data}"
            )
            self._data = cast(Any, self._data)
        return self._data

    @data.setter
    def data(self, data: Any) -> None:
        self._data = data

    def input_data(self, input_source: Any, columns: list[str] | None = None) -> Any:
        """Inputing data into the handler starts here

        Args:
            input_source (Any): The raw input data, which can be a file path, DataFrame, Series, list, dict, numpy array, string, or byte sequence.
            columns (list[str], optional): A list of column names that can be added as dict keys.
        Returns:
            Any: The processed data, typically as an array ready for encoding.

        Raises:
            FileNotFoundError: If a file path is provided but the file does not exist.
            ValueError: If data validation fails.


        """

        self._appended_required_columns.clear()

        # Check if input is a file path and convert to string if it's a PathLike object. This allows the handler to accept both string paths and Path objects from pathlib.
        if isinstance(input_source, os.PathLike):
            input_source = os.fspath(input_source)

            file = self.check_file_path(input_source)

        elif isinstance(input_source, str):

            file = self.check_file_path(input_source)

            if file:

                # Load data from file :: raw_data -> normalized_frame
                raw_data = self._load_from_file(input_source)
                normalized_frame = self._raw_to_sequence(self._to_dataframe(raw_data))

                self._data = normalized_frame

                if columns:
                    self._apply_required_columns(columns)
                # self._validate_data(columns)
                return self._data

            # if file_extension in self._DATAFRAME_READERS or file_extension == self._TEXT_EXTENSION:
            # raise FileNotFoundError(f"No file found at {input_source}")

        normalized_frame = self._raw_to_sequence(input_source)
        self._data = normalized_frame
        if columns:
            self._apply_required_columns(columns)
        self._validate_data(columns)
        return self._data

    def check_file_path(self, input_string: str) -> bool:
        """Check if a string is a valid file path and has a supported extension.

        Args:
            input_string (str): The string to check.
        Returns:
            bool: True if the string is a valid file path with a supported extension, False otherwise.
        """
        file_extension = os.path.splitext(input_string)[1].lower()
        if os.path.exists(input_string) and (
            file_extension in self._DATAFRAME_READERS or file_extension == self._TEXT_EXTENSION
        ):
            return True
        return False

    # return a np.ndarray from a pd.DataFrame
    def to_numpy(self, data: pd.DataFrame) -> np.ndarray:
        """Convert a pandas DataFrame to a numpy ndarray.

        Args:
            data (pd.DataFrame): The input DataFrame.
        Returns:
            np.ndarray: The converted numpy ndarray.
        """
        # safety checks and conversions to ensure the DataFrame is in a suitable format for conversion to a numpy array. This includes handling duplicates, coercing non-numeric values to NaN, and filling missing values with the mean of each column before performing the conversion.
        data = data.drop_duplicates().reset_index(drop=True)
        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.fillna(data.mean(numeric_only=True))

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
        """
        Coerce raw payloads into DataFrames while tracking whether the data was inherently
        multi-dimensional.

        Returns:
            tuple[pd.DataFrame, bool]: The coerced DataFrame and a flag indicating whether the
            input provided multiple columns (either originally or due to nested iterables).
        """

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
        """Normalize each value and report whether datetime-like content was encountered.
        Returns:
            tuple[pd.DataFrame, bool]: The normalized DataFrame/Series and a boolean indicating
            presence of temporal data.
        """
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
        """Return ISO-8601 strings for date-like values and note if a conversion occurred.

        Args:
            value: Scalar under inspection.

        Returns:
            tuple[object, bool]: Possibly-transformed value and whether a datetime was detected.
        """
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

    def _validate_data(self, required_columns: list[str] = []) -> bool:
        """Verify structure, NaN patterns, duplicates, and caller-specified schemas.

        #TODO: break each of these checks into separate helper methods for better readability and maintainability, and to allow for more granular error reporting. For example, we could have _check_type(), _check_empty(), _check_nan_columns(), _check_duplicates(), and _check_required_columns() methods that are called in sequence from _validate_data().

        """

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

    def _apply_required_columns(self, required_columns: list[str] = []) -> None:
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
