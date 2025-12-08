"""InputHandler singleton to pass a Data object to the Encoder layer.
Implemented as a singleton  layer handler for now, with methods to convert raw data to
DataFrame, sequence, etc.
"""

import datetime
import os
import re
from typing import Any, ClassVar, Dict, Generic, List, Sequence, TypeVar, Union

import numpy as np
import pandas as pd

from psu_capstone.log import logger

T = TypeVar("T")


class InputHandler(Generic[T]):
    """
    Singleton InputHandler class to handle input data.

    """

    __instance: ClassVar["InputHandler"] | None = None

    def __new__(cls) -> "InputHandler":
        """Constructor -- Singleton pattern implementation."""

        if cls.__instance is None:
            cls.__instance = super(InputHandler, cls).__new__(cls)

        return cls.__instance

    def __init__(self):
        """Initialize the InputHandler singleton."""

        # this will have to be more abstract later to handle different data types
        self._data: pd.DataFrame = pd.DataFrame()
        """The input data of any type."""

    @classmethod
    def get_instance(cls) -> "InputHandler":
        """Static access method to get the singleton instance."""

        if cls.__instance is None:
            cls.__instance = InputHandler()
        return cls.__instance

    # Getters, maybe use properties later
    def get_data(self) -> pd.DataFrame:
        """Getter for the data attribute"""

        # more dynamic type checks may be needed here
        return self._data

    def input_data(
        self,
        input_source: Union[str, pd.DataFrame, list, dict, np.ndarray],
        required_columns: list = [],
    ) -> pd.DataFrame:
        """
        Public method to load, convert, and validate data in one step.

        Args:
            input_source: Path to the input file or raw data object.
            required_columns (list): List of required columns for validation.

        Returns:
            pd.DataFrame: The validated DataFrame.

        Raises:
            ValueError, FileNotFoundError, TypeError: On failure.
        """
        raw_data = input_source

        # If input is a string, check if it is a file path
        if isinstance(input_source, str):
            if os.path.exists(input_source):
                raw_data = self._load_from_file(input_source)
            else:
                raise FileNotFoundError(f"The file {input_source} does not exist.")

        # Convert to DataFrame
        self._data = self._to_dataframe(raw_data)

        # Validate
        self._validate_data(required_columns)

        return self._data

    def get_sequence(
        self, data: Sequence | bytearray | bytes | np.ndarray | str | pd.DataFrame
    ) -> Sequence:
        """
        Public method to get the current data as a normalized sequence.

        Returns:
            Sequence: The data as a normalized sequence.
        """

        # more logic needed to handle different data types -- to build sequence datasets
        return self._raw_to_sequence(data)

    def _load_from_file(self, filepath: str) -> Any:
        """Load data from a file with pandas based on file extension.

        Args:
            filepath (str): Path to the input file.

        Returns:
            object: The loaded data as a DataFrame or other supported type.

        Raises:
            ValueError: If the filepath is invalid or unsupported.
        """
        logger.info(f"Loading data from {filepath}")

        loaders = {
            ".csv": pd.read_csv,
            ".xls": pd.read_excel,
            ".xlsx": pd.read_excel,
            ".json": pd.read_json,
            ".parquet": pd.read_parquet,
        }

        file_extension = os.path.splitext(filepath)[1].lower()

        if file_extension in loaders:
            return loaders[file_extension](filepath)
        elif file_extension == ".txt":
            with open(filepath, "r") as file:
                return file.readlines()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert input data to a pandas DataFrame, supporting DataFrame, list, dict, bytearray, or numpy ndarray.

        Args:
            data (Any): The input data to convert.

        Returns:
            pd.DataFrame: The converted DataFrame.

        Raises:
            TypeError: If the input data type is unsupported.
        """
        logger.info("converting data to dataframe")
        if isinstance(data, pd.DataFrame):
            logger.info("already dataframe")
            dataframe = data.copy()
        elif isinstance(data, (Sequence, bytearray, np.ndarray, dict)):
            dataframe = pd.DataFrame(data)
        else:
            raise TypeError(
                "Unsupported data type for conversion to DataFrame. Supported types: "
                "DataFrame, list, dict, bytearray, numpy ndarray."
            )

        self._fill_missing_values(dataframe)
        return dataframe

    def _raw_to_sequence(
        self, data: Sequence | bytearray | bytes | np.ndarray | str | pd.DataFrame
    ) -> Sequence:
        """
        Convert raw data to a normalized sequence list with guaranteed date metadata.

        Args:
            data: The input data, which can be a list, bytearray, bytes, numpy ndarray, or string.

        Returns:
            list: A normalized sequence with at least one date entry.

        Raises:
            TypeError: If the input data type is unsupported.
        """
        logger.info("converting to sequence")
        if isinstance(data, np.ndarray):
            iterable = data.tolist()
        elif isinstance(data, (bytearray, bytes)):
            iterable = list(data)
        elif isinstance(data, list):
            iterable = data[:]
        elif isinstance(data, str):
            iterable = [data]
        else:
            raise TypeError(
                "Unsupported data type for conversion to sequence. "
                "Supported types: list, bytearray, bytes, numpy ndarray, or string."
            )

        sequence = []
        contains_date = False

        for item in iterable:
            normalized, is_date = self._normalize_datetime_entry(item)
            sequence.append(normalized)
            if is_date:
                contains_date = True

        if not contains_date:
            sequence.insert(0, datetime.datetime.now().isoformat())

        return sequence

    def _normalize_datetime_entry(self, value: object) -> tuple[object, bool]:
        """
        Normalize datetime-like values to ISO strings and report detection.

        Args:
            value (object): The value to normalize.

        Returns:
            tuple[object, bool]: The normalized value and a flag indicating if it was a date.
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

    def _fill_missing_values(self, data: pd.DataFrame) -> None:
        """Fill missing values in the input data.

        Args:
            data (pd.DataFrame): The DataFrame in which to fill missing values.
        """

        # Placeholder implementation; actual logic will depend on data type and requirements

        if isinstance(data, pd.DataFrame):
            data.fillna(data.mean(numeric_only=True), inplace=True)

        # Add more cases as needed for different data types

    def _validate_data(self, required_columns: list = []) -> bool:
        """Validate the input data for common issues.

        Args:
            required_columns (list): List of required columns to check for.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        logger.info("validating data...")
        # Check type
        if not isinstance(self._data, pd.DataFrame):
            raise ValueError("data is not a pandas DataFrame.")

        # Check empty
        if self._data.empty:
            raise ValueError("DataFrame is empty")

        # Check for all-NaN columns
        nan_cols = self._data.columns[self._data.isna().all()].tolist()
        if nan_cols:
            raise ValueError(f"Columns with all NaN values: {nan_cols}")

        # Check for duplicate columns
        if self._data.columns.duplicated().any():
            raise ValueError("DataFrame has duplicate column names.")

        # Check for duplicate rows
        if self._data.duplicated().any():
            logger.warning("DataFrame has duplicate rows.")

        # Check for non-numeric columns
        non_numeric = self._data.select_dtypes(exclude=["number"])
        if not non_numeric.empty:
            logger.info(f"Non-numeric columns detected: {non_numeric.columns.tolist()}")

        # Check for required columns (customize as needed)
        if required_columns is not None:
            missing = [col for col in required_columns if col not in self._data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        logger.info("data has been validated")
        return True


if __name__ == "__main__":

    handler = InputHandler.get_instance()
    sample = {"feature": [1, 2], "target": [3, 4]}
    frame = handler.input_data(sample, required_columns=["feature"])
    assert frame.shape == (2, 2)
    assert list(frame.columns) == ["feature", "target"]
    print("InputHandler input_data smoke test passed.")
