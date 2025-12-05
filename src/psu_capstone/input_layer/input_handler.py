"""InputHandler singleton to pass a Data object to the Encoder layer.
Implemented as a singleton  layer handler for now, with methods to convert raw data to
DataFrame, sequence, etc.
"""

import datetime
import os
from typing import Union

import numpy as np
import pandas as pd

from psu_capstone.log import logger


class InputHandler:
    """
    Singleton InputHandler class to handle input data.

    """

    __instance = None

    def __new__(cls) -> "InputHandler":
        """Constructor -- Singleton pattern implementation."""

        if cls.__instance is None:
            cls.__instance = super(InputHandler, cls).__new__(cls)

        return cls.__instance

    def __init__(self):
        """Initialize the InputHandler singleton."""

        self._instance = None
        """The singleton instance."""

        # this will have to be more abstract later to handle different data types
        self._data = pd.DataFrame()
        """The input data of any type."""

    # Getters, maybe use properties later
    def get_data(self) -> pd.DataFrame:
        """Getter for the data attribute"""

        # more dynamic type checks may be needed here
        assert isinstance(self._data, pd.DataFrame)
        return pd.DataFrame(self._data)

    def input_data(self, filepath: str, required_columns: list) -> pd.DataFrame:
        """
        Public method to load, convert, and validate data in one step.

        Args:
            filepath (str): Path to the input file.
            required_columns (list): List of required columns for validation.

        Returns:
            pd.DataFrame: The validated DataFrame.

        Raises:
            ValueError, FileNotFoundError, TypeError: On failure.
        """
        self._load_raw_data(filepath, required_columns)
        # Optionally, you could call _to_dataframe here if needed
        return self.get_data()

    def get_sequence(
        self, data: Union[list, bytearray, bytes, np.ndarray, str, pd.DataFrame]
    ) -> list:
        """
        Public method to get the current data as a normalized sequence.

        Returns:
            list: The data as a normalized sequence.
        """
        # more type checks may be needed here
        assert isinstance(data, pd.DataFrame)
        return self._raw_to_sequence(data.values.tolist())

    def _load_raw_data(self, filepath: str, required_columns: list) -> object:
        """Load data from a file with pandas based on file extension.
        This will automatically create a dataframe.

        Args:
            filepath (str): Path to the input file.
            required_columns (list): List of required columns for validation.

        Returns:
            object: The loaded data as a DataFrame or other supported type.

        Raises:
            ValueError: If the filepath is invalid or unsupported.
            FileNotFoundError: If the file does not exist.
            TypeError: If called on an invalid instance.
        """
        logger.info("handling input data")
        if not isinstance(filepath, str) or not filepath:
            raise ValueError("Filepath must be a non-empty string.")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")
        if not isinstance(self, InputHandler):
            raise TypeError("load_raw_data must be called on an InputHandler instance.")

        loaders = {
            ".csv": pd.read_csv,
            ".xls": pd.read_excel,
            ".xlsx": pd.read_excel,
            ".json": pd.read_json,
        }

        file_extension = os.path.splitext(filepath)[1].lower()
        print(f"Loading file: {file_extension} {filepath}")

        if file_extension in loaders:
            self._data = loaders[file_extension](filepath)
        elif file_extension == ".txt":
            with open(filepath, "r") as file:
                self._data = file.readlines()
            self._data = pd.DataFrame(self._data)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        self._validate_data(required_columns=required_columns)
        logger.info("data converted to dataframe")
        return self._data

    def _to_dataframe(self, data: object) -> pd.DataFrame:
        """Convert input data to a pandas DataFrame, supporting DataFrame, list, bytearray, or numpy ndarray.

        Args:
            data (object): The input data to convert.

        Returns:
            pd.DataFrame: The converted DataFrame.

        Raises:
            TypeError: If the input data type is unsupported.
        """
        logger.info("converting data to dataframe")
        if isinstance(data, pd.DataFrame):
            logger.info("already dataframe")
            dataframe = data
        elif isinstance(data, (list, bytearray, np.ndarray)):
            dataframe = pd.DataFrame(data)
        else:
            raise TypeError(
                "Unsupported data type for conversion to DataFrame. Supported types: DataFrame, list, bytearray, numpy ndarray."
            )

        self._fill_missing_values(dataframe)
        return dataframe

    def _raw_to_sequence(self, data: Union[list, bytearray, bytes, np.ndarray, str]) -> list:
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

    def _validate_data(self, required_columns) -> bool:
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
            print("DataFrame has duplicate rows.")

        # Check for non-numeric columns
        non_numeric = self._data.select_dtypes(exclude=["number"])
        if not non_numeric.empty:
            print(f"Non-numeric columns detected: {non_numeric.columns.tolist()}")

        # Check for required columns (customize as needed)
        if required_columns is not None:
            missing = [col for col in required_columns if col not in self._data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        logger.info("data has been validated")
        return True
