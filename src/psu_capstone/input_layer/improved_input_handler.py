"""InputHandler singleton to pass a Data object to the Encoder layer.
Implemented as a singleton  layer handler for now, with methods to convert raw data to
DataFrame, sequence, etc.

input_data call paths:
1. File path input -> _load_from_file -> _to_dataframe -> _raw_to_sequence -> (_apply_required_columns) -> _validate_data -> _data
2. Non-file input -> _raw_to_sequence -> (_apply_required_columns) -> _validate_data -> _data
"""

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
    """
    Canonical entry point for shuttling arbitrary raw inputs into normalized pandas DataFrames.

    The handler is implemented as a singleton so downstream layers can share a consistent view of
    the most recently ingested dataset (`data`) and its normalized, timestamp-aware representation
    (`sequence`). All public APIs make sure that:

    * Any supported payload (Python iterables, numpy arrays, pandas objects, strings, files, etc.)
      is converted into a DataFrame.
    * Missing values are filled, and required columns are appended or renamed as requested.
    * Validation runs after every ingestion so consumers always receive structurally sound frames.
    """

    __instance: ClassVar[InputHandler | None] = None
    __interface: ClassVar[Any]

    _DATAFRAME_READERS: ClassVar[dict[str, Callable[[str], pd.DataFrame]]] = {
        ".csv": pd.read_csv,
        ".xls": pd.read_excel,
        ".xlsx": pd.read_excel,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
    }
    _TEXT_EXTENSION: ClassVar[str] = ".txt"

    def __new__(cls) -> "InputHandler":
        """Constructor -- Singleton pattern implementation."""

        if cls.__instance is None:
            cls.__instance = super(InputHandler, cls).__new__(cls)

        return cls.__instance

    def __init__(self, data: Any = None) -> None:
        """
        Initialize the singleton with an optional eager payload.

        Args:
            data: Optional raw payload to load immediately. Any supported type is accepted. When
                  omitted, the handler starts empty and will populate itself on the next `input_data`invocation.
        """

        self._data: pd.DataFrame = (
            self._raw_to_sequence(data) if data is not None else pd.DataFrame()
        )
        """The normalized input data."""

        self._appended_required_columns: set[str] = set()
        """Set of required columns that were auto-appended with placeholder values."""

    @classmethod
    def get_instance(cls) -> "InputHandler":
        """Static access method to get the singleton instance."""

        if cls.__instance is None:
            cls.__instance = InputHandler()
        return cls.__instance

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @property
    def interface(self) -> Any:
        return type(self).__interface

    @interface.setter
    def interface(self, interface: Any) -> None:
        type(self).__interface = interface

    def input_data(
        self, input_source: Any, required_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Ingest a payload, normalize it, optionally enforce column names, and validate the result.

        The method covers both disk-based and in-memory inputs. When `input_source` is a string,
        the handler first checks the filesystem; if a matching path exists it is loaded with the
        appropriate pandas reader. Otherwise the string is treated as a literal payload (e.g.,
        ISO timestamps). Non-string payloads are passed directly into the normalization pipeline.

        Args:
            input_source: Raw payload. Can be a path (str), iterable, numpy array, pandas object,
                          etc.
            required_columns: Optional ordered list of column names. Existing columns will berenamed in-place to match the
            provided order; surplus names are
            appended as placeholder columns filled with NA values.

        Returns:
            pd.DataFrame: A validated DataFrame that reflects both the normalized sequence and any
            column requirements.

        Raises:
            FileNotFoundError: If a string path is supplied but does not exist.
            TypeError: When the payload type cannot be coerced into a DataFrame.
            ValueError: If validation fails (missing required columns, all-NaN columns, etc.).
        """

        self._appended_required_columns.clear()

        # If input is a string, check if it is a file path
        if isinstance(input_source, os.PathLike):
            raise TypeError("Path-like objects must be converted to strings before ingestion.")

        if isinstance(input_source, str):
            file_extension = os.path.splitext(input_source)[1].lower()
            if os.path.exists(input_source):
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

    def _load_from_file(self, filepath: str) -> Any:
        """Load data from a file with pandas based on file extension.

        Args:
            filepath (str): Path to the input file.

        Returns:
            object: The loaded data as a DataFrame or other supported type.

        Raises:
            ValueError: If the filepath is invalid or unsupported.
        """

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
        """
        Convert arbitrary raw input into a timestamp-aware DataFrame.

        This method handles shape detection, coerces nested iterables into multi-column DataFrames,
        normalizes datetime-like values into ISO strings, and injects a dedicated `timestamp` column
        when no temporal metadata is present.

        Args:
            data: Supported raw payload (see `input_data`).

        Returns:
            pd.DataFrame: Normalized data with at least one timestamp column.
        """

        logger.info("converting to sequence")

        dataframe, is_nested = self._coerce_dataframe_for_sequence(data)
        normalized_df, contains_sequential = self._normalize_dataframe_entries(dataframe)

        if is_nested:
            logger.info("Detected multi-column input from nested iterables.")

        is_seq, details = self._detect_sequential_dataset(normalized_df)
        logger.info("Sequential detection: %s", details.get("reason"))

        # If we did not detect temporal data, only inject a timestamp column when the caller
        # explicitly requested one (required_columns includes "timestamp").
        if not contains_sequential:
            logger.info("No temporal data detected;")

        return normalized_df

    # Still very much so a work in progress. - Alex
    def _detect_sequential_dataset(self, df: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        details: dict[str, Any] = {
            "passed": [],
            "failed": [],
            "reason": None,
        }

        # empty dataframe cannot be empty
        if df.empty:
            details["reason"] = "empty"
            return False, details

        lowered = {c.lower(): c for c in df.columns}
        time_name_candidates = ["timestamp", "time", "date", "datetime"]
        for key in time_name_candidates:
            if key in lowered:
                col = lowered[key]

                s = pd.to_datetime(df[col], errors="coerce", utc=False)
                valid = s.notna().mean() if len(s) else 0.0
                if valid >= 0.9:
                    # monotonic check on valid values only
                    sv = s[s.notna()]
                    if len(sv) >= 3 and (sv.is_monotonic_increasing or sv.is_monotonic_decreasing):
                        details["passed"].append(f"explicit_time_column:{col}")

        # allow datetime-like columns even if not named something like date or time
        for col in df.columns:
            if df[col].dtype == "object" or isinstance(df[col].dtype, pd.StringDtype):
                s = pd.to_datetime(df[col], errors="coerce", utc=False)
                valid = s.notna().mean()
                if valid >= 0.9:
                    sv = s[s.notna()]
                    if len(sv) >= 3 and (sv.is_monotonic_increasing or sv.is_monotonic_decreasing):
                        details["passed"].append(f"datetime_like_column:{col}")

        num = df.select_dtypes(include=["number"]).copy()
        if num.shape[1] == 0:
            details["reason"] = "no_numeric_columns"
            return False, details

        n = len(num)
        if n < 8:
            details["reason"] = "too_short_for_detection"
            return False, details

        AUTOCORR_THRESH = 0.65
        FFT_PEAK_RATIO_THRESH = 0.30

        for col in num.columns:
            x = num[col].to_numpy(dtype=float, copy=False)
            x = x[np.isfinite(x)]
            if len(x) < 8:
                continue

            # Build shifted vectors to align each value
            x0, x1 = x[:-1], x[1:]

            print(x0)
            print(np.std(x0))

            print(x1)
            print(np.std(x1))

            # Check for constant data
            # Example of constant data:
            # x0 = [5, 5, 5] and x1 = [5, 5, 5]
            # Would be constant so just set ac1 to 0.0 instead of NaN
            if np.std(x0) > 0 and np.std(x1) > 0:
                # Not constant so ac1 = lag-1 autocorrelation
                ac1 = float(np.corrcoef(x0, x1)[0, 1])
            else:
                # Because it is constant, set ac1 to 0.0
                ac1 = 0.0

            # FFT analysis from here

            # Remove the mean because the mean would create a giant spike at frequency 0
            y = x - np.mean(x)

            # Find the FFT
            fft = np.fft.rfft(y)

            # Tells us how strong each frequency is
            power = fft.real**2 + fft.imag**2

            # Get the ratio
            if len(power) > 1:
                power_no_dc = power[1:]
                peak = float(np.max(power_no_dc))
                total = float(np.sum(power_no_dc)) + 1e-12
                peak_ratio = peak / total
            else:
                peak_ratio = 0.0

            print("Hello")
            print(ac1)
            print(peak_ratio)

            # Either condition holds either strong lag-1 autocorrelation or strong periodic frequency
            passed = (abs(ac1) >= AUTOCORR_THRESH) or (peak_ratio >= FFT_PEAK_RATIO_THRESH)

            if passed:
                print(passed)
                print(num[col])
                details["passed"].append((col, {"ac1": ac1, "peak_ratio": peak_ratio}))
            else:
                print(passed)
                print(num[col])
                details["failed"].append((col, {"ac1": ac1, "peak_ratio": peak_ratio}))

        if details["failed"]:
            details["reason"] = "no_sequential_signal_detected"
            return False, details

        details["reason"] = "Detecting_sequential_signals"
        return True, details

    def _should_ignore_for_sequential_check(self, s: pd.Series) -> bool:
        non_null = s.dropna()
        if len(non_null) == 0:
            print(f"ignore:{s.name}: all NaN")
            return True

        return False

    def _coerce_dataframe_for_sequence(self, data: Any) -> tuple[pd.DataFrame, bool]:
        """
        Coerce raw payloads into DataFrames while tracking whether the data was inherently
        multi-dimensional.

        Returns:
            Tuple[pd.DataFrame, bool]: The coerced DataFrame and a flag indicating whether the
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
        """
        Convert iterable-friendly inputs into plain Python lists so downstream DataFrame creation
        behaves consistently.

        Supported types include pandas Series, numpy arrays, bytes/bytearrays, generic Sequences,
        dictionaries (values only), and strings (wrapped in a single-element list).
        """
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
        """
        Apply `_normalize_datetime_entry` element-wise across the DataFrame/Series and flag whether
        any datetime-like content was discovered.

        Returns:
            Tuple[pd.DataFrame, bool]: The normalized DataFrame/Series and a boolean indicating
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

    def _normalize_datetime_entry(self, value: object) -> tuple[object, bool]:
        """
        Normalize datetime-like scalars into ISO-8601 strings while signaling detection to callers.

        Args:
            value: Scalar under inspection.

        Returns:
            Tuple[object, bool]: Possibly-transformed value and whether a datetime was detected.
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
        """
        Impute missing values in-place for several container types.

        * DataFrames: numeric columns receive column means.
        * Dicts of lists: numeric lists receive mean-substitution; non-numerics are left untouched.
        * Flat lists: numeric series receive mean substitution analogous to DataFrame columns.
        """
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
        """
        Run a battery of structural and semantic checks on the current DataFrame.

        Checks include emptiness, all-NaN columns (excluding freshly appended required columns),
        duplicate columns/rows, mixed dtypes, and presence of caller-specified required columns.

        Args:
            required_columns: Optional list of columns that must exist post-normalization.

        Returns:
            bool: True when the frame passes all checks.

        Raises:
            ValueError: When any invariant is violated.
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

    def _apply_required_columns(self, required_columns: list[str] | None) -> None:
        """
        Rename existing columns to align with the requested order and append any missing names.

        The first `len(existing_cols)` entries in `required_columns` replace the current column
        names, preserving positional intent. Surplus names are added as NA-filled columns and
        tracked via `_appended_required_columns` so validation can treat them as intentional
        placeholders.
        """
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
        """
        Validate that the input data can be treated as a sequence.

        Args:
            data: The input data to validate.
        """
        # New Algorithm to peek at data set and find a periodic sequence in it

        logger.info("validating sequence...")

        # Treat common containers as acceptable "sequence-like" inputs for our pipeline.
        # A DataFrame is a valid sequence of rows, and numpy arrays are also iterable.
        if isinstance(data, (pd.DataFrame, pd.Series, list, tuple, np.ndarray)):
            return True

        logger.warning("Input data is not a sequence.")
        return False
