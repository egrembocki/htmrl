"""Driver code to test InputHandler functionality."""

import datetime as dt
import os
import pathlib as path
from typing import Union

import numpy as np
import pandas as pd

import psu_capstone.input_layer.input_handler as ih

ROOT_PATH = path.Path(__file__).parent.parent.parent.parent

DATA_PATH = ROOT_PATH / "Data"

"""Driver code to test InputHandler functionality."""

# test harness

test_inputs: dict[str, Union[list, bytearray, bytes, np.ndarray]] = {
    # 1) Simple list with a datetime -> should keep ISO date string at index 0
    "list_with_datetime": [
        dt.datetime(2025, 1, 2, 3, 4, 5),
        42,
        "foo",
    ],
    # 2) List with date only -> should convert to datetime at midnight
    "list_with_date": [
        dt.date(2025, 1, 2),
        3.14,
        "bar",
    ],
    # 3) List with pandas Timestamp
    "list_with_timestamp": [
        pd.Timestamp("2025-01-02T03:04:05"),
        {"a": 1},
    ],
    # 4) List with ISO date string -> should parse as datetime
    "list_with_iso_string": [
        "2025-01-02T03:04:05",
        "2024-12-31",
        99,
    ],
    # 5) List with NON-ISO string -> should not be treated as date
    "list_with_non_iso_string": [
        "not-a-date",
        "2025/01/02",  # fails fromisoformat
        123,
    ],
    # 6) List with no date-like values -> should prepend now() ISO string
    "list_without_date": [
        1,
        2,
        3,
        "abc",
        b"\x01\x02",
    ],
    # 7) Numpy array of plain values -> uses np.ndarray branch, no dates
    "numpy_array_no_dates": np.array([10, 20, 30]),
    # 8) Numpy array with an ISO string inside -> one date, rest unchanged
    "numpy_array_with_iso": np.array(["2025-01-02T03:04:05", "x", "y"], dtype=object),
    # 9) Bytearray input -> hits bytearray/bytes branch, values are ints 0-255
    "bytearray_data": bytearray(b"\x01\x02\x03\x04"),
    # 10) Bytes input -> same branch as bytearray
    "bytes_data": b"\x10\x20\x30",
    # 11) Mixed list that includes several datetime-like forms
    "mixed_many_dates": [
        "2025-01-02T03:04:05",
        dt.date(2025, 1, 3),
        pd.Timestamp("2025-01-04"),
        "not-a-date",
        0,
    ],
}


def main():
    """Main function to demonstrate InputHandler usage."""
    # Create an instance of InputHandler
    handler = ih.InputHandler()

    # test harness for loop

    for test_name, test_data in test_inputs.items():
        try:
            print(f"--- Test: {test_name} ---")
            sequence = handler.get_sequence(test_data)
            print("Resulting Sequence:", sequence)
            print()
        except Exception as e:
            print(f"Error during test {test_name}: {e}")
            print()

    # Set some raw data, will need more abstraction later
    data_set = handler.input_data(
        os.path.join(DATA_PATH, "concat_ESData.xlsx"), required_columns=["Date"]
    )

    print("Raw Data Loaded.", type(data_set), "\n", DATA_PATH)

    # Explicitly convert raw data to DataFrame
    # data_frame = handler.to_dataframe(data_set)

    print("Data Frame Created.", type(data_set), "\n", data_set.head())
    # print("Data Validation:", handler.validate_data())
    print(data_set.info())


if __name__ == "__main__":

    main()
