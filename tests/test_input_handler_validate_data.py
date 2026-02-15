"""
Test suite for the InputHandler class.

This suite verifies the following aspects:
    - Input normalization: Ensures various input types (list of dicts, bytearray, scalar)
        are correctly converted into a sequence of records.
    - Required columns: Missing columns are added and ordering is enforced.
    - Missing values: Rows with missing values are dropped during normalization.
    - Duplicate columns: Duplicates are removed during processing.

Test Flow:
    1. Each test provides a different form of input to InputHandler.input_data().
    2. The input is normalized via _process_dataframe and related helpers.
    3. The test asserts on the final state or output of InputHandler.
"""

import pandas as pd
import pytest

from psu_capstone.input_layer.input_handler import InputHandler


@pytest.fixture
def handler():
    h = InputHandler()
    return h


def test_validate_data_valid(handler):
    handler.input_data(
        [
            {"id": 1, "timestamp": "2024-01-01", "value": 10},
            {"id": 2, "timestamp": "2024-01-02", "value": 20},
            {"id": 3, "timestamp": "2024-01-03", "value": 30},
        ],
        required_columns=["id", "timestamp", "value"],
    )
    records = handler.to_records()

    assert records == [
        {"id": 1, "timestamp": "2024-01-01T00:00:00", "value": 10},
        {"id": 2, "timestamp": "2024-01-02T00:00:00", "value": 20},
        {"id": 3, "timestamp": "2024-01-03T00:00:00", "value": 30},
    ]
    assert handler._columns == ["id", "timestamp", "value"]


def test_validate_data_missing_required_columns(handler):
    handler.input_data(
        [{"id": 1, "value": 10}, {"id": 2, "value": 20}, {"id": 3, "value": 30}],
        required_columns=["id", "timestamp", "value"],
    )
    records = handler.to_records()

    assert records == []
    assert handler._columns == ["id", "timestamp", "value"]


def test_validate_data_empty_dataframe(handler):
    handler.input_data([], required_columns=["id"])
    records = handler.to_records()
    assert records == []
    assert handler._columns == []


def test_validate_data_all_nan_column(handler):
    handler.input_data(
        [
            {"id": 1, "timestamp": None, "value": 10},
            {"id": 2, "timestamp": None, "value": 20},
            {"id": 3, "timestamp": None, "value": 30},
        ],
        required_columns=["id", "timestamp", "value"],
    )
    records = handler.to_records()

    assert records == []


def test_validate_data_duplicate_columns(handler):
    df = pd.DataFrame([[1, 10, 20]], columns=["id", "value", "value"])
    handler.input_data(df, required_columns=["id", "value"])
    records = handler.to_records()

    assert records == [{"id": 1, "value": 10}]
    assert handler._columns == ["id", "value"]


def test_input_data_sequence_of_scalars(handler):
    handler.input_data([1, 2, 3], required_columns=["value"])
    records = handler.to_records()
    assert records == [{"value": 1}, {"value": 2}, {"value": 3}]


def test_input_data_bytearray(handler):
    byte_data = bytearray(b"id,timestamp,value\n1,2024-01-01,10\n2,2024-01-02,20")
    handler.input_data(byte_data, required_columns=["id", "timestamp", "value"])
    result = handler.to_records()
    assert result == [
        {"id": "1", "timestamp": "2024-01-01T00:00:00", "value": "10"},
        {"id": "2", "timestamp": "2024-01-02T00:00:00", "value": "20"},
    ]


def test_input_data_scalar(handler):
    scalar_data = 42
    handler.input_data(scalar_data, required_columns=["value"])
    result = handler.to_records()
    assert result == [{"value": 42}]
    assert handler._data == {"value": [42]}
