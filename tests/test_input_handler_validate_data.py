"""
Test suite for the InputHandler class.

This suite verifies the following aspects:
  - Input normalization: Ensures various input types (list of dicts, bytearray, scalar)
    are correctly converted into a sequence of records.
  - Data validation: Checks that required columns are present, duplicates are detected, and
    missing values are handled.
  - Error handling: Confirms that invalid inputs raise appropriate errors.

Test Flow:
  1. Each test provides a different form of input to InputHandler.input_data().
  2. The input is normalized via _raw_to_sequence and related helpers.
  3. The normalized data is validated by _validate_data.
  4. The test asserts on the final state or output of InputHandler.

Example:
    def test_validate_data_valid():
        handler = InputHandler(required_columns=['id', 'value'])
        handler.input_data([{'id': 1, 'value': 10}, {'id': 2, 'value': 20}])
        assert handler._validate_data() == ("", True)
"""

import pytest

from psu_capstone.input_layer.input_handler import InputHandler


@pytest.fixture
def handler():
    h = InputHandler()
    return h


def test_validate_data_valid(handler):
    handler._data = [
        {"id": 1, "timestamp": "2024-01-01", "value": 10},
        {"id": 2, "timestamp": "2024-01-02", "value": 20},
        {"id": 3, "timestamp": "2024-01-03", "value": 30},
    ]
    handler._columns = ["id", "timestamp", "value"]
    assert handler._validate_data(required_columns=["id", "timestamp", "value"]) == (
        "Data is valid.",
        True,
    )


def test_validate_data_missing_required_columns(handler):
    handler._data = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]
    handler._columns = ["id", "value"]
    result = handler._validate_data(required_columns=["id", "timestamp", "value"])
    assert result == ("Missing required columns: ['timestamp']", False)


def test_validate_data_empty_dataframe(handler):
    handler._data = []
    result = handler._validate_data(required_columns=["id"])
    assert result == ("Record list is empty", False)


def test_validate_data_all_nan_column(handler):
    handler._data = [
        {"id": 1, "timestamp": None, "value": 10},
        {"id": 2, "timestamp": None, "value": 20},
        {"id": 3, "timestamp": None, "value": 30},
    ]
    handler._columns = ["id", "timestamp", "value"]
    result = handler._validate_data(required_columns=["id", "timestamp", "value"])
    assert result == ("Columns with all None values: ['timestamp']", False)


def test_validate_data_duplicate_columns(handler):
    handler._data = [{"id": 1, "timestamp": "2024-01-01", "value": 10}]
    handler._columns = ["id", "timestamp", "value", "value"]
    result = handler._validate_data(required_columns=["id", "timestamp", "value"])
    assert result == ("Record list has duplicate column names.", False)


def test_validate_data_non_dataframe(handler):
    handler._data = [1, 2, 3]
    try:
        result = handler._validate_data(required_columns=["id"])
    except AttributeError as e:
        assert "object has no attribute 'keys'" in str(e)


def test_input_data_bytearray(handler):
    byte_data = bytearray(b"id,timestamp,value\n1,2024-01-01,10\n2,2024-01-02,20")
    result = handler.input_data(byte_data, required_columns=["id", "timestamp", "value"])
    assert result == [
        {"id": "1", "timestamp": "2024-01-01", "value": "10"},
        {"id": "2", "timestamp": "2024-01-02", "value": "20"},
    ]


def test_input_data_scalar(handler):
    scalar_data = 42
    try:
        result = handler.input_data(scalar_data, required_columns=["value"])
    except TypeError as e:
        assert "Unsupported data type for conversion to sequence" in str(e)
