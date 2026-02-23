"""
tests.test_input_handler_validate_data

Test suite for InputHandler data validation functionality.

Validates that InputHandler correctly validates loaded data for consistency, type correctness,
and required field presence. Tests ensure the validation step correctly identifies:
- Missing or null values in required columns
- Data type mismatches and format errors
- Required column presence and completeness
- Data quality issues that would impact downstream encoding

These tests ensure the input data pipeline correctly validates data before encoder processing,
catching issues early in the pipeline.
"""

import pandas as pd
import pytest

from psu_capstone.input_layer.input_handler import InputHandler


def _columnar_to_records(data: dict[str, list[object]]) -> list[dict[str, object]]:
    if not data:
        return []
    lengths = {len(values) for values in data.values()}
    if not lengths:
        return []
    if len(lengths) != 1:
        raise AssertionError("Columnar data has inconsistent lengths.")
    row_count = lengths.pop()
    return [{column: data[column][idx] for column in data} for idx in range(row_count)]


@pytest.fixture
def handler():
    h = InputHandler()
    return h


def test_validate_data_valid(handler):
    data = handler.input_data(
        [
            {"id": 1, "timestamp": "2024-01-01", "value": 10},
            {"id": 2, "timestamp": "2024-01-02", "value": 20},
            {"id": 3, "timestamp": "2024-01-03", "value": 30},
        ],
        required_columns=["id", "timestamp", "value"],
    )
    assert data == {
        "id": [1, 2, 3],
        "timestamp": ["2024-01-01T00:00:00", "2024-01-02T00:00:00", "2024-01-03T00:00:00"],
        "value": [10, 20, 30],
    }
    assert handler._columns == ["id", "timestamp", "value"]


def test_validate_data_missing_required_columns(handler):
    data = handler.input_data(
        [{"id": 1, "value": 10}, {"id": 2, "value": 20}, {"id": 3, "value": 30}],
        required_columns=["id", "timestamp", "value"],
    )
    assert data == {"id": [], "timestamp": [], "value": []}
    assert handler._columns == ["id", "timestamp", "value"]


def test_validate_data_empty_dataframe(handler):
    data = handler.input_data([], required_columns=["id"])
    assert data == {}
    assert handler._columns == []


def test_validate_data_all_nan_column(handler):
    data = handler.input_data(
        [
            {"id": 1, "timestamp": None, "value": 10},
            {"id": 2, "timestamp": None, "value": 20},
            {"id": 3, "timestamp": None, "value": 30},
        ],
        required_columns=["id", "timestamp", "value"],
    )
    assert data == {"id": [], "timestamp": [], "value": []}


def test_validate_data_duplicate_columns(handler):
    df = pd.DataFrame([[1, 10, 20]], columns=["id", "value", "value"])
    data = handler.input_data(df, required_columns=["id", "value"])
    records = _columnar_to_records(data)

    assert records == [{"id": 1, "value": 10}]
    assert handler._columns == ["id", "value"]


def test_input_data_sequence_of_scalars(handler):
    data = handler.input_data([1, 2, 3], required_columns=["value"])
    assert data == {"value": [1, 2, 3]}


def test_input_data_bytearray(handler):
    byte_data = bytearray(b"id,timestamp,value\n1,2024-01-01,10\n2,2024-01-02,20")
    data = handler.input_data(byte_data, required_columns=["id", "timestamp", "value"])
    assert data == {
        "id": ["1", "2"],
        "timestamp": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
        "value": ["10", "20"],
    }


def test_input_data_scalar(handler):
    scalar_data = 42
    data = handler.input_data(scalar_data, required_columns=["value"])
    assert data == {"value": [42]}
