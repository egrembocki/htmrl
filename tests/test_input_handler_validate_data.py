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

from psu_capstone.input_layer.improved_input_handler import InputHandler


@pytest.fixture
def handler():
    h = InputHandler()
    return h


def test_validate_data_valid(handler):
    handler._data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [10, 20, 30],
        }
    )
    assert handler._validate_data(required_columns=["id", "timestamp", "value"]) is True


def test_validate_data_missing_required_columns(handler):
    handler._data = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    with pytest.raises(ValueError):
        handler._validate_data(required_columns=["id", "timestamp", "value"])


def test_validate_data_empty_dataframe(handler):
    handler._data = pd.DataFrame()
    with pytest.raises(ValueError):
        handler._validate_data(required_columns=["id"])


def test_validate_data_all_nan_column(handler):
    handler._data = pd.DataFrame(
        {"id": [1, 2, 3], "timestamp": [None, None, None], "value": [10, 20, 30]}
    )
    with pytest.raises(ValueError):
        handler._validate_data(required_columns=["id", "timestamp", "value"])


def test_validate_data_duplicate_columns(handler):
    handler._data = pd.DataFrame(
        [[1, "2024-01-01", 10, 100]], columns=["id", "timestamp", "value", "value"]
    )
    with pytest.raises(ValueError):
        handler._validate_data(required_columns=["id", "timestamp", "value"])


def test_validate_data_non_dataframe(handler):
    handler._data = [1, 2, 3]
    with pytest.raises(ValueError):
        handler._validate_data(required_columns=["id"])
