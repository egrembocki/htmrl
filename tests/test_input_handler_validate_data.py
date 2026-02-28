"""
tests.test_input_handler_validate_data

Test suite for InputHandler data validation functionality.

Validates that InputHandler correctly processes and validates input data for consistency,
type correctness, and required field presence. Tests ensure the processing step correctly:
- Handles missing or required columns
- Normalizes data types and datetime columns
- Detects and handles duplicate columns
- Fills missing values appropriately
- Validates data structures before encoding

These tests ensure the input data pipeline correctly processes data before encoder handling.
"""

import numpy as np
import pandas as pd
import pytest

from psu_capstone.input_layer.input_handler import InputHandler


@pytest.fixture
def handler():
    """Create a fresh InputHandler instance for each test."""
    return InputHandler()


def test_process_dataframe_valid(handler):
    """Test that valid data with required columns is processed correctly."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [10, 20, 30],
        }
    )
    result = handler._process_dataframe(df, required_columns=["id", "timestamp", "value"])
    assert not result.empty
    assert list(result.columns) == ["id", "timestamp", "value"]


def test_process_dataframe_adds_missing_required_columns(handler):
    """Test that missing required columns are added with None values."""
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    result = handler._process_dataframe(df, required_columns=["id", "timestamp", "value"])
    assert "timestamp" in result.columns
    assert list(result.columns) == ["id", "timestamp", "value"]


def test_process_dataframe_empty(handler):
    """Test that empty dataframes are handled correctly."""
    df = pd.DataFrame()
    result = handler._process_dataframe(df, required_columns=None)
    assert result.empty


def test_process_dataframe_removes_all_nan_rows(handler):
    """Test that rows with all NaN values are dropped during datetime normalization."""
    df = pd.DataFrame({"id": [1, 2, 3], "timestamp": [None, None, None], "value": [10, 20, 30]})
    result = handler._process_dataframe(df)
    # After datetime normalization, rows with NaN are dropped
    assert len(result) <= len(df)


def test_process_dataframe_duplicate_columns(handler):
    """Test that duplicate columns are removed automatically."""
    df = pd.DataFrame([[1, "2024-01-01", 10, 100]], columns=["id", "timestamp", "value", "value"])
    result = handler._process_dataframe(df)
    assert result.columns.is_unique
    assert "value" in result.columns


def test_normalize_column_types_mixed_numeric(handler):
    """Test that mixed numeric types are coerced correctly."""
    df = pd.DataFrame({"value": [1, 2.5, 3, 4.0]})
    result = handler._normalize_column_types(df)
    assert pd.api.types.is_numeric_dtype(result["value"])


def test_normalize_column_types_unsupported_type_raises(handler):
    """Test that unsupported types raise ValueError."""
    df = pd.DataFrame({"value": [1, 2, {"key": "value"}]})
    with pytest.raises(ValueError, match="Corrupt data detected"):
        handler._normalize_column_types(df)


def test_fill_missing_values_numeric(handler):
    """Test that missing numeric values are filled with mean."""
    df = pd.DataFrame({"value": [10.0, None, 30.0, 40.0]})
    result = handler._fill_missing_values(df)
    # Mean of [10, 30, 40] is ~26.67
    assert result["value"].notna().all()
    assert np.isclose(result["value"].iloc[1], 26.666666, rtol=0.01)


def test_detect_repeating_values(handler):
    """Test detection of repeating values in columns."""
    df = pd.DataFrame({"category": ["A", "A", "A", "A", "B"]})
    is_repeating, cols = handler._detect_repeating_values(df, threshold=3)
    assert is_repeating is True
    assert "category" in cols


def test_input_data_with_dict(handler):
    """Test that dictionary input is processed correctly."""
    data = {"id": [1, 2, 3], "value": [10, 20, 30]}
    result = handler.input_data(data)
    assert isinstance(result, dict)
    assert "id" in result
    assert "value" in result
    assert result["id"] == [1, 2, 3]


def test_input_data_with_required_columns(handler):
    """Test that required columns are enforced."""
    data = {"id": [1, 2, 3], "value": [10, 20, 30]}
    result = handler.input_data(data, required_columns=["id", "value", "timestamp"])
    assert "timestamp" in result
