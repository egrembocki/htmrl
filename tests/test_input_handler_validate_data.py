import pytest

from psu_capstone.input_layer.improved_input_handler import InputHandler


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
    assert handler._validate_data(required_columns=["id", "timestamp", "value"]) is True


def test_validate_data_missing_required_columns(handler):
    handler._data = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]
    handler._columns = ["id", "value"]
    with pytest.raises(ValueError):
        handler._validate_data(required_columns=["id", "timestamp", "value"])


def test_validate_data_empty_dataframe(handler):
    handler._data = []
    with pytest.raises(ValueError):
        handler._validate_data(required_columns=["id"])


def test_validate_data_all_nan_column(handler):
    handler._data = [
        {"id": 1, "timestamp": None, "value": 10},
        {"id": 2, "timestamp": None, "value": 20},
        {"id": 3, "timestamp": None, "value": 30},
    ]
    handler._columns = ["id", "timestamp", "value"]
    with pytest.raises(ValueError):
        handler._validate_data(required_columns=["id", "timestamp", "value"])


def test_validate_data_duplicate_columns(handler):
    handler._data = [{"id": 1, "timestamp": "2024-01-01", "value": 10}]
    handler._columns = ["id", "timestamp", "value", "value"]
    with pytest.raises(ValueError):
        handler._validate_data(required_columns=["id", "timestamp", "value"])


def test_validate_data_non_dataframe(handler):
    handler._data = [1, 2, 3]
    with pytest.raises(ValueError):
        handler._validate_data(required_columns=["id"])
