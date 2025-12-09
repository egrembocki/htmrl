"""
tests.test_input_handler_load_data
"""

from pathlib import Path

import pandas as pd
import pytest

from psu_capstone.input_layer.input_handler import InputHandler


@pytest.fixture
def handler() -> InputHandler:
    # Fresh singleton reference each time -- no teardown needed
    return InputHandler()


@pytest.fixture
def temp_path(tmp_path: Path) -> Path:
    # e.g. create a subdirectory or seed test files here
    work_dir = tmp_path / "input_handler"
    work_dir.mkdir()
    return work_dir


def test_load_input_data_csv(temp_path: Path, handler: InputHandler) -> None:
    """Ensure CSV files are parsed with a timestamp column and exact value preservation."""
    # Arrange
    csv_path = temp_path / "sample.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n")
    required = ["timestamp", "a", "b", "c"]

    # Act
    df = handler.input_data(str(csv_path), required_columns=required)

    # Assert
    assert list(df.columns) == required
    assert df.shape == (2, 4)
    assert df.loc[0, ["a", "b", "c"]].values.tolist() == [1, 2, 3]  # type: ignore


def test_load_input_data_excel_xlsx(temp_path: Path, handler: InputHandler) -> None:
    """Confirm XLSX ingestion yields the requested timestamp and numeric columns."""
    # Arrange
    xlsx_path = temp_path / "sample.xlsx"
    df_in = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
    df_in.to_excel(xlsx_path, index=False)
    required = ["timestamp", "a", "b"]

    # Act
    df = handler.input_data(str(xlsx_path), required_columns=required)

    # Assert
    assert list(df.columns) == required
    assert df["a"].tolist() == [10, 20]
    assert df["b"].tolist() == [30, 40]


def test_load_input_data_excel_xls(temp_path: Path, handler: InputHandler) -> None:
    """Validate legacy XLS files are supported and maintain row counts."""
    # Arrange
    xls_path = temp_path / "sample.xls"
    df_in = pd.DataFrame({"a": [1], "b": [2]})
    df_in.to_excel(xls_path, index=False)
    required = ["timestamp", "a", "b"]

    # Act
    df = handler.input_data(str(xls_path), required_columns=required)

    # Assert
    assert list(df.columns) == required
    assert df.shape == (1, 3)
    assert df.loc[0, "a"] == 1


def test_load_input_data_json(temp_path: Path, handler: InputHandler) -> None:
    """Check JSON records are converted into the expected DataFrame layout."""
    # Arrange
    json_path = temp_path / "sample.json"
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_in.to_json(json_path, orient="records")
    required = ["timestamp", "a", "b"]

    # Act
    df = handler.input_data(str(json_path), required_columns=required)

    # Assert
    assert list(df.columns) == required
    assert df["a"].tolist() == [1, 2]
    assert df["b"].tolist() == [3, 4]


def test_load_input_data_txt_returns_dataframe_of_lines(
    temp_path: Path, handler: InputHandler
) -> None:
    """Verify plain-text files become line-per-row DataFrames with timestamps."""
    # Arrange
    txt_path = temp_path / "sample.txt"
    lines = ["first line\n", "second line\n", "third line\n"]
    txt_path.write_text("".join(lines))

    # Act
    df = handler.input_data(str(txt_path), required_columns=["timestamp", "value"])

    # Assert
    assert list(df.columns) == ["timestamp", "value"]
    assert df.shape == (len(lines), 2)
    assert df["value"].tolist() == lines


def test_load_input_data_unsupported_extension_raises_value_error(
    temp_path: Path, handler: InputHandler
) -> None:
    """Assert unknown file extensions raise ValueError to signal unsupported formats."""
    # Arrange
    bad_path = temp_path / "sample.xml"
    bad_path.write_text("<root><a>1</a></root>")

    # Act / Assert
    with pytest.raises(ValueError):
        handler.input_data(str(bad_path))


def test_load_input_data_missing_file_raises(temp_path: Path, handler: InputHandler) -> None:
    """Ensure missing files raise FileNotFoundError for clearer diagnostics."""
    # Arrange
    missing_path = temp_path / "missing.csv"

    # Act / Assert
    with pytest.raises(FileNotFoundError):
        handler.input_data(str(missing_path), required_columns=["timestamp", "a"])


def test_load_input_data_requires_string_path(temp_path: Path, handler: InputHandler) -> None:
    """Guarantee only string-like paths are accepted by the input API."""
    # Arrange
    csv_path = temp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n")

    # Act / Assert
    with pytest.raises(TypeError):
        handler.input_data(csv_path)  # type: ignore[arg-type]


def test_input_handler_is_singleton() -> None:
    """Confirm InputHandler enforces a singleton instance."""
    # Arrange / Act
    h1 = InputHandler().get_instance()
    h2 = InputHandler().get_instance()

    # Assert
    assert h1 is h2


def test_load_input_data_sets_internal_data(temp_path: Path, handler: InputHandler) -> None:
    """Check the handler caches the latest DataFrame without sharing references."""
    # Arrange
    csv_path = temp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n")
    required = ["timestamp", "a", "b"]

    # Act
    df_one = handler.input_data(str(csv_path), required_columns=required)
    df_two = handler.input_data(str(csv_path), required_columns=required)

    # Assert
    assert isinstance(df_one, pd.DataFrame)
    assert isinstance(df_two, pd.DataFrame)
    assert df_two["a"].equals(df_one["a"]) and df_two["b"].equals(df_one["b"])
    assert df_two is not df_one
