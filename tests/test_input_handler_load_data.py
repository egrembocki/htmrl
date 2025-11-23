from pathlib import Path

import pandas as pd
import pytest

from psu_capstone.input_layer.input_handler import InputHandler


@pytest.fixture
def handler() -> InputHandler:
    # Fresh singleton reference each time -- no teardown needed
    return InputHandler()


def test_load_raw_data_csv(tmp_path: Path, handler: InputHandler) -> None:
    """Test loading a simple CSV file."""

    # Arrange
    csv_path = tmp_path / "sample.csv"
    csv_content = "a,b,c\n1,2,3\n4,5,6\n"
    csv_path.write_text(csv_content)

    # Act
    df = handler._load_raw_data(str(csv_path), required_columns=["a", "b"])

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)
    assert df.iloc[0].tolist() == [1, 2, 3]


def test_load_raw_data_excel_xlsx(tmp_path: Path, handler: InputHandler) -> None:
    xlsx_path = tmp_path / "sample.xlsx"
    df_in = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
    df_in.to_excel(xlsx_path, index=False)

    df = handler._load_raw_data(str(xlsx_path), required_columns=["a", "b"])

    assert isinstance(df, pd.DataFrame)
    # Excel may coerce types but values should match
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)
    assert df.iloc[0]["a"] == 10
    assert df.iloc[1]["b"] == 40


def test_load_raw_data_excel_xls(tmp_path: Path, handler: InputHandler) -> None:
    xls_path = tmp_path / "sample.xls"
    df_in = pd.DataFrame({"a": [1], "b": [2]})
    df_in.to_excel(xls_path, index=False)

    df = handler._load_raw_data(str(xls_path), required_columns=["a", "b"])

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (1, 2)


def test_load_raw_data_json(tmp_path: Path, handler: InputHandler) -> None:
    json_path = tmp_path / "sample.json"
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_in.to_json(json_path, orient="records")

    df = handler._load_raw_data(str(json_path), required_columns=["a", "b"])

    assert isinstance(df, pd.DataFrame)
    # pd.read_json(orient="records") yields default columns
    assert df.shape == (2, 2)


def test_load_raw_data_txt_returns_dataframe_of_lines(
    tmp_path: Path, handler: InputHandler
) -> None:
    txt_path = tmp_path / "sample.txt"
    lines = ["first line\n", "second line\n", "third line\n"]
    txt_path.write_text("".join(lines))

    df = handler._load_raw_data(str(txt_path), required_columns=[])

    # _data is a list of lines; method wraps it in a DataFrame
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == len(lines)
    # First column with raw text
    assert df.iloc[0, 0] == "first line\n"
    assert df.iloc[-1, 0] == "third line\n"


def test_load_raw_data_unsupported_extension_raises_value_error(
    tmp_path: Path, handler: InputHandler
) -> None:
    bad_path = tmp_path / "sample.xml"
    bad_path.write_text("<root><a>1</a></root>")

    with pytest.raises(ValueError) as excinfo:
        handler._load_raw_data(str(bad_path), required_columns=["a", "b"])

    assert "Unsupported file type" in str(excinfo.value)


def test_load_raw_data_missing_file_raises(tmp_path: Path, handler: InputHandler) -> None:
    """Test that _load_raw_data raises an error when the file does not exist."""

    # Arrange
    missing_path = tmp_path / "missing.csv"

    # Act & Assert
    # Code uses both assert and explicit FileNotFoundError
    with pytest.raises((AssertionError, FileNotFoundError)):
        handler._load_raw_data(str(missing_path), required_columns=["a", "b"])


def test_load_raw_data_requires_string_path(tmp_path: Path, handler: InputHandler) -> None:
    """Test that _load_raw_data raises an error when given a non-string path."""

    # Arrange
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n")

    # Act & Assert
    # Call with non-string to trigger the type assertion
    with pytest.raises(ValueError):
        handler._load_raw_data(csv_path, required_columns=["a", "b"])  # type: ignore[arg-type]


def test_input_handler_is_singleton() -> None:
    h1 = InputHandler()
    h2 = InputHandler()
    assert h1 == h2


def test_load_raw_data_sets_internal_data(tmp_path: Path, handler: InputHandler) -> None:
    """Test that _load_raw_data sets the internal _data attribute correctly."""

    # Arrange
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n")

    # Act
    df = handler._load_raw_data(str(csv_path), required_columns=["a", "b"])

    # Assert
    # get_data returns a new DataFrame copy
    df_copy = handler.get_data()
    assert isinstance(df_copy, pd.DataFrame)
    assert df_copy.equals(df)
    assert df_copy is not df  # not the same object
