import pandas as pd
import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.input_layer.input_interface import InputInterface


@pytest.fixture
def handler() -> InputInterface:
    return InputHandler()


@pytest.fixture
def encoder() -> BaseEncoder:
    class _DummyEncoder(BaseEncoder[float]):

        def __init__(self, interface: InputInterface | None = None) -> None:
            super().__init__([1])
            self._df: pd.DataFrame | None = None
            self._interface = interface

        def setup(self, df: pd.DataFrame) -> None:
            self._df = df.copy()
            if self._interface is not None:
                self._interface.input_data(self._df)
                self._data = self._interface.data

        def encode(self, input_value: float, output_sdr: SDR) -> None:
            # This method is intentionally left empty because _DummyEncoder is a test stub
            # and does not require an actual encoding implementation for this test.
            pass

    return _DummyEncoder()


def test_input_to_encoder_passes_same_dataframe_object(handler, encoder):
    """
    This test verifies that InputInterface.data returns a pandas DataFrame
    and that passing that DataFrame into an encoder keeps the *same object*.

    The purpose is to ensure we are not copying, re-wrapping, or rebuilding
    the DataFrame — the encoder should receive the identical object produced
    by the handler.
    """

    # Arrange
    data_list = [0, 1, 1, 2, 3, 5, 8, 13]

    handler.input_data(data_list, required_columns=["timestamp", "value"])
    df = handler.data

    # Act
    encoder.attach_input(df)

    # Assert
    assert isinstance(handler, InputInterface)
    assert isinstance(df, pd.DataFrame)
    assert encoder._df is df  # identity (same memory reference)
    assert encoder.input_df is not None
    pd.testing.assert_frame_equal(encoder.input_df, df)
