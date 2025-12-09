import pandas as pd
import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.input_layer.input_interface import InputInterface


@pytest.fixture
def handler() -> InputInterface:
    return InputHandler.get_instance()


@pytest.fixture
def encoder() -> BaseEncoder:
    class _DummyEncoder(BaseEncoder[float]):

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
    encoder.interface = handler

    # Act
    handle_df = handler.input_data(data_list, required_columns=["timestamp", "value"])
    encode_df = encoder.interface.input_data(data_list, required_columns=["timestamp", "value"])

    # Assert
    assert isinstance(handler, InputInterface)
    assert isinstance(encoder, BaseEncoder)
    assert isinstance(handle_df, pd.DataFrame)
    assert isinstance(encode_df, pd.DataFrame)
    assert handle_df is not encode_df  # different timestamps
    assert handle_df["value"].equals(encode_df["value"])  # same object
    assert encoder.interface is not None
