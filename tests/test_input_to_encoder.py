import pandas as pd
import pytest

# should not have to pull from multiple layers like this
from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.encoder_interface import EncoderInterface
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.input_layer.input_interface import InputInterface


@pytest.fixture
def input_handler() -> InputInterface:
    return InputHandler.get_instance()


@pytest.fixture
def encoder() -> BaseEncoder:
    class _DummyEncoder(BaseEncoder[float]):

        def encode(self, input_value: float, output_sdr: SDR) -> None:
            # This method is intentionally left empty because _DummyEncoder is a test stub
            # and does not require an actual encoding implementation for this test.
            pass

    return _DummyEncoder()


@pytest.fixture
def encoder_interface(encoder) -> EncoderInterface:
    return encoder.get_instance()


def test_input_to_encoder_passes_same_dataframe_object(input_handler, encoder):
    """
    This test verifies that EncoderInterface.buffered_data returns a pandas DataFrame
    and that passing that DataFrame from an InputHandler to an encoder keeps the *same object*.

    The purpose is to ensure we are not copying, re-wrapping, or rebuilding
    the DataFrame — the encoder should receive the identical object produced
    by the handler.
    """

    # Arrange
    data_list = [[0, 1, 1, 2, 3, 5, 8, 13], [21, 34, 55, 89, 144, 233, 377, 610]]
    input_handler.interface = encoder

    # Act
    input_df = input_handler.input_data(data_list, required_columns=["timestamp", "value"])
    encoder_df = input_handler.interface.buffer_data(input_df)

    # Assert
    assert isinstance(input_handler, InputInterface)
    assert isinstance(encoder, EncoderInterface)
    assert isinstance(input_df, pd.DataFrame)
    assert isinstance(encoder_df, pd.DataFrame)
    assert input_df["value"].equals(encoder_df["value"])
    assert input_df is encoder_df
