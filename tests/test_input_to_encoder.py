import pytest

# should not have to pull from multiple layers like this
from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.encoder_interface import EncoderInterface
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.input_layer.input_interface import InputInterface


@pytest.fixture
def input_handler() -> InputInterface:
    return InputHandler.get_instance()


@pytest.fixture
def encoder() -> BaseEncoder:
    class _DummyEncoder(BaseEncoder[float]):

        def encode(self, input_value: float) -> None:
            # This method is intentionally left empty because _DummyEncoder is a test stub
            # and does not require an actual encoding implementation for this test.
            pass

    return _DummyEncoder()


@pytest.fixture
def encoder_interface(encoder) -> EncoderInterface:
    return encoder.get_instance()


def test_input_to_encoder_passes_records_into_encoder_dataframe(input_handler, encoder):
    """
    This test verifies that InputHandler yields record dictionaries and that
    EncoderInterface.buffered_data can ingest those records
    without data loss.
    """

    # Arrange
    data_list = [[0, 1, 1, 2, 3, 5, 8, 13], [21, 34, 55, 89, 144, 233, 377, 610]]
    input_handler.interface = encoder

    # Act
    records = input_handler.input_data(data_list, required_columns=["timestamp", "value"])
    encoder_records = input_handler.interface.buffer_data(records)

    # Assert
    assert isinstance(input_handler, InputInterface)
    assert isinstance(encoder, EncoderInterface)
    assert isinstance(records, list)
    assert isinstance(encoder_records, list)
    assert [row["value"] for row in encoder_records] == [row["value"] for row in records]
