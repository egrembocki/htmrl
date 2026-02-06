import pytest

from psu_capstone.encoder_layer.encoder_interface import EncoderInterface
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.input_layer.input_interface import InputInterface


@pytest.fixture
def input_handler() -> InputHandler:
    return InputHandler.get_instance()


def _make_scalar_params() -> ScalarEncoderParameters:
    return ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=False,
        periodic=False,
        category=False,
        active_bits=3,
        sparsity=0.0,
        size=15,
        radius=0.0,
        resolution=1.0,
    )


@pytest.fixture
def encoder() -> ScalarEncoder:
    return ScalarEncoder(_make_scalar_params())


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
    values = [5.0, 10.0, 15.0, 10.0, 5.0]
    data_stream = bytearray(int(value) for value in values)
    input_handler.interface = encoder

    # Act
    records = input_handler.input_data(data_stream, required_columns=["value"])
    encoder_records = input_handler.interface.buffer_data(records)
    encoded_from_buffer = [encoder.encode(row["value"]) for row in encoder_records]
    reference_encoder = ScalarEncoder(_make_scalar_params())
    encoded_reference = [reference_encoder.encode(value) for value in values]

    # Assert
    assert isinstance(input_handler, InputInterface)
    assert isinstance(encoder, EncoderInterface)
    assert isinstance(records, list)
    assert isinstance(encoder_records, list)
    assert [row["value"] for row in encoder_records] == values
    assert all(row["value"] is not None for row in encoder_records)
    assert encoder_records == records
    assert encoded_from_buffer == encoded_reference
