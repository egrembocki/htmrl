import pytest

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


def test_input_to_encoder_passes_records_into_encoder(input_handler, encoder):
    """
    This test verifies that InputHandler yields record dictionaries and that
    encoder-ready sequences can ingest those records without data loss.
    """

    # Arrange
    values = [5.0, 10.0, 15.0, 10.0, 5.0]
    data_stream = bytearray(int(value) for value in values)
    # Act
    records = input_handler.input_data(data_stream, required_columns=["value"])
    encoder_sequence = input_handler.to_encoder_sequence(records, column="value")
    encoded_from_sequence = [encoder.encode(value) for value in encoder_sequence]
    reference_encoder = ScalarEncoder(_make_scalar_params())
    encoded_reference = [reference_encoder.encode(value) for value in values]

    # Assert
    assert isinstance(input_handler, InputInterface)
    assert isinstance(encoder, ScalarEncoder)
    assert isinstance(records, list)
    assert isinstance(encoder_sequence, list)
    assert encoder_sequence == values
    assert all(value is not None for value in encoder_sequence)
    assert encoded_from_sequence == encoded_reference
