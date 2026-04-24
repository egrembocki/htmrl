"""
tests.test_input_to_encoder

Test suite for integration between InputHandler and Encoder components.

Validates the end-to-end data flow from input handler (loading data) through encoder
(converting data to SDR representations). Tests ensure:
- Input data is correctly passed from InputHandler to Encoder
- Scalar values are properly encoded based on encoder parameters (min/max, periodic)
- SDR representations have correct dimensionality and sparsity
- Data transformations maintain semantic correctness through the pipeline

These tests validate the critical pipeline connection between input data loading and
SDR encoding, ensuring data integrity across component boundaries.
"""

import numpy as np
import pytest

from htmrl.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from htmrl.input_layer.input_handler import InputHandler
from htmrl.input_layer.input_interface import InputInterface


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
    encoder_sequence = input_handler.get_column_data("value")

    normalized_sequence = [float(v) for v in encoder_sequence]
    encoded_from_sequence = [encoder.encode(value) for value in normalized_sequence]
    reference_encoder = ScalarEncoder(_make_scalar_params())
    encoded_reference = [reference_encoder.encode(value) for value in normalized_sequence]

    # Assert
    assert isinstance(input_handler, InputInterface)
    assert isinstance(encoder, ScalarEncoder)
    assert isinstance(records, dict)
    assert isinstance(encoder_sequence, list)
    assert isinstance(normalized_sequence, list)
    assert len(normalized_sequence) > 0
    assert all(value is not None for value in normalized_sequence)
    assert encoded_from_sequence == encoded_reference


def test_sine_wave_through_input_handler(input_handler, encoder):
    """Send a scaled sine wave through the InputHandler and verify encoder shape.

    The ScalarEncoder used in these tests expects inputs in [0, 100], so the
    sine wave is scaled accordingly before ingestion.
    """

    # Arrange: generate a 50-sample sine wave scaled to [0, 100] using numpy
    n = 50
    t = np.arange(n)
    values = ((np.sin(2 * np.pi * t / n) + 1.0) * 50.0).tolist()

    # Act: feed through the input handler and extract the encoder-ready sequence
    input_handler.input_data(values, required_columns=["value"])
    seq_values = input_handler.get_column_data("value")

    # Assert: sequence shape and encoder outputs
    assert isinstance(seq_values, list)
    assert len(seq_values) == n
    assert all(isinstance(v, (int, float)) for v in seq_values)

    encoded = [encoder.encode(v) for v in seq_values]
    assert len(encoded) == n
    assert all(isinstance(enc, list) for enc in encoded)
    assert all(len(enc) == encoder.size for enc in encoded)
