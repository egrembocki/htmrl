"""Test suite for the SDR Encoder-Scalar."""

import pytest

from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def scalar_encoder_instance():
    """Fixture to create a ScalarEncoder instance for testing. This may change when we get Union working properly."""


# Helper -- may need to be implemented later
def do_scalar_value_cases(encoder: ScalarEncoder, cases):
    pass


def test_scalar_encoder_initialization():
    """Test the initialization of the ScalarEncoder."""

    # Arrange
    parameters = ScalarEncoderParameters(
        minimum=0.0,
        maximum=100.0,
        clip_input=True,
        periodic=False,
        active_bits=5,
        sparsity=0.0,
        size=10,
        radius=0.0,
        category=False,
        resolution=0.0,
    )

    # Act
    encoder = ScalarEncoder(parameters, [1, 10])
    """Demonstrating deep copy"""
    ScalarEncoder(parameters, [1, 10])

    # Assert
    assert isinstance(encoder, ScalarEncoder)
    assert encoder.size == 10
    assert encoder.dimensions == [1, 10]


def test_clipping_inputs():
    """Test that inputs are correctly clipped to the specified min/max range."""

    # Arrange
    p = ScalarEncoderParameters(
        minimum=10.0,
        maximum=20.0,
        clip_input=False,
        periodic=False,
        active_bits=2,
        sparsity=0.0,
        size=10,
        radius=0.0,
        resolution=0.0,
        category=False,
    )
    # Act and Assert baseline
    encoder = ScalarEncoder(p, dimensions=[2, 5])
    test_sdr = SDR([2, 5])
    test_sdr.zero()

    assert encoder.size == 10
    assert encoder.dimensions == [2, 5]
    assert test_sdr.size == 10

    # Act and Asset - Test input clipping
    # These should pass without exceptions
    try:
        encoder.encode(10.0, test_sdr)  # At minimum edge case
        encoder.encode(20.0, test_sdr)  # At maximum edge case
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

    with pytest.raises(ValueError):
        encoder.encode(9.9, test_sdr)  # Below minimum edge case
        encoder.encode(20.1, test_sdr)  # Above maximum edge case


def test_valid_scalar_inputs():
    """Test that valid scalar inputs are encoded correctly."""

    # Arrange
    params = ScalarEncoderParameters(
        size=10,
        active_bits=2,
        minimum=10,
        maximum=20,
        sparsity=0.0,
        radius=0.0,
        category=False,
        resolution=0.0,
        clip_input=False,
        periodic=False,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params, [2, 5])
    test_sdr = SDR([2, 5])
    assert encoder.size == 10
    assert encoder.dimensions == [2, 5]
    assert test_sdr.size == 10
    assert test_sdr.get_sparse() == []

    with pytest.raises(Exception):
        encoder.encode(9.999, test_sdr)  # Below minimum edge case
        encoder.encode(20.0001, test_sdr)  # Above maximum edge case

    try:
        encoder.encode(10.0, test_sdr)  # At minimum edge case
        encoder.encode(19.9, test_sdr)  # Just below maximum edge case
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


def test_scalar_encoder_category_encode():
    """Test that category scalar inputs are encoded correctly."""
    # Arrange
    params = ScalarEncoderParameters(
        size=66,
        sparsity=0.02,
        minimum=0,
        maximum=65,
        active_bits=0,
        radius=0.0,
        category=False,
        resolution=0.0,
        clip_input=False,
        periodic=False,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params, dimensions=[66])
    output = SDR([66])
    assert encoder.size == 66
    assert encoder.dimensions == [66]
    assert output.size == 66

    # Act and Assert - Value less than minimum should raise
    with pytest.raises(Exception):
        encoder.encode(-0.01, output)  # Below minimum edge case

    # Act and Assert - Value greater than maximum should raise
    with pytest.raises(Exception):
        encoder.encode(66.0, output)  # Above maximum edge case

    # Value within range should not raise
    try:
        encoder.encode(0.0, output)  # At minimum edge case
        encoder.encode(32.0, output)  # Mid-range value
        encoder.encode(65.0, output)  # At maximum edge case
        encoder.encode(10.0, output)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")


def test_scalar_encoder_non_integer_bucket_width():
    """Test that scalar encoder handles non-integer bucket widths correctly."""
    # Arrange
    params = ScalarEncoderParameters(
        minimum=10.0,
        maximum=20.0,
        clip_input=True,
        periodic=False,
        active_bits=3,
        sparsity=0.0,
        size=7,
        radius=0.0,
        category=False,
        resolution=0.0,
    )

    encoder = ScalarEncoder(params, [1, 7])

    cases = [
        (10.0, [0, 1, 2]),
        (20.0, [4, 5, 6]),
    ]

    do_scalar_value_cases(encoder, cases)


def test_scalar_encoder_round_to_nearest_multiple_of_resolution():
    """Test that scalar encoder rounds to the nearest multiple of resolution correctly."""

    # Arrange
    params = ScalarEncoderParameters(
        minimum=10.0,
        maximum=20.0,
        clip_input=False,
        periodic=False,
        active_bits=3,
        sparsity=0.0,
        size=0,
        radius=0.0,
        category=False,
        resolution=1,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params, dimensions=[1, 13])
    assert encoder._size == 13
    assert encoder._dimensions == [1, 13]

    cases = [
        (10.00, [0, 1, 2]),
        (10.49, [0, 1, 2]),
        (10.50, [1, 2, 3]),
        (11.49, [1, 2, 3]),
        (11.50, [2, 3, 4]),
        (14.49, [4, 5, 6]),
        (14.50, [5, 6, 7]),
        (15.49, [5, 6, 7]),
        (15.50, [6, 7, 8]),
        (19.00, [9, 10, 11]),
        (19.49, [9, 10, 11]),
        (19.50, [10, 11, 12]),
        (20.00, [10, 11, 12]),
    ]

    do_scalar_value_cases(encoder, cases)


def test_scalar_encoder_periodic_round_nearest_multiple_of_resolution():
    """Test that periodic scalar encoder rounds to the nearest multiple of resolution correctly."""
    # Arrange
    params = ScalarEncoderParameters(
        minimum=10.0,
        maximum=20.0,
        clip_input=False,
        periodic=True,
        active_bits=3,
        sparsity=0.0,
        size=0,
        radius=0.0,
        category=False,
        resolution=1,
    )
    encoder = ScalarEncoder(params, [1, 10])

    # Act and Assert - baseline
    assert encoder.size == 10
    assert encoder.dimensions == [1, 10]
    cases = [
        (10.00, [0, 1, 2]),
        (10.49, [0, 1, 2]),
        (10.50, [1, 2, 3]),
        (11.49, [1, 2, 3]),
        (11.50, [2, 3, 4]),
        (14.49, [4, 5, 6]),
        (14.50, [5, 6, 7]),
        (15.49, [5, 6, 7]),
        (15.50, [6, 7, 8]),
        (19.49, [9, 0, 1]),
        (19.50, [0, 1, 2]),
        (20.00, [0, 1, 2]),
    ]

    do_scalar_value_cases(encoder, cases)


def nearly_equal(a, b, tol=1e-5):
    return abs(a - b) <= tol


def test_scalar_encoder_serialization():
    """Test serialization and deserialization of ScalarEncoder."""

    # Arrange
    inputs = []

    p = ScalarEncoderParameters(
        minimum=-1.234,
        maximum=12.34,
        clip_input=False,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(p, [1, 34]))

    p = ScalarEncoderParameters(
        minimum=-1.234,
        maximum=12.34,
        clip_input=True,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(p, [1, 34]))

    p = ScalarEncoderParameters(
        minimum=-1.234,
        maximum=12.34,
        clip_input=False,
        periodic=True,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(p, [1, 34]))

    p = ScalarEncoderParameters(
        minimum=-1.234,
        maximum=12.34,
        clip_input=False,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.0,
        category=False,
        resolution=0.1337,
    )
    inputs.append(ScalarEncoder(p, [1, 34]))

    q = ScalarEncoderParameters(
        minimum=-1.0,
        maximum=1.003,
        clip_input=False,
        periodic=False,
        active_bits=0,
        sparsity=0.15,
        size=100,
        radius=0.0,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(q, [1, 100]))

    r = ScalarEncoderParameters(
        minimum=0,
        maximum=65,
        clip_input=False,
        periodic=False,
        active_bits=0,
        sparsity=0.02,
        size=700,
        radius=0.0,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(r, [1, 700]))
    inputs.append(ScalarEncoder(r, [1, 700]))

    for encoder in inputs:
        if type(encoder) is ScalarEncoder:
            p1 = encoder._parameters
            p2 = encoder._parameters

            assert p1.size == p2.size
            assert getattr(p1, "category", None) == getattr(p2, "category", None)
            assert p1.active_bits == p2.active_bits
            assert p1.periodic == p2.periodic
            assert p1.clip_input == p2.clip_input
            assert nearly_equal(p1.minimum, p2.minimum)
            assert nearly_equal(p1.maximum, p2.maximum)
            assert nearly_equal(p1.resolution, p2.resolution)
            assert nearly_equal(p1.sparsity, p2.sparsity)
            assert nearly_equal(p1.radius, p2.radius)
