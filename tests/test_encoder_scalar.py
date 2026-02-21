"""Test suite for the SDR Encoder-Scalar."""

import pytest

from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters


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
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=5,
        sparsity=0.0,
        size=10,
        radius=1.0,
        category=False,
        resolution=0.0,
    )

    # Act
    encoder = ScalarEncoder(parameters)
    """Demonstrating deep copy"""
    ScalarEncoder(parameters)

    # Assert
    assert isinstance(encoder, ScalarEncoder)
    assert encoder.size == 10


def test_clipping_inputs():
    """Test that inputs are correctly clipped to the specified min/max range."""

    # Arrange
    p = ScalarEncoderParameters(
        minimum=10,
        maximum=20,
        clip_input=False,
        periodic=False,
        active_bits=2,
        sparsity=0.0,
        size=10,
        radius=1.0,
        resolution=0.0,
        category=False,
    )
    # Act and Assert baseline
    encoder = ScalarEncoder(p)

    assert encoder.size == 10
    # assert test_sdr.size == 10

    # Act and Asset - Test input clipping
    # These should pass without exceptions
    try:
        encoder.encode(10.0)  # At minimum edge case
        encoder.encode(20.0)  # At maximum edge case
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

    with pytest.raises(ValueError):
        encoder.encode(9.9)  # Below minimum edge case
        encoder.encode(20.1)  # Above maximum edge case


def test_valid_scalar_inputs():
    """Test that valid scalar inputs are encoded correctly."""

    # Arrange
    params = ScalarEncoderParameters(
        size=10,
        active_bits=2,
        minimum=10,
        maximum=20,
        sparsity=0.0,
        radius=1.0,
        category=False,
        resolution=0.0,
        clip_input=False,
        periodic=False,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params)
    assert encoder.size == 10

    with pytest.raises(Exception):
        encoder.encode(9.999)  # Below minimum edge case
        encoder.encode(20.0001)  # Above maximum edge case

    try:
        encoder.encode(10.0)  # At minimum edge case
        encoder.encode(19.9)  # Just below maximum edge case
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
        radius=1.0,
        category=False,
        resolution=0.0,
        clip_input=False,
        periodic=False,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params)
    assert encoder.size == 66

    # Act and Assert - Value less than minimum should raise
    with pytest.raises(Exception):
        encoder.encode(-0.01)  # Below minimum edge case

    # Act and Assert - Value greater than maximum should raise
    with pytest.raises(Exception):
        encoder.encode(66.0)  # Above maximum edge case

    # Value within range should not raise
    try:
        encoder.encode(0.0)  # At minimum edge case
        encoder.encode(32.0)  # Mid-range value
        encoder.encode(65.0)  # At maximum edge case
        encoder.encode(10.0)
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
        radius=1.0,
        category=False,
        resolution=0.0,
    )

    encoder = ScalarEncoder(params)

    cases = [
        (10.0, [0, 1, 2]),
        (20.0, [4, 5, 6]),
    ]

    do_scalar_value_cases(encoder, cases)


def test_scalar_encoder_round_to_nearest_multiple_of_resolution():
    """Test that scalar encoder rounds to the nearest multiple of resolution correctly."""

    # Arrange
    params = ScalarEncoderParameters(
        minimum=10,
        maximum=20,
        clip_input=False,
        periodic=False,
        active_bits=3,
        sparsity=0.0,
        size=0,
        radius=0.0,
        category=False,
        resolution=1.0,
    )

    # Act and Assert - baseline
    encoder = ScalarEncoder(params)
    assert encoder.size == 13

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
        minimum=10,
        maximum=20,
        clip_input=False,
        periodic=True,
        active_bits=3,
        sparsity=0.0,
        size=0,
        radius=0.0,
        category=False,
        resolution=1,
    )
    encoder = ScalarEncoder(params)

    # Act and Assert - baseline
    assert encoder.size == 10
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
        minimum=1,
        maximum=100,
        clip_input=False,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(p))

    p = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(p))

    p = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=False,
        periodic=True,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.1337,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(p))

    p = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=False,
        periodic=False,
        active_bits=34,
        sparsity=0.0,
        size=0,
        radius=0.0,
        category=False,
        resolution=0.1337,
    )
    inputs.append(ScalarEncoder(p))

    q = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=False,
        periodic=False,
        active_bits=0,
        sparsity=0.15,
        size=100,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(q))

    r = ScalarEncoderParameters(
        minimum=1,
        maximum=100,
        clip_input=False,
        periodic=False,
        active_bits=0,
        sparsity=0.02,
        size=700,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    inputs.append(ScalarEncoder(r))
    inputs.append(ScalarEncoder(r))

    for encoder in inputs:
        if type(encoder) is ScalarEncoder:
            assert encoder.size > 0
            assert encoder._active_bits > 0
            assert encoder._active_bits < encoder.size
            assert encoder._minimum <= encoder._maximum
            assert encoder._resolution > 0
            assert encoder._radius > 0
            assert encoder._sparsity > 0


# ---------------------------------------------------------------------------
# Output format and parameter conformance (binary 0/1 only, length, active_bits)
# ---------------------------------------------------------------------------


def test_scalar_encode_output_only_zeros_and_ones():
    """Encoder output must contain only 0 and 1."""
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=5,
        sparsity=0.0,
        size=50,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    for value in (0, 10, 50, 100):
        out = encoder.encode(value)
        assert all(b in (0, 1) for b in out), f"Output must be binary (0/1), got {set(out)}"


def test_scalar_encode_output_length_equals_size():
    """Encoder output length must equal the configured size."""
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=4,
        sparsity=0.0,
        size=32,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    out = encoder.encode(50.0)
    assert len(out) == 32, f"Output length must equal size (32), got {len(out)}"


def test_scalar_encode_output_active_bits_conforms():
    """Output must have exactly active_bits ones; sparsity = active_bits/size."""
    size = 64
    active_bits = 8
    p = ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        clip_input=True,
        periodic=False,
        active_bits=active_bits,
        sparsity=0.0,
        size=size,
        radius=1.0,
        category=False,
        resolution=0.0,
    )
    encoder = ScalarEncoder(p)
    out = encoder.encode(25.0)
    num_ones = sum(out)
    assert num_ones == active_bits, f"Exactly {active_bits} ones expected, got {num_ones}"
    assert num_ones / len(out) == pytest.approx(
        active_bits / size
    ), "Sparsity should equal active_bits/size"
