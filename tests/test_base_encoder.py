"""Base Encoder Test Suite"""

import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder

# from psu_capstone.sdr_layer.sdr import SDR


@pytest.fixture
def base_encoder_instance() -> BaseEncoder:
    """Fixture to create a BaseEncoder instance for testing."""

    # Arrange @mock
    class TestEncoder(BaseEncoder):
        def encode(self, input_value) -> list[int]:
            """Dummy encode method for testing."""
            return []

    return TestEncoder(100)


def test_base_encoder_initialization(base_encoder_instance):
    """Test that the BaseEncoder initializes correctly."""
    # Act
    encoder = base_encoder_instance

    # Assert
    assert encoder.size == 100


def test_base_encoder_size_setter(base_encoder_instance):
    """Ensure size setter enforces constraints."""

    encoder = base_encoder_instance
    encoder.size = 64
    assert encoder.size == 64

    with pytest.raises(ValueError):
        encoder.size = 0

    with pytest.raises(ValueError):
        encoder.size = -1
