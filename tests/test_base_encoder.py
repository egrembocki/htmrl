"""Base Encoder Test Suite"""

import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder


@pytest.fixture
def base_encoder_instance() -> BaseEncoder:
    """Fixture to create a BaseEncoder instance for testing."""

    # Arrange @mock
    class TestEncoder(BaseEncoder):
        def encode(self, input_value) -> list[int]:
            """Dummy encode method for testing."""
            return []

    return TestEncoder(size=100)


def test_base_encoder_initialization(base_encoder_instance):
    """Test that the BaseEncoder initializes correctly."""
    # Act
    encoder = base_encoder_instance

    # Assert
    print(encoder.size)

    assert encoder.size == 100
