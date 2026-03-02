"""
tests.test_base_encoder

Test suite for BaseEncoder abstract base class functionality.

Validates that BaseEncoder correctly enforces the encoder interface contract, including:
- Initialization with configurable size parameters
- Abstract encode() method implementation requirements
- Proper encoding dimension (size parameter) validation
- Fixture setup for concrete encoder implementations

These tests ensure BaseEncoder provides a consistent foundation for all encoder subclasses
(ScalarEncoder, CategoryEncoder, DateEncoder, RDSE, etc.) to inherit from.
"""

from typing import Any

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

        def decode(self, input_sdr: list[int]) -> Any:
            return 1

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
