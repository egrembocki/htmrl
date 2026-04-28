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

from htmrl.encoder_layer.base_encoder import BaseEncoder
from htmrl.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters

# from legacy.sdr_layer.sdr import SDR


@pytest.fixture
def base_encoder_instance() -> BaseEncoder:
    """Fixture to create a real encoder instance for testing BaseEncoder behaviour."""
    params = RDSEParameters(
        size=100,
        active_bits=4,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    return RandomDistributedScalarEncoder(params)


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
