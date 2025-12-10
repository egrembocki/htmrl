"""Test suite for the RDSE."""

import pytest

from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def rdse_instance():
    """Fixture to create an RDSE instance for tests."""


def test_rdse_initialization():
    """Test the initialization of the RDSE."""

    parameters = RDSEParameters(
        size=1000, active_bits=0, sparsity=0.05, radius=0.0, resolution=1.23, category=False, seed=0
    )

    encoder = RandomDistributedScalarEncoder(parameters, [1, 1000])
    """Makes sure it is the correct instance"""
    assert isinstance(encoder, RandomDistributedScalarEncoder)


def test_size():
    """Test to make sure the encoder size is correct."""
    parameters = RDSEParameters(
        size=1000, active_bits=0, sparsity=0.05, radius=0.0, resolution=1.23, category=False, seed=0
    )

    encoder = RandomDistributedScalarEncoder(parameters, [1, 1000])
    """Checks that the size is correct."""
    assert encoder._size == 1000


def test_dimensions():
    """Test to make sure the encoder dimensions is correct."""
    parameters = RDSEParameters(
        size=1000, active_bits=0, sparsity=0.05, radius=0.0, resolution=1.23, category=False, seed=0
    )

    encoder = RandomDistributedScalarEncoder(parameters, [1, 1000])
    RandomDistributedScalarEncoder(parameters, [1, 1000])
    """Checks that the dimensions are correct in the encoder."""
    assert encoder.dimensions == [1, 1000]


def test_encode_active_bits():
    """Checks to make sure the proper active bit range is set, the density of the SDR is correct,
    and the size plus dimensions are correct for the SDR after the RDSE encodes it.
    """
    parameters = RDSEParameters(
        size=1000, active_bits=50, sparsity=0.0, radius=0.0, resolution=1.5, category=False, seed=0
    )
    encoder = RandomDistributedScalarEncoder(parameters, [1, 1000])
    a = SDR(encoder.dimensions)
    encoder.encode(10, a)
    """Is the SDR the correct size?"""
    assert a.size == 1000
    """Is the SDR the correct dimensions?"""
    assert a.dimensions == [1, 1000]
    sparse_indices = a.get_sparse()
    sparse_size = len(sparse_indices)
    """Since we have hash collision we are making sure between 45 and 50 bits are encoded."""
    assert 45 <= sparse_size <= 50
    dense_indices = a.get_dense()
    dense_size = len(dense_indices)
    """Is the dense size equal to size?"""
    assert dense_size == 1000


def test_resolution_plus_radius_plus_category():
    """This makes sure proper safe-guards are raised when multiple parameters are entered
    that should not be entered together."""
    parameters = RDSEParameters(
        size=1000, active_bits=50, sparsity=0.0, radius=1.0, resolution=1.5, category=False, seed=0
    )
    """
    Make sure an exception is thrown here since these parameters should
    not be used together.
    """
    with pytest.raises(Exception):
        RandomDistributedScalarEncoder(parameters, [1, 1000])
        parameters.radius = 0
        parameters.category = True
        RandomDistributedScalarEncoder(parameters, [1, 1000])
        parameters.resolution = 0
        parameters.radius = 1
        RandomDistributedScalarEncoder(parameters, [1, 1000])


def test_sparsity_or_activebits():
    """This makes sure that eitehr sparsity or active bits are entered and not both."""
    parameters = RDSEParameters(
        size=1000, active_bits=50, sparsity=1.0, radius=0.0, resolution=1.5, category=False, seed=0
    )
    """Make sure an exception is thrown here since both sparsity and active bits are set."""
    with pytest.raises(Exception):
        RandomDistributedScalarEncoder(parameters, [1, 1000])
    """These should be able to run without an exception or assert since both are not set at once."""
    parameters.sparsity = 0.0
    RandomDistributedScalarEncoder(parameters, [1, 1000])
    parameters.sparsity = 1.0
    parameters.active_bits = 0
    RandomDistributedScalarEncoder(parameters, [1, 1000])


def test_one_of_resolution_radius_category_should_be_entered():
    """We need exactly one of these parameters set otherwise it should return an exception."""
    parameters = RDSEParameters(
        size=1000, active_bits=50, sparsity=1.0, radius=0.0, resolution=0.0, category=False, seed=0
    )
    """Make sure an exception is thrown here since neither radius, resolution, or category were entered."""
    with pytest.raises(Exception):
        RandomDistributedScalarEncoder(parameters, [1, 1000])


def test_one_of_activebit_or_sparsity_is_entered():
    """We need exactly one of these parameters set otherwise it should return an exception."""
    parameters = RDSEParameters(
        size=1000, active_bits=0, sparsity=0.0, radius=1.0, resolution=0.0, category=False, seed=0
    )
    """Make sure an exception is thrown here since neither active bits or sparsity was entered"""
    with pytest.raises(Exception):
        RandomDistributedScalarEncoder(parameters, [1, 1000])


def test_2048_bits_40_active_bits():
    """This test is to check and make sure our current RDSE can handler this many bits."""
    parameters = RDSEParameters(
        size=2048, active_bits=40, sparsity=0.0, radius=1.0, resolution=0.0, category=False, seed=0
    )
    rdse = RandomDistributedScalarEncoder(parameters, [1, 2048])

    s = SDR([1, 2048])
    rdse.encode(10, s)

    """Checking for active bits accounting for hash collisions."""
    assert 35 <= len(s.get_sparse()) <= 40
    print(s.get_sparse)
    """Make sure the density is 2048."""
    assert len(s.get_dense()) == 2048
