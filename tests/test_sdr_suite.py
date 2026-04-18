# Test Suite: TS-02 (SDR Operations)
"""
tests.test_sdr_suite

Test suite for SDR (Sparse Distributed Representation) class operations.

Validates that SDR objects correctly implement sparse binary representations with
support for multi-dimensional arrays. Tests cover:
- SDR creation with specified dimensions
- SDR storage and retrieval of active bit indices
- Overlap calculation between two SDRs
- Union operations combining multiple SDRs
- Intersection operations finding common active bits
- Sparsity calculation and metrics
- Dimensionality validation
- SDR comparison and equality
- Bit access by dimension

SDRs are fundamental building blocks in HTM systems, serving as the sparse
encoding/representation format that flows through the temporal memory.

These tests ensure SDR operations maintain correctness for all downstream
HTM processing.
"""

import pytest

from legacy.sdr_layer.sdr import SDR
from legacy.sdr_layer.sdr_interface import SDRInterface


@pytest.fixture
def sdr_fixture():
    """Fixture for creating a standard SDR for tests."""
    return SDR([3, 5])


# Test Type: unit test
def test_sdr_creation():
    # TS-02 TC-011
    """[TS-02 TC-011] Test SDR creation and basic properties."""

    # Arrange
    dimensions = [10]

    # Act
    sdr = SDR(dimensions)

    # Assert
    assert sdr.dimensions == [10]
    assert sdr.size == 10
    assert sdr.get_sparse() == []


# Test Type: unit test
def test_sdr_initialization_and_properties(sdr_fixture):
    # TS-02 TC-012
    # Arrange
    sdr = sdr_fixture
    # Act
    size = sdr.size
    dims = sdr.dimensions
    dims_copy = sdr.get_dimensions()
    # Assert
    assert size == 15
    assert dims == [3, 5]
    assert dims_copy == [3, 5]


# Test Type: unit test
def test_sdr_zero_and_dense_sparse(sdr_fixture):
    # TS-02 TC-013
    # Arrange
    sdr = sdr_fixture
    # Act
    sdr.zero()
    dense_zero = sdr.get_dense()
    sparse_zero = sdr.get_sparse()
    sdr.set_dense([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    sparse_after_dense = sdr.get_sparse()
    sdr.set_sparse([1, 3])
    dense_after_sparse = sdr.get_dense()
    # Assert
    assert dense_zero == [0] * 15
    assert sparse_zero == []
    assert sparse_after_dense == [0, 2, 5, 13]
    assert dense_after_sparse[1] == 1 and dense_after_sparse[3] == 1


# Test Type: unit test
def test_sdr_set_coordinates_and_get_coordinates(sdr_fixture):
    # TS-02 TC-014
    # Arrange
    sdr = sdr_fixture
    coords = [[0, 1, 2], [2, 1, 4]]
    # Act
    sdr.set_coordinates(coords)
    out = sdr.get_coordinates()
    # Assert
    assert out == coords


# Test Type: unit test
def test_sdr_reshape(sdr_fixture):
    # TS-02 TC-015
    # Arrange
    sdr = sdr_fixture
    sdr.set_dense([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    # Act
    sdr.reshape([3, 5])
    # Assert
    assert sdr.dimensions == [3, 5]
    assert sdr.size == 15


# Test Type: unit test
def test_sdr_at_byte(sdr_fixture):
    # TS-02 TC-016
    # Arrange
    sdr = sdr_fixture
    sdr.set_dense([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    sdr.reshape([3, 5])
    # Act & Assert
    assert sdr.at_byte([0, 0]) == 1
    assert sdr.at_byte([1, 1]) == 0
    assert sdr.at_byte([2, 4]) == 1


# Test Type: unit test
def test_sdr_set_sdr(sdr_fixture):
    # TS-02 TC-017
    # Arrange
    sdr1 = SDR([4])
    sdr2 = SDR([4])
    sdr2.set_sparse([1, 3])
    # Act
    sdr1.set_sdr(sdr2)
    # Assert
    assert sdr1.get_sparse() == [1, 3]


# Test Type: unit test
def test_sdr_metrics(sdr_fixture):
    # TS-02 TC-018
    # Arrange
    sdr = SDR([5])
    sdr.set_sparse([0, 2, 4])
    # Act
    s = sdr.get_sum()
    sparsity = sdr.get_sparsity()
    # Assert
    assert s == 3
    assert sparsity == 3 / 5


# Test Type: unit test
def test_sdr_get_overlap(sdr_fixture):
    # TS-02 TC-019
    # Arrange
    sdr1 = SDR([5])
    sdr2 = SDR([5])
    sdr1.set_sparse([1, 2, 3])
    sdr2.set_sparse([2, 3, 4])
    # Act
    overlap = sdr1.get_overlap(sdr2)
    # Assert
    assert overlap == 2


# Test Type: unit test
def test_sdr_intersection_and_union(sdr_fixture):
    # TS-02 TC-020
    # Arrange
    sdr1 = SDR([3])
    sdr2 = SDR([3])
    sdr3 = SDR([3])
    sdr1.set_sparse([0, 1])
    sdr2.set_sparse([1, 2])
    sdr3.set_sparse([0, 2])
    # Act
    sdr1.intersection([sdr1, sdr2, sdr3])
    intersection_result = sdr1.get_sparse()
    sdr1.set_union([sdr1, sdr2, sdr3])

    # Assert
    assert intersection_result == []
    assert sdr1.get_sparse() == [0, 1, 2]


# Test Type: unit test
def test_sdr_concatenate(sdr_fixture):
    # TS-02 TC-021
    # Arrange
    sdr1 = SDR([2])
    sdr2 = SDR([2])
    sdr1.set_dense([1, 0])
    sdr2.set_dense([0, 1])
    sdr_cat = SDR([4])
    # Act
    sdr_cat.concatenate([sdr1, sdr2], axis=0)
    result = sdr_cat.get_dense()
    # Assert
    assert result == [1, 0, 0, 1]


# Test Type: unit test
def test_sdr_callbacks(sdr_fixture):
    # TS-02 TC-022
    # Arrange
    sdr = SDR([2])
    called = []

    def cb():
        called.append(True)

    # Act
    idx = sdr.add_on_change_callback(cb)
    sdr.set_dense([1, 0])
    # Assert
    assert called

    # Act
    sdr.remove_on_change_callback(idx)
    called.clear()
    sdr.set_dense([0, 1])
    # Assert
    assert not called

    # Act
    sdr.add_destroy_callback(cb)
    sdr.destroy()
    # Assert
    assert called


# Test Type: unit test
def test_sdr_randomize_add_noise_kill_cells(sdr_fixture):
    # TS-02 TC-023
    # Arrange
    sdr = SDR([10])
    # Act
    sdr.randomize(0.2)
    sum_after_random = sdr.get_sum()
    before = set(sdr.get_sparse())
    sdr.add_noise(0.5)
    after = set(sdr.get_sparse())
    sdr.kill_cells(0.5)
    sum_after_kill = sdr.get_sum()
    # Assert
    assert 0 < sum_after_random <= 10
    assert before != after or sum_after_random == 0
    assert sum_after_kill <= 5


# Test Type: unit test
def test_sdr_eq_repr(sdr_fixture):
    # TS-02 TC-024
    # Arrange
    sdr1 = SDR([3])
    sdr2 = SDR([3])
    sdr1.set_dense([1, 0, 1])
    sdr2.set_dense([1, 0, 1])
    # Act
    eq_result = sdr1 == sdr2
    repr_result = repr(sdr1)
    # Assert
    assert eq_result
    assert "SDR(dimensions=[3], size=3, active=2)" in repr_result


# Test Type: unit test
def test_sdr_set_and_get_sparse():
    # TS-02 TC-025
    """[TS-02 TC-025] Test setting and getting sparse representation."""

    # Arrange
    dimensions = [10]
    sdr = SDR(dimensions)

    # Act
    sdr.set_sparse([1, 3, 5])

    # Assert
    assert sdr.get_sparse() == [1, 3, 5]


# Test Type: unit test
def test_sdr_zero():
    # TS-02 TC-026
    """[TS-02 TC-026] Test zeroing the SDR."""

    # Arrange
    dimensions = [10]
    sdr = SDR(dimensions)
    sdr.set_sparse([2, 4, 6])

    # Act
    sdr.zero()

    # Assert
    assert sdr.get_sparse() == []


# Test Type: unit test
def test_sdr_set_dense():
    # TS-02 TC-027
    """[TS-02 TC-027] Test setting dense representation and converting to sparse."""

    # Arrange
    dimensions = [5]
    sdr = SDR(dimensions)
    dense_representation = [0, 1, 0, 1, 1]

    # Act
    sdr.set_dense(dense_representation)

    # Assert
    assert sdr.get_sparse() == [1, 3, 4]


# Test Type: unit test
def test_sdr_64_32_init():
    # TS-02 TC-028
    """[TS-02 TC-028] Test SDR creation with dimensions [64, 32]."""

    # Arrange
    dimensions = [64, 32]

    # Act
    sdr = SDR(dimensions)
    sdr.randomize(0.02)
    test_bits = sdr.get_sparse()

    # Assert
    assert sdr.dimensions == [64, 32]
    assert sdr.size == 2048
    assert sdr.get_sparse() == test_bits


# Test Type: unit test
def test_sdr_destroy():
    # TS-02 TC-029
    """[TS-02 TC-029] Test SDR destruction."""

    # Arrange
    dimensions = [10]
    sdr = SDR(dimensions)
    sdr.randomize(0.3)

    # Act
    sdr.destroy()

    # Assert
    assert sdr.dimensions == []
    assert sdr.size == 0


# Test Type: unit test
def test_sdr_interface_object(sdr_fixture: SDR):
    # TS-02 TC-030
    """Test that SDR implements SDRInterface."""

    # Arrange
    sdr = sdr_fixture

    # Assert
    assert isinstance(sdr, SDRInterface)
