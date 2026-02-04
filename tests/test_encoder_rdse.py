"""Test suite for the RDSE."""

import numpy as np
import pytest

from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


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
    a = encoder.encode(10)
    """Is the SDR the correct size?"""
    assert len(a) == 1000
    """Is the SDR the correct dimensions?"""
    assert [1, len(a)] == [1, 1000]
    sparse_indices = [i for i, x in enumerate(a) if x == 1]
    sparse_size = len(sparse_indices)
    """Since we have hash collision we are making sure between 45 and 50 bits are encoded."""
    assert 45 <= sparse_size <= 50
    dense_indices = a
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

    a = rdse.encode(10)
    sparse = [i for i, x in enumerate(a) if x == 1]
    """Checking for active bits accounting for hash collisions."""
    assert 35 <= len(sparse) <= 40
    print(sparse)
    """Make sure the density is 2048."""
    assert len(a) == 2048


def test_deterministic_same_seed():
    """
    This test assures that the same value encoded by two different
    RDSE encoders with the same seed will output the same sdrs.
    """
    params = RDSEParameters(
        size=2048,
        active_bits=40,
        seed=40,
    )

    encoder1 = RandomDistributedScalarEncoder(params)
    encoder2 = RandomDistributedScalarEncoder(params)

    sdr1 = encoder1.encode(12.34)
    sdr2 = encoder2.encode(12.34)

    assert sdr1 == sdr2


def test_different_seed_produces_different_sdr_with_same_input_value():
    """
    This test assures that the same value encoded by two different
    RDSE encoders with different seeds will produce different sdrs.
    """
    params = RDSEParameters(
        size=2048,
        active_bits=40,
        seed=40,
    )
    params1 = RDSEParameters(
        size=2048,
        active_bits=40,
        seed=39,
    )
    encoder1 = RandomDistributedScalarEncoder(params)
    encoder2 = RandomDistributedScalarEncoder(params1)

    sdr1 = encoder1.encode(12.34)
    sdr2 = encoder2.encode(12.34)

    assert sdr1 != sdr2


def test_resolution_boundary():
    """The goal of this test to determine if the resolution is functioning
    correctly. The index inside of the rdse is determing by int(input_value/resolution).
    This means that a resolution of 1.0 when encoding 1.0 will be 1, resolution of 1.0 and
    encoding 1.999 will also be 1. But, if you have a resolution of 1.0 and a value of 2.0
    it will be 2. So, the encoding of sdr_c should not be equal to a and b. To add to this
    you can see that a resolution of 1.0 only cares about increments of 1."""
    params = RDSEParameters(
        size=2048,
        active_bits=40,
        resolution=1.0,
        radius=0,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)

    sdr_a = encoder.encode(1.0)
    sdr_b = encoder.encode(1.999)
    sdr_c = encoder.encode(2.0)

    assert sdr_a == sdr_b

    assert sdr_a != sdr_c


def hamming_distance_helper(first: np.ndarray, second: np.ndarray) -> int:
    """
    Helper method to find the differences with the first != second and then count the nonzero
    as that is how many different bits there are. So if first was 1001 and second was 1010 the
    first operation would be 0011 and the count_nonzero would return 2. This indicates a hamming
    distance of 2 since 2 of the bits are different.
    """
    return int(np.count_nonzero(first != second))


def test_locality_checking_mmh3():
    """
    This test compares the mean hamming distances between consecutive encoded values like 1 compared to 2 all
    of the way up to 1000. Then we take the mean of these hamming distances. On top of that it compares 1 through 500
    of encoded values to 9000 through 10000. We then compare these hamming distances. The thought is that the values
    right next to each other should have less bit differences than ones far away.
    """
    import random

    params = RDSEParameters(
        size=2048,
        active_bits=40,
        resolution=1.0,
        radius=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)

    encodings_first = []
    for v in range(1, 1001):
        encodings_first.append(np.array(encoder.encode(float(v))))

    encodings_second = []
    for v in range(9000, 10001):
        encodings_second.append(np.array(encoder.encode(float(v))))

    random_values = random.sample(range(0, 10000), 2000)
    encodings_random = []
    for v in random_values:
        encodings_random.append(np.array(encoder.encode(float(v))))

    # check two encodings by each others hamming distances. small numbers
    consecutive_distances = []
    for v in range(len(encodings_first) - 1):
        d = hamming_distance_helper(encodings_first[v], encodings_first[v + 1])
        consecutive_distances.append(d)
    mean_consecutive = np.mean(consecutive_distances)

    # check two encodings by each others hamming distances, large numbers
    consecutive_distances_large = []
    for v in range(len(encodings_second) - 1):
        d = hamming_distance_helper(encodings_second[v], encodings_second[v + 1])
        consecutive_distances_large.append(d)
    mean_consecutive_large = np.mean(consecutive_distances_large)

    # check the hamming distance between far apart input values
    far_distances = []
    for v in range(1000):
        d = hamming_distance_helper(encodings_first[v], encodings_second[v])
        far_distances.append(d)
    mean_far = np.mean(far_distances)

    # check the hamming distance between two random encodings
    random_distances = []
    for i, j in zip(range(0, 1000), range(1000, 2000)):
        d = hamming_distance_helper(encodings_random[i], encodings_random[j])
        random_distances.append(d)
    mean_random = np.mean(random_distances)

    print("\n")
    print("Consecutive distances mean: ", mean_consecutive)
    print("Far distances mean: ", mean_far)
    print("Large consecutive numbers mean distance: ", mean_consecutive_large)
    print("Random hamming distance mean: ", mean_random)

    assert mean_consecutive < mean_random < mean_far


# By: Dr. Agrawal
def _make_large_encoder(radius: float = 1.0) -> RandomDistributedScalarEncoder:
    params = RDSEParameters(
        size=2048,
        sparsity=0.02,
        radius=radius,
        resolution=0,
        active_bits=0,
        category=False,
        seed=12345,
    )
    return RandomDistributedScalarEncoder(params)


def _overlap_count(first: list[int], second: list[int]) -> int:
    return int(np.count_nonzero(first == second))


# By: Dr. Agrawal
def test_rdse_encodings_are_mostly_orthogonal():
    encoder = _make_large_encoder(radius=1.0)
    import random

    values = [random.randint(0, 100000) for i in range(3000)]
    encodings = [encoder.encode(value) for value in values]

    firsts = random.choices(range(len(values)), k=3000)
    seconds = random.choices(range(len(values)), k=3000)
    overlaps = []
    for i, j in zip(firsts, seconds):
        first = encodings[i]
        second = encodings[j]
        overlaps.append(_overlap_count(first, second))

    orthogonal_ratio = sum(1 for overlap in overlaps if overlap <= 2) / len(overlaps)
    mean_overlap = sum(overlaps) / len(overlaps)

    print(f"Orthogonal ratio: {orthogonal_ratio:.3f}, Mean overlap: {mean_overlap:.3f}")
    assert orthogonal_ratio >= 0.94
    assert mean_overlap <= 1.0


# By: Dr. Agrawal
def test_rdse_no_overlap_outside_radius_large_encoding():
    encoder = _make_large_encoder(radius=1.0)
    values = [i * 0.1 for i in range(200)]
    for value in values:
        outside = value + 5.0
        overlap = _overlap_count(encoder.encode(value), encoder.encode(outside))
        assert overlap < 3


# ---------------------------------------------------------------------------
# Output format and parameter conformance (binary 0/1 only, sparsity/active_bits)
# ---------------------------------------------------------------------------


def test_rdse_encode_output_only_zeros_and_ones():
    """Encoder output must contain only 0 and 1."""
    params = RDSEParameters(
        size=256,
        active_bits=20,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    for value in (0.0, 1.0, 5.0, 100.0):
        out = encoder.encode(value)
        assert all(b in (0, 1) for b in out), f"Output must be binary (0/1), got {set(out)}"


def test_rdse_encode_output_length_equals_size():
    """Encoder output length must equal the configured size."""
    params = RDSEParameters(
        size=512,
        active_bits=30,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    out = encoder.encode(7.0)
    assert len(out) == 512, f"Output length must equal size (512), got {len(out)}"


def test_rdse_encode_output_active_bits_conforms():
    """When active_bits is set, number of 1s should be at most active_bits (hash collisions can reduce it)."""
    size = 1024
    active_bits = 40
    params = RDSEParameters(
        size=size,
        active_bits=active_bits,
        sparsity=0.0,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    for value in (0.0, 1.0, 10.0, 100.0):
        out = encoder.encode(value)
        num_ones = sum(out)
        assert num_ones <= active_bits, f"At most {active_bits} ones expected, got {num_ones}"
        # Hash collisions can reduce; allow down to ~90% of active_bits
        assert num_ones >= int(
            0.9 * active_bits
        ), f"Too few ones ({num_ones}), expected ~{active_bits}"


def test_rdse_encode_output_sparsity_conforms():
    """When sparsity is set (e.g. 0.2), fraction of 1s in output should be approximately that sparsity."""
    size = 1000
    sparsity = 0.2
    params = RDSEParameters(
        size=size,
        active_bits=0,
        sparsity=sparsity,
        radius=1.0,
        resolution=0.0,
        category=False,
        seed=42,
    )
    encoder = RandomDistributedScalarEncoder(params)
    out = encoder.encode(5.0)
    num_ones = sum(out)
    actual_sparsity = num_ones / len(out)
    # Allow tolerance: hash collisions can reduce ones slightly
    assert (
        0.15 <= actual_sparsity <= 0.25
    ), f"Sparsity={sparsity} => ~{sparsity*100}% ones expected, got {actual_sparsity:.3f} ({num_ones}/{len(out)})"
