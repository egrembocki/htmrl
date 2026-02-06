"""
Tests for RandomDistributedScalarEncoder (RDSE) decode.

decode() returns (value, confidence) by finding the cached encoding with best
overlap to the input SDR. Cache is populated on encode().
"""

import pytest

from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


def test_rdse_decode_returns_tuple_value_confidence():
    """decode() returns (value, confidence) tuple."""
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
    encoded = encoder.encode(5.0)
    decoded = encoder.decode(encoded)
    assert isinstance(decoded, tuple)
    assert len(decoded) == 2
    value, confidence = decoded
    assert isinstance(confidence, (int, float))
    assert 0 <= confidence <= 1


def test_rdse_decode_round_trip_same_value():
    """decode(encode(x)) returns (x, high confidence) for same encoder instance."""
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
    for x in (0.0, 1.0, 5.0, 10.0, 100.0):
        encoded = encoder.encode(x)
        value, confidence = encoder.decode(encoded)
        assert value == x, f"Round-trip: encode({x}) then decode should yield {x}, got {value}"
        assert confidence >= 0.9, f"Round-trip confidence should be high, got {confidence}"


def test_rdse_decode_wrong_size_raises():
    """decode() with wrong-length SDR raises ValueError."""
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
    encoder.encode(1.0)  # populate cache so decode has candidates
    with pytest.raises(ValueError, match="does not match encoder size"):
        encoder.decode([0] * 100)
    with pytest.raises(ValueError, match="does not match encoder size"):
        encoder.decode([0] * 300)


def test_rdse_decode_no_candidates_raises():
    """decode() with no prior encode (empty cache) raises ValueError."""
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
    # No encode() call -> _encoding_cache empty -> no candidates
    with pytest.raises(ValueError, match="No candidate encodings"):
        encoder.decode([0] * 256)
