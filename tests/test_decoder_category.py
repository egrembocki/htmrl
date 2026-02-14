"""
Tests for CategoryEncoder decode (RDSE path).

CategoryEncoder.decode is only implemented when rdse_used=True; it returns (value, confidence)
where value is the decoded category string (or "NA" for unknown).
"""

import pytest

from psu_capstone.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters


def test_decode_returns_tuple_of_two():
    """Decode returns (value, confidence) tuple."""
    params = CategoryParameters(w=3, category_list=["ES", "GB", "US"], rdse_used=True)
    encoder = CategoryEncoder(params)
    encoded = encoder.encode("US")
    decoded = encoder.decode(encoded)
    assert isinstance(decoded, tuple)
    assert len(decoded) == 2
    value, confidence = decoded
    assert value is not None
    assert isinstance(confidence, (int, float))


def test_decode_value_in_categories_or_na():
    """Decoded value is one of the category strings or 'NA'."""
    categories = ["ES", "GB", "US"]
    params = CategoryParameters(w=3, category_list=categories, rdse_used=True)
    encoder = CategoryEncoder(params)
    valid_values = set(categories) | {"NA"}
    for cat in categories:
        encoded = encoder.encode(cat)
        decoded = encoder.decode(encoded)
        value = decoded[0]
        assert value in valid_values, f"Decoded value {value!r} not in {valid_values}"
    encoded_unknown = encoder.encode("NA")
    decoded_unknown = encoder.decode(encoded_unknown)
    assert decoded_unknown[0] in valid_values


def test_decode_confidence_in_range():
    """Decoded confidence is in [0, 1]."""
    params = CategoryParameters(w=3, category_list=["ES", "GB", "US"], rdse_used=True)
    encoder = CategoryEncoder(params)
    for cat in ["ES", "GB", "US", "NA"]:
        encoded = encoder.encode(cat)
        decoded = encoder.decode(encoded)
        _, confidence = decoded
        assert 0 <= confidence <= 1, f"Confidence {confidence} not in [0, 1]"


def test_decode_round_trip_same_category():
    """Encode then decode returns the same category (round-trip)."""
    categories = ["ES", "GB", "US"]
    params = CategoryParameters(w=3, category_list=categories, rdse_used=True)
    encoder = CategoryEncoder(params)
    for cat in categories:
        encoded = encoder.encode(cat)
        decoded = encoder.decode(encoded)
        value = decoded[0]
        assert value == cat, f"Round-trip: encoded {cat!r}, got back {value!r}"


def test_decode_round_trip_unknown():
    """Encode unknown category then decode returns 'NA'."""
    params = CategoryParameters(w=3, category_list=["ES", "GB", "US"], rdse_used=True)
    encoder = CategoryEncoder(params)
    encoded = encoder.encode("NA")
    decoded = encoder.decode(encoded)
    assert decoded[0] == "NA", f"Unknown should decode to 'NA', got {decoded[0]!r}"


def test_decode_wrong_sdr_size_raises():
    """Decode with wrong SDR length raises."""
    params = CategoryParameters(w=3, category_list=["ES", "GB", "US"], rdse_used=True)
    encoder = CategoryEncoder(params)
    # Encoder size is (len(categories)+1)*w = 4*3 = 12
    with pytest.raises(ValueError, match="does not match"):
        encoder.decode([0] * 10)
    with pytest.raises(ValueError, match="does not match"):
        encoder.decode([0] * 20)
