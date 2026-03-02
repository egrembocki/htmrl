"""
Test suite for Category Encoder.

The Category Encoder produces one-hot style encodings for categorical data (discrete values).
Each category gets a unique SDR representation, with no semantic relationship between categories.

Key Features:
  - One-hot or sparse encoding for categorical inputs
  - Each category receives unique bit pattern
  - No overlap expected between different categories (orthogonal representations)
  - Deterministic encoding (same category → same SDR)
  - Width parameter (w) controls encoding sparsity

Parameter Validation:
  - size: total bits in encoded output
  - w: number of active bits per encoding
  - category_list: list of valid categories
  - rdse_used: whether to use RDSE internally (False for basic category)

Tests validate:
  1. Encoder initialization with valid category list
  2. Encoding specific categories produces expected SDR
  3. Output format (binary only, correct length)
  4. Orthogonality (different categories → no overlap)
  5. Determinism and consistency
"""

import pytest

from psu_capstone.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters


@pytest.fixture
def category_instance():
    """Fixture to create a Category encoder instance for tests
    This is used for teardown purposes if needed in the future.
    """


def test_category_initialization():
    """
    This tests to make sure the Category Encoder can succesfully be created.
    Note: there is an optional dimensions parameter not being used here.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, category_list=categories, rdse_used=False)
    e = CategoryEncoder(parameters=parameters)

    assert isinstance(e, CategoryEncoder)
    """Checking if the instance is correct."""


def test_encode_us():
    """
    This encodes the category "US" into an SDR of 1x12. That bit number is determined from
    3 categories and 1 unknown category. This is w or width of 3 times 4 which is 12 long.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, category_list=categories, rdse_used=False)
    e = CategoryEncoder(parameters=parameters)
    a = e.encode("US")
    """This makes sure our encoding is accurate and matches a known SDR outcome."""
    assert a == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]


def test_unknown_category():
    """
    This encodes an unknown category. Here we use "NA" which as you can see is not one of
    the categories specified.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, category_list=categories, rdse_used=False)
    e = CategoryEncoder(parameters=parameters)
    a = e.encode("NA")
    """This makes sure our encoding is accurate and matches a known SDR outcome."""
    assert a == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_encode_es():
    """
    This is almost idential to the "US" encoding, I am just deomonstrating that the encoding
    shows different active bits for different categories.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, category_list=categories, rdse_used=False)
    e = CategoryEncoder(parameters=parameters)
    a = e.encode("ES")
    """This makes sure our encoding is accurate and matches a known SDR outcome."""
    assert a == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_with_width_one():
    """This test is used to show how SDR outputs look with a single w or width."""
    categories = ["cat1", "cat2", "cat3", "cat4", "cat5"]
    """Note: I think since width is 1, each category is 1 bit and there is the first bit that is the unknown category."""
    expected = [
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
    parameters = CategoryParameters(w=1, category_list=categories, rdse_used=False)
    e = CategoryEncoder(parameters=parameters)
    i = 0
    """The respective category should equal their index of expected results."""
    for cat in categories:
        a = e.encode(cat)
        assert a == expected[i]
        i = i + 1


def test_rdse_used():
    """
    This test uses the RDSE and demonstrates that the same encoder encoding a category twice
    to two different SDRs yields the same encoding. This is important since it shows we can
    decode this if needed and get the category back from our SDR.
    """
    categories = ["ES", "GB", "US"]
    parameters = CategoryParameters(w=3, category_list=categories)
    e1 = CategoryEncoder(parameters=parameters)
    """These asserts just check that both SDRs are identical when the same category is encoded."""
    a1 = e1.encode("ES")
    a2 = e1.encode("ES")
    assert a1 == a2
    a1 = e1.encode("GB")
    a2 = e1.encode("GB")
    assert a1 == a2
    a1 = e1.encode("US")
    a2 = e1.encode("US")
    assert a1 == a2
    a1 = e1.encode("NA")
    a2 = e1.encode("NA")
    assert a1 == a2


# ---------------------------------------------------------------------------
# Output format and parameter conformance (binary 0/1 only, length)
# ---------------------------------------------------------------------------


def test_category_encode_output_only_zeros_and_ones():
    """CategoryEncoder output must contain only 0 and 1."""
    categories = ["ES", "GB", "US"]
    for rdse_used in (False, True):
        parameters = CategoryParameters(w=3, category_list=categories, rdse_used=rdse_used)
        encoder = CategoryEncoder(parameters)
        for cat in categories + ["NA"]:
            out = encoder.encode(cat)
            assert all(
                b in (0, 1) for b in out
            ), f"Output must be binary (0/1), rdse_used={rdse_used}, cat={cat!r}, got {set(out)}"


def test_category_encode_output_length_equals_size():
    """CategoryEncoder output length must equal (num_categories + 1) * w."""
    categories = ["ES", "GB", "US"]
    w = 4
    parameters = CategoryParameters(w=w, category_list=categories, rdse_used=False)
    encoder = CategoryEncoder(parameters)
    expected_size = (len(categories) + 1) * w  # +1 for unknown
    out = encoder.encode("US")
    assert (
        len(out) == expected_size
    ), f"Output length must equal (1+len(categories))*w = {expected_size}, got {len(out)}"


"""
tests.test_decoder_category

Test suite for CategoryEncoder decoding functionality.

Validates that CategoryEncoder correctly decodes SDRs back to original category strings.
Tests cover:
- decode() returns (value, confidence) tuple for RDSE-based encoders
- Decoded values match original encoded categories
- Confidence scores reflect encoding strength
- Unknown/undecodable SDRs return "NA" for value
- decode() is only implemented when rdse_used=True parameter is set

Note: CategoryEncoder.decode() follows the standard decoder interface where decode()
returns a tuple of (decoded_value, confidence) where decoded_value is the category string.

These tests validate the reverse transformation from SDR representations back to
interpretable category values.
"""


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
