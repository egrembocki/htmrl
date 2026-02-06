"""Test suite for the Category Encoder"""

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
