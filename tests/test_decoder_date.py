"""
Tests for date decoder
"""

from datetime import datetime

import pytest

from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.log import logger


def test_season():
    # Arrange
    date_params = DateEncoderParameters(
        season_size=366,
        season_active_bits=4,
        season_sparsity=0.0,
        season_radius=91.5,
        season_resolution=0.0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )

    date_encoder = DateEncoder(date_params)

    # [year, month, day, hour, minute]
    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
    ]

    actual_decoded = []

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)

        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    # RDSE is random; assert structure, valid range, and round-trip consistency
    assert len(actual_decoded) == len(test_case)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(
            decoded, tuple
        ), f"Date {test_case[i]}: decoded should be tuple, got {type(decoded)}"
        assert (
            len(decoded) == 1
        ), f"Date {test_case[i]}: one encoder => 1 element, got {len(decoded)}"
        value = decoded[0][0]
        assert (
            0 <= value <= 366
        ), f"Date {test_case[i]}: season (day of year) in [0, 366], got {value}"

    # Same input encoded/decoded twice gives same result (deterministic for this instance)
    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert dec1[0][0] == dec2[0][0], "Round-trip should be deterministic for same encoder instance"


def test_rdse_decode_same_across_instances_with_same_params():
    """With same params, DateEncoder does not pass a seed to RDSE, so all instances use default seed and produce the same decode."""
    date_params = DateEncoderParameters(
        season_active_bits=0,
        day_of_week_size=2048,
        day_of_week_active_bits=2,
        day_of_week_radius=292.57,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )

    dt = datetime(2020, 1, 1, 0, 0)
    decoded_values = []
    for _ in range(5):
        encoder = DateEncoder(date_params)
        encoded = encoder.encode(dt)
        decoded = encoder.decode(encoded)
        decoded_values.append(decoded[0][0])

    # Same params => same default RDSE seed => same decode (deterministic across instances)
    assert (
        len(set(decoded_values)) == 1
    ), f"With same params (no seed override), all instances should produce same decode; got {decoded_values}"


def test_day_of_week():
    # Arrange: only day-of-week encoder, RDSE (decode returns values)
    date_params = DateEncoderParameters(
        season_active_bits=0,
        day_of_week_size=2048,
        day_of_week_active_bits=2,
        day_of_week_radius=292.57,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )

    date_encoder = DateEncoder(date_params)
    # [year, month, day, hour, minute]
    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
    ]

    actual_decoded = []

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    # RDSE is random; assert structure, valid range, and round-trip consistency
    assert len(actual_decoded) == len(test_case)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(
            decoded, tuple
        ), f"Date {test_case[i]}: decoded should be tuple, got {type(decoded)}"
        assert (
            len(decoded) == 1
        ), f"Date {test_case[i]}: one encoder => 1 element, got {len(decoded)}"
        value = decoded[0][0]
        assert (
            0 <= value < 7
        ), f"Date {test_case[i]}: day_of_week (Mon=0..Sun=6) in [0, 7), got {value}"

    # Same input encoded/decoded twice gives same result (deterministic for this instance)
    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert dec1[0][0] == dec2[0][0], "Round-trip should be deterministic for same encoder instance"


# Shared test cases for multi-encoder decoder tests ([year, month, day, hour, minute])
_DECODER_TEST_CASES = [
    [2020, 1, 1, 0, 0],
    [2019, 12, 11, 14, 45],
    [2010, 11, 4, 14, 55],
    [2019, 7, 4, 0, 0],
    [2019, 4, 21, 0, 0],
    [2017, 4, 17, 0, 0],
    [2017, 4, 17, 22, 59],
    [1988, 5, 29, 20, 0],
    [1988, 5, 27, 20, 0],
]


def test_weekend():
    """Decode weekend encoder (RDSE): value 0 or 1 (weekday vs weekend)."""
    date_params = DateEncoderParameters(
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_size=2048,
        weekend_active_bits=2,
        weekend_radius=39.38,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )
    date_encoder = DateEncoder(date_params)
    actual_decoded = []
    for test in _DECODER_TEST_CASES:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    assert len(actual_decoded) == len(_DECODER_TEST_CASES)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(decoded, tuple)
        assert len(decoded) == 1
        value = decoded[0][0]
        assert 0 <= value <= 1, f"Date {_DECODER_TEST_CASES[i]}: weekend in [0, 1], got {value}"

    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert dec1[0][0] == dec2[0][0], "Round-trip should be deterministic for same encoder instance"


def test_custom_days():
    """Decode custom days encoder (RDSE): value 0 or 1 (not in group vs in group)."""
    date_params = DateEncoderParameters(
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_size=2048,
        custom_active_bits=2,
        custom_radius=730.0,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["mon,wed,fri"],
        rdse_used=True,
    )
    date_encoder = DateEncoder(date_params)
    actual_decoded = []
    for test in _DECODER_TEST_CASES:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    assert len(actual_decoded) == len(_DECODER_TEST_CASES)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(decoded, tuple)
        assert len(decoded) == 1
        value = decoded[0][0]
        assert 0 <= value <= 1, f"Date {_DECODER_TEST_CASES[i]}: custom_days in [0, 1], got {value}"

    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert dec1[0][0] == dec2[0][0], "Round-trip should be deterministic for same encoder instance"


def test_holiday():
    """Decode holiday encoder (RDSE): value 0 to ~2 (holiday ramp)."""
    date_params = DateEncoderParameters(
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_size=2048,
        holiday_active_bits=4,
        holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]],
        holiday_radius=186.18,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )
    date_encoder = DateEncoder(date_params)
    actual_decoded = []
    for test in _DECODER_TEST_CASES:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    assert len(actual_decoded) == len(_DECODER_TEST_CASES)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(decoded, tuple)
        assert len(decoded) == 1
        value = decoded[0][0]
        assert (
            0 <= value <= 3
        ), f"Date {_DECODER_TEST_CASES[i]}: holiday ramp in [0, 3], got {value}"

    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert dec1[0][0] == dec2[0][0], "Round-trip should be deterministic for same encoder instance"


def test_time_of_day():
    """Decode time-of-day encoder (RDSE): value 0..24 (hours)."""
    date_params = DateEncoderParameters(
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_size=1024,
        time_of_day_active_bits=4,
        time_of_day_radius=42.67,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        custom_active_bits=0,
        rdse_used=True,
    )
    date_encoder = DateEncoder(date_params)
    actual_decoded = []
    for test in _DECODER_TEST_CASES:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    assert len(actual_decoded) == len(_DECODER_TEST_CASES)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(decoded, tuple)
        assert len(decoded) == 1
        value = decoded[0][0]
        assert (
            0 <= value <= 24
        ), f"Date {_DECODER_TEST_CASES[i]}: time_of_day in [0, 24], got {value}"

    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    assert dec1[0][0] == dec2[0][0], "Round-trip should be deterministic for same encoder instance"


def test_all_combined():
    """Decode with all six encoders enabled (RDSE): season, day_of_week, weekend, custom, holiday, time_of_day."""
    date_params = DateEncoderParameters(
        season_size=100,
        season_active_bits=2,
        season_sparsity=0.0,
        season_radius=25.0,
        season_resolution=0.0,
        day_of_week_size=100,
        day_of_week_active_bits=2,
        day_of_week_radius=14.28,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_size=100,
        weekend_active_bits=2,
        weekend_radius=1.92,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        holiday_size=100,
        holiday_active_bits=2,
        holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]],
        holiday_radius=9.09,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        time_of_day_size=100,
        time_of_day_active_bits=2,
        time_of_day_radius=0.0278,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        custom_size=100,
        custom_active_bits=2,
        custom_radius=25.0,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["Monday", "Mon, Wed, Fri"],
        rdse_used=True,
    )

    date_encoder = DateEncoder(date_params)
    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 20, 0],
        [1988, 5, 27, 11, 0],
    ]

    actual_decoded = []
    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoded = date_encoder.encode(dt)
        decoded = date_encoder.decode(encoded)
        actual_decoded.append(decoded)
        logger.info(f"Date: {dt} -> Encoding: {encoded} -> Decoding: {decoded}")

    # Decode returns tuple of 6 (value, confidence) pairs: season, day_of_week, weekend, custom, holiday, time_of_day
    assert len(actual_decoded) == len(test_case)
    for i, decoded in enumerate(actual_decoded):
        assert isinstance(decoded, tuple), f"Date {test_case[i]}: decoded should be tuple"
        assert (
            len(decoded) == 6
        ), f"Date {test_case[i]}: all combined => 6 elements, got {len(decoded)}"
        season, dow, weekend, custom, holiday, tod = (
            decoded[0][0],
            decoded[1][0],
            decoded[2][0],
            decoded[3][0],
            decoded[4][0],
            decoded[5][0],
        )
        assert 0 <= season <= 366, f"Date {test_case[i]}: season in [0, 366], got {season}"
        assert 0 <= dow < 7, f"Date {test_case[i]}: day_of_week in [0, 7), got {dow}"
        assert 0 <= weekend <= 1, f"Date {test_case[i]}: weekend in [0, 1], got {weekend}"
        assert 0 <= custom <= 1, f"Date {test_case[i]}: custom in [0, 1], got {custom}"
        assert 0 <= holiday <= 3, f"Date {test_case[i]}: holiday in [0, 3], got {holiday}"
        assert 0 <= tod <= 24, f"Date {test_case[i]}: time_of_day in [0, 24], got {tod}"

    dt = datetime(2020, 1, 1, 0, 0)
    enc1 = date_encoder.encode(dt)
    enc2 = date_encoder.encode(dt)
    dec1 = date_encoder.decode(enc1)
    dec2 = date_encoder.decode(enc2)
    for idx in range(6):
        assert dec1[idx][0] == dec2[idx][0], f"Round-trip deterministic for encoder {idx}"
