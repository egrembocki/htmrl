from __future__ import annotations

from datetime import datetime

import pytest

from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoderParameters
from psu_capstone.log import logger


@pytest.fixture
def date_encoder_instance() -> DateEncoder:
    """Fixture to create a DateEncoder instance for testing. This can be used to test any defualt DateEncoder object.

    Usage:
        def test_example(date_encoder_instance):
            # Use date_encoder_instance in your test
            pass

    """

    return DateEncoder()


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
        rdse_used=False,
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
    ]

    actual_encoding = []

    expected_encoding = [
        [0, 1, 2, 3],
        [125, 126, 127, 128],
        [111, 112, 113, 114],
        [67, 68, 69, 70],
        [40, 41, 42, 43],
        [38, 39, 40, 41],
        [38, 39, 40, 41],
        [54, 55, 56, 57],
        [53, 54, 55, 56],
    ]

    # Act
    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    # Assert
    assert actual_encoding == expected_encoding, "DateEncoder season test failed!"


def test_day_of_week():

    # Arrange
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
        rdse_used=False,
    )

    date_encoder = DateEncoder(date_params)

    actual_encoding = []

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
    expected_encoding = [[4, 5], [4, 5], [6, 7], [6, 7], [12, 13], [0, 1], [0, 1], [12, 13], [8, 9]]

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    assert actual_encoding == expected_encoding, "DateEncoder day_of_week test failed!"


def test_weekend():
    # Weekend defined as Fri after noon until Sun midnight
    date_params = DateEncoderParameters(
        weekend_size=2048,
        weekend_active_bits=2,
        weekend_radius=39.38,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=False,
    )

    date_encoder = DateEncoder(date_params)

    actual_encoding = []

    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 11, 0],
        [1988, 5, 27, 20, 0],
    ]

    expected_encoding = [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [2, 3],
        [0, 1],
        [0, 1],
        [2, 3],
        [0, 1],
        [2, 3],
    ]

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    assert actual_encoding == expected_encoding, "DateEncoder weekend test failed!"


def test_holiday():
    date_params = DateEncoderParameters(
        holiday_size=2048,
        holiday_active_bits=4,
        holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]],
        holiday_radius=186.18,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=False,
    )
    date_encoder = DateEncoder(date_params)

    actual_encoding = []

    test_case = [
        [2019, 12, 31, 0, 0],
        [2019, 12, 31, 12, 00],
        [2020, 1, 1, 0, 0],
        [2020, 1, 1, 12, 0],
        [2020, 1, 1, 23, 59],
        [2020, 1, 2, 12, 0],
        [2020, 1, 3, 0, 0],
        [2019, 12, 11, 14, 4],
        [2010, 1, 3, 0, 0],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2019, 4, 17, 0, 0],
    ]

    expected_encoding = [
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [2, 3, 4, 5],
        [0, 1, 2, 3],
    ]

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    assert actual_encoding == expected_encoding, "DateEncoder holiday test failed!"


def test_time_of_day():
    date_params = DateEncoderParameters(
        time_of_day_size=1024,
        time_of_day_active_bits=4,
        time_of_day_radius=42.67,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        custom_active_bits=0,
        rdse_used=False,
    )

    date_encoder = DateEncoder(date_params)

    actual_encoding = []

    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 12, 0],
        [2017, 4, 17, 1, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 11, 0],
        [1988, 5, 27, 20, 0],
    ]

    expected_encoding = [
        [0, 1, 2, 3],
        [15, 16, 17, 18],
        [15, 16, 17, 18],
        [0, 1, 2, 3],
        [12, 13, 14, 15],
        [1, 2, 3, 4],
        [23, 24, 25, 26],
        [20, 21, 22, 23],
        [11, 12, 13, 14],
        [20, 21, 22, 23],
    ]

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    assert actual_encoding == expected_encoding, "DateEncoder time_of_day test failed!"


def test_custom_day():
    date_params = DateEncoderParameters(
        custom_size=2048,
        custom_active_bits=2,
        custom_radius=730.0,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["Monday", "Mon, Wed, Fri"],
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        rdse_used=False,
    )

    date_encoder = DateEncoder(date_params)

    actual_encoding = []

    test_case = [
        [2020, 1, 1, 0, 0],
        [2019, 12, 11, 14, 45],
        [2010, 11, 4, 14, 55],
        [2019, 7, 4, 0, 0],
        [2019, 4, 21, 0, 0],
        [2017, 4, 17, 0, 0],
        [2017, 4, 17, 22, 59],
        [1988, 5, 29, 20, 0],
        [1988, 5, 27, 11, 0],
        [1988, 5, 27, 20, 0],
    ]

    expected_encoding = [
        [2, 3],
        [2, 3],
        [0, 1],
        [0, 1],
        [0, 1],
        [2, 3],
        [2, 3],
        [0, 1],
        [2, 3],
        [2, 3],
    ]

    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    assert actual_encoding == expected_encoding, "DateEncoder custom_day test failed!"


def test_all_combined():
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
        rdse_used=False,
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

    actual_encoding = []

    expected_encoding = [
        [0, 1, 100, 101, 200, 201, 300, 301, 400, 401, 500, 501],
        [34, 35, 100, 101, 200, 201, 300, 301, 400, 401, 501, 502],
        [30, 31, 100, 101, 200, 201, 300, 301, 400, 401, 501, 502],
        [18, 19, 100, 101, 200, 201, 300, 301, 400, 401, 500, 501],
        [11, 12, 101, 102, 200, 201, 300, 301, 400, 401, 500, 501],
        [10, 11, 100, 101, 200, 201, 300, 301, 400, 401, 500, 501],
        [10, 11, 100, 101, 200, 201, 300, 301, 400, 401, 502, 503],
        [15, 16, 101, 102, 200, 201, 300, 301, 400, 401, 502, 503],
        [14, 15, 100, 101, 200, 201, 300, 301, 400, 401, 502, 503],
        [14, 15, 100, 101, 200, 201, 300, 301, 400, 401, 501, 502],
    ]

    # Act
    for test in test_case:
        dt = datetime(test[0], test[1], test[2], test[3], test[4])
        encoding = date_encoder.encode(dt)
        sparsity = [i for i, x in enumerate(encoding) if x == 1]
        actual_encoding.append(sparsity)
        logger.info(f"Date: {dt} -> Encoding: {encoding}")

    # Assert
    assert (
        actual_encoding == expected_encoding
    ), "DateEncoder season_day_of_week_combined test failed!"
