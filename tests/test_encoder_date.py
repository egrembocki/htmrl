from __future__ import annotations

from dataclasses import dataclass
from typing import List

from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR


@dataclass
class DateValueCase:
    time: List[int]
    bucket: List[float]
    excepted_output: List[int]


def _to_timestamp(parts: List[int]) -> float:
    """
    Mirror C++ DateEncoder::mktime(y, m, d, h, min, s) using your static helper.

    parts is [y, m, d, h, min] or [y, m, d, h, min, s].
    """
    if len(parts) == 5:
        year, mon, day, hr, minute = parts
        sec = 0
    elif len(parts) == 6:
        year, mon, day, hr, minute, sec = parts
    else:
        raise ValueError(f"Invalid time spec: {parts}")

    return DateEncoder.mktime(year, mon, day, hr, minute, sec)


def do_date_value_cases(encoder: DateEncoder, cases: List[DateValueCase]) -> None:
    """
    Port of the C++ helper doDateValueCases.
    For each case:
      - build expected SDR from expected_output
      - encode the given time
      - assert buckets and SDR match expectations
    """
    for c in cases:
        expected = SDR(dimensions=[encoder._size])
        expected.set_sparse(sorted(c.excepted_output))

        ts = _to_timestamp(c.time)

        actual = SDR(dimensions=[encoder._size])
        encoder.encode(ts, actual)

        assert encoder._buckets == c.bucket

        assert actual == expected

        expected.destroy()
        actual.destroy()


def test_season():
    params = DateEncoderParameters(season_width=5, rdse_used=False)
    encoder = DateEncoder(params, [1, 5])

    cases = [
        # date/time                            bucket   expected output
        DateValueCase([2020, 1, 1, 0, 0], [0.0], [0, 1, 2, 3, 4]),  # New Year's Day, midnight
        DateValueCase([2019, 12, 11, 14, 45], [3.0], [0, 1, 2, 3, 19]),  # winter, Wed, afternoon
        DateValueCase([2010, 11, 4, 14, 55], [3.0], [0, 1, 17, 18, 19]),  # Nov 4, fall, Thu
        DateValueCase([2019, 7, 4, 0, 0], [2.0], [10, 11, 12, 13, 14]),  # July 4, summer, holiday
        DateValueCase([2019, 4, 21, 0, 0], [1.0], [6, 7, 8, 9, 10]),  # Easter
        DateValueCase([2017, 4, 17, 0, 0], [1.0], [6, 7, 8, 9, 10]),
        DateValueCase([2017, 4, 17, 22, 59], [1.0], [6, 7, 8, 9, 10]),
        DateValueCase([1988, 5, 29, 20, 0], [1.0], [8, 9, 10, 11, 12]),
        DateValueCase([1988, 5, 27, 20, 0], [1.0], [8, 9, 10, 11, 12]),
    ]

    do_date_value_cases(encoder, cases)


def test_day_of_week():
    params = DateEncoderParameters(day_of_week_width=2, rdse_used=False)
    encoder = DateEncoder(params, [1, 2])

    cases = [
        # date/time                           bucket   expected
        DateValueCase([2020, 1, 1, 0, 0], [2.0], [4, 5]),  # Wed
        DateValueCase([2019, 12, 11, 14, 45], [2.0], [4, 5]),  # Wed
        DateValueCase([2010, 11, 4, 14, 55], [3.0], [6, 7]),  # Thu
        DateValueCase([2019, 7, 4, 0, 0], [3.0], [6, 7]),  # Thu
        DateValueCase([2019, 4, 21, 0, 0], [6.0], [12, 13]),  # Sun
        DateValueCase([2017, 4, 17, 0, 0], [0.0], [0, 1]),  # Mon
        DateValueCase([2017, 4, 17, 22, 59], [0.0], [0, 1]),  # Mon
        DateValueCase([1988, 5, 29, 20, 0], [6.0], [12, 13]),  # Sun
        DateValueCase([1988, 5, 27, 20, 0], [4.0], [8, 9]),  # Fri
    ]

    do_date_value_cases(encoder, cases)


def test_weekend():
    # Weekend defined as Fri after noon until Sun midnight
    params = DateEncoderParameters(weekend_width=2, rdse_used=False)
    encoder = DateEncoder(params, [1, 2])

    cases = [
        # date/time                          bucket   expected
        DateValueCase([2020, 1, 1, 0, 0], [0.0], [0, 1]),  # Wed
        DateValueCase([2019, 12, 11, 14, 45], [0.0], [0, 1]),  # Wed
        DateValueCase([2010, 11, 4, 14, 55], [0.0], [0, 1]),  # Thu
        DateValueCase([2019, 7, 4, 0, 0], [0.0], [0, 1]),  # Thu
        DateValueCase([2019, 4, 21, 0, 0], [1.0], [2, 3]),  # Sun (weekend)
        DateValueCase([2017, 4, 17, 0, 0], [0.0], [0, 1]),  # Mon
        DateValueCase([2017, 4, 17, 22, 59], [0.0], [0, 1]),  # Mon
        DateValueCase([1988, 5, 29, 20, 0], [1.0], [2, 3]),  # Sun evening
        DateValueCase([1988, 5, 27, 11, 0], [0.0], [0, 1]),  # Fri morning
        DateValueCase([1988, 5, 27, 20, 0], [1.0], [2, 3]),  # Fri evening
    ]

    do_date_value_cases(encoder, cases)


def test_holiday():
    params = DateEncoderParameters(
        holiday_width=4, holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]], rdse_used=False
    )
    encoder = DateEncoder(params, [1, 4])

    cases = [
        # date/time                           bucket    expected
        DateValueCase([2019, 12, 31, 0, 0], [0.0], [0, 1, 2, 3]),  # off - 24 hrs before
        DateValueCase([2019, 12, 31, 12, 0], [0.0], [2, 3, 4, 5]),  # 50% ramp before
        DateValueCase([2020, 1, 1, 0, 0], [1.0], [4, 5, 6, 7]),  # on
        DateValueCase([2020, 1, 1, 12, 0], [1.0], [4, 5, 6, 7]),
        DateValueCase([2020, 1, 1, 23, 59], [1.0], [4, 5, 6, 7]),
        DateValueCase([2020, 1, 2, 12, 0], [1.0], [0, 1, 6, 7]),  # ramp after
        DateValueCase([2020, 1, 3, 0, 0], [0.0], [0, 1, 2, 3]),  # off
        DateValueCase([2019, 12, 11, 14, 45], [0.0], [0, 1, 2, 3]),  # ordinary day
        DateValueCase([2010, 11, 4, 14, 55], [0.0], [0, 1, 2, 3]),
        DateValueCase([2019, 7, 4, 0, 0], [1.0], [4, 5, 6, 7]),  # holiday
        DateValueCase([2019, 4, 21, 0, 0], [1.0], [4, 5, 6, 7]),  # Easter
        DateValueCase([2017, 4, 17, 0, 0], [0.0], [0, 1, 2, 3]),
    ]

    do_date_value_cases(encoder, cases)


def test_time_of_day():
    params = DateEncoderParameters(time_of_day_width=4, time_of_day_radius=4.0, rdse_used=False)
    encoder = DateEncoder(params, [1, 4])

    cases = [
        # date/time                             bucket    expected
        DateValueCase([2020, 1, 1, 0, 0], [0.0], [0, 1, 2, 3]),  # 0:00
        DateValueCase([2019, 12, 11, 14, 45], [12.0], [15, 16, 17, 18]),  # ~14.75 → bucket 12
        DateValueCase([2010, 11, 4, 14, 55], [12.0], [15, 16, 17, 18]),
        DateValueCase([2019, 7, 4, 0, 0], [0.0], [0, 1, 2, 3]),
        DateValueCase([2019, 4, 21, 12, 0], [12.0], [12, 13, 14, 15]),
        DateValueCase([2017, 4, 17, 1, 0], [0.0], [1, 2, 3, 4]),  # 1:00
        DateValueCase([2017, 4, 17, 22, 59], [20.0], [0, 1, 2, 23]),  # ~22.98 → bucket 20
        DateValueCase([1988, 5, 29, 20, 0], [20.0], [20, 21, 22, 23]),
        DateValueCase([1988, 5, 27, 11, 0], [8.0], [11, 12, 13, 14]),
        DateValueCase([1988, 5, 27, 20, 0], [20.0], [20, 21, 22, 23]),
    ]

    do_date_value_cases(encoder, cases)


def test_custom_day():
    params = DateEncoderParameters(
        custom_width=2, custom_days=["Monday", "Mon, Wed, Fri"], rdse_used=False
    )
    encoder = DateEncoder(params, [1, 2])

    cases = [
        # date/time                          bucket   expected
        DateValueCase([2020, 1, 1, 0, 0], [1.0], [2, 3]),  # Wed matches "Mon, Wed, Fri"
        DateValueCase([2019, 12, 11, 14, 45], [1.0], [2, 3]),  # Wed
        DateValueCase([2010, 11, 4, 14, 55], [0.0], [0, 1]),  # Thu
        DateValueCase([2019, 7, 4, 0, 0], [0.0], [0, 1]),  # Thu
        DateValueCase([2019, 4, 21, 0, 0], [0.0], [0, 1]),  # Sun
        DateValueCase([2017, 4, 17, 0, 0], [1.0], [2, 3]),  # Mon
        DateValueCase([2017, 4, 17, 22, 59], [1.0], [2, 3]),  # Mon
        DateValueCase([1988, 5, 29, 20, 0], [0.0], [0, 1]),  # Sun
        DateValueCase([1988, 5, 27, 11, 0], [1.0], [2, 3]),  # Fri
        DateValueCase([1988, 5, 27, 20, 0], [1.0], [2, 3]),  # Fri
    ]

    do_date_value_cases(encoder, cases)


def test_combined():
    params = DateEncoderParameters(
        season_width=5,
        day_of_week_width=2,
        weekend_width=2,
        custom_width=2,
        custom_days=["Monday", "Mon, Wed, Fri"],
        holiday_width=2,
        holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]],
        time_of_day_width=4,
        time_of_day_radius=4.0,
        rdse_used=False,
    )
    encoder = DateEncoder(params, [1, 17])

    cases = [
        DateValueCase(
            [2020, 1, 1, 0, 0],  # date/time
            [0, 2, 0, 1, 1, 0],  # buckets
            [0, 1, 2, 3, 4, 24, 25, 34, 35, 40, 41, 44, 45, 46, 47, 48, 49],  # expected
        ),
        DateValueCase(
            [2019, 12, 11, 14, 45],
            [3, 2, 0, 1, 0, 12],
            [0, 1, 2, 3, 19, 24, 25, 34, 35, 40, 41, 42, 43, 61, 62, 63, 64],
        ),
        DateValueCase(
            [2010, 11, 4, 14, 55],
            [3, 3, 0, 0, 0, 12],
            [0, 1, 17, 18, 19, 26, 27, 34, 35, 38, 39, 42, 43, 61, 62, 63, 64],
        ),
        DateValueCase(
            [2019, 7, 4, 0, 0],
            [2, 3, 0, 0, 1, 0],
            [10, 11, 12, 13, 14, 26, 27, 34, 35, 38, 39, 44, 45, 46, 47, 48, 49],
        ),
        DateValueCase(
            [2019, 4, 21, 0, 0],
            [1, 6, 1, 0, 1, 0],
            [6, 7, 8, 9, 10, 32, 33, 36, 37, 38, 39, 44, 45, 46, 47, 48, 49],
        ),
        DateValueCase(
            [2017, 4, 17, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [6, 7, 8, 9, 10, 20, 21, 34, 35, 40, 41, 42, 43, 46, 47, 48, 49],
        ),
        DateValueCase(
            [2017, 4, 17, 22, 59],
            [1, 0, 0, 1, 0, 20],
            [6, 7, 8, 9, 10, 20, 21, 34, 35, 40, 41, 42, 43, 46, 47, 48, 69],
        ),
        DateValueCase(
            [1988, 5, 29, 20, 0],
            [1, 6, 1, 0, 0, 20],
            [8, 9, 10, 11, 12, 32, 33, 36, 37, 38, 39, 42, 43, 66, 67, 68, 69],
        ),
        DateValueCase(
            [1988, 5, 27, 11, 0],
            [1, 4, 0, 1, 0, 8],
            [8, 9, 10, 11, 12, 28, 29, 34, 35, 40, 41, 42, 43, 57, 58, 59, 60],
        ),
        DateValueCase(
            [1988, 5, 27, 20, 0],
            [1, 4, 1, 1, 0, 20],
            [8, 9, 10, 11, 12, 28, 29, 36, 37, 40, 41, 42, 43, 66, 67, 68, 69],
        ),
    ]

    do_date_value_cases(encoder, cases)
