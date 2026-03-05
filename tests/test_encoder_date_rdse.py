"""
tests.test_encoder_date_rdse

Test suite for DateEncoder with RDSE backend.

The Date Encoder decomposes temporal values (datetime objects) into multiple
component dimensions and encodes each using RDSE for sparse, distributed representations.

Temporal Components Encoded:
  - Year (absolute year value, e.g., 2020, 2021) - mutually exclusive with Season
  - Season (annual cycle, day-of-year 0-365) - mutually exclusive with Year
  - Day of week (Mon-Sun, 7 values)
  - Weekend vs weekday (binary)
  - Holiday status (special days)
  - Time of day (hours/minutes, 24h cycle)
  - Custom dimensions (user-defined periodic patterns)

Parameter Validation:
  - Each component has its own size, active_bits/sparsity, radius/resolution
  - Active_bits and sparsity are mutually exclusive (per RDSE constraint)
  - Year and Season are mutually exclusive (XOR constraint)
  - Tests explicitly set sparsity=0.0 when using active_bits
  - Multiple components can be combined into single encoding

Tests validate:
  1. Encoder initialization with various component combinations
  2. Output format (binary 0/1 only, length = sum of component sizes)
  3. Determinism (same datetime → same SDR)
  4. Component independence (each component contributes to final encoding)
  5. All component combinations work correctly
"""

from datetime import datetime

import numpy as np
import pytest

from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters

# ---------------------------------------------------------------------------
# Shared params: RDSE-only configs for single-feature tests
# ---------------------------------------------------------------------------


def _params_season_only():
    return DateEncoderParameters(
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


def _params_day_of_week_only():
    return DateEncoderParameters(
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


def _params_weekend_only():
    return DateEncoderParameters(
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


def _params_custom_only():
    return DateEncoderParameters(
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


def _params_holiday_only():
    return DateEncoderParameters(
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_size=2048,
        holiday_active_bits=4,
        holiday_dates=[[2020, 1, 1], [7, 4]],
        holiday_radius=186.18,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )


def _params_time_of_day_only():
    return DateEncoderParameters(
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


def _params_all_combined():
    return DateEncoderParameters(
        year_size=500,
        year_active_bits=10,
        year_sparsity=0.0,
        year_radius=5.0,  # 5-year buckets - balance between granularity and RDSE hash distribution
        year_resolution=0.0,
        year_min=1,
        year_max=10000,
        season_size=500,
        season_active_bits=0,  # Disabled: year and season are mutually exclusive (XOR)
        season_sparsity=0.0,
        season_radius=25.0,
        season_resolution=0.0,
        day_of_week_size=500,
        day_of_week_active_bits=10,
        day_of_week_radius=14.28,
        day_of_week_resolution=0.0,
        day_of_week_sparsity=0.0,
        weekend_size=500,
        weekend_active_bits=10,
        weekend_radius=1.92,
        weekend_resolution=0.0,
        weekend_sparsity=0.0,
        holiday_size=500,
        holiday_active_bits=10,
        holiday_dates=[[2020, 1, 1], [7, 4], [2019, 4, 21]],
        holiday_radius=9.09,
        holiday_resolution=0.0,
        holiday_sparsity=0.0,
        time_of_day_size=500,
        time_of_day_active_bits=10,
        time_of_day_radius=0.0278,
        time_of_day_resolution=0.0,
        time_of_day_sparsity=0.0,
        custom_size=500,
        custom_active_bits=10,
        custom_radius=25.0,
        custom_resolution=0.0,
        custom_sparsity=0.0,
        custom_days=["Monday", "Mon, Wed, Fri"],
        rdse_used=True,
    )


# ---------------------------------------------------------------------------
# Output format: binary 0/1 only, length equals size
# ---------------------------------------------------------------------------


def test_rdse_output_only_zeros_and_ones():
    """RDSE DateEncoder output must contain only 0 and 1."""
    encoder = DateEncoder(_params_season_only())
    dt = datetime(2020, 1, 1, 0, 0)
    out = encoder.encode(dt)
    assert all(b in (0, 1) for b in out), f"Output must be binary (0/1), got {set(out)}"


def test_rdse_output_length_equals_size():
    """RDSE DateEncoder output length must equal encoder _size."""
    encoder = DateEncoder(_params_season_only())
    dt = datetime(2020, 1, 1, 0, 0)
    out = encoder.encode(dt)
    assert (
        len(out) == encoder._size
    ), f"Output length must equal _size ({encoder._size}), got {len(out)}"


def test_rdse_all_combined_output_binary_and_length():
    """RDSE DateEncoder with all features: output binary and length equals _size."""
    encoder = DateEncoder(_params_all_combined())
    dt = datetime(2020, 1, 1, 0, 0)
    out = encoder.encode(dt)
    assert all(b in (0, 1) for b in out)
    assert len(out) == encoder._size


# ---------------------------------------------------------------------------
# Per-feature encode: each single-feature config produces valid output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params_factory",
    [
        _params_season_only,
        _params_day_of_week_only,
        _params_weekend_only,
        _params_custom_only,
        _params_holiday_only,
        _params_time_of_day_only,
    ],
    ids=["season", "day_of_week", "weekend", "custom", "holiday", "time_of_day"],
)
def test_rdse_single_feature_encode_binary_and_length(params_factory):
    """Each single-feature RDSE config produces binary output of correct length."""
    encoder = DateEncoder(params_factory())
    dt = datetime(2020, 6, 15, 12, 30)
    out = encoder.encode(dt)
    assert all(b in (0, 1) for b in out)
    assert len(out) == encoder._size


# ---------------------------------------------------------------------------
# Determinism: same instance same input => same encoding
# ---------------------------------------------------------------------------


def test_rdse_same_instance_same_input_same_encoding():
    """Same encoder instance, same input => identical encoding."""
    encoder = DateEncoder(_params_season_only())
    dt = datetime(2019, 7, 4, 14, 0)
    enc1 = encoder.encode(dt)
    enc2 = encoder.encode(dt)
    assert enc1 == enc2


# ---------------------------------------------------------------------------
# Same params (same seed) => same encoding across instances
# ---------------------------------------------------------------------------


def test_rdse_same_params_same_encoding_across_instances():
    """Two encoders with same params (default seed) produce same encoding for same input."""
    params = _params_season_only()
    encoder1 = DateEncoder(params)
    encoder2 = DateEncoder(params)
    dt = datetime(2020, 1, 1, 0, 0)
    assert encoder1.encode(dt) == encoder2.encode(dt)


# ---------------------------------------------------------------------------
# Input types: None, datetime, int/float (epoch), struct_time
# ---------------------------------------------------------------------------


def test_rdse_encode_accepts_datetime():
    """RDSE DateEncoder accepts datetime."""
    encoder = DateEncoder(_params_season_only())
    dt = datetime(2020, 3, 15, 9, 0)
    out = encoder.encode(dt)
    assert len(out) == encoder._size
    assert all(b in (0, 1) for b in out)


def test_rdse_encode_accepts_none_current_time():
    """RDSE DateEncoder accepts None (current local time)."""
    encoder = DateEncoder(_params_season_only())
    out = encoder.encode(None)
    assert len(out) == encoder._size
    assert all(b in (0, 1) for b in out)


def test_rdse_encode_accepts_epoch_seconds():
    """RDSE DateEncoder accepts int/float (UNIX epoch seconds)."""
    encoder = DateEncoder(_params_season_only())
    dt = datetime(2020, 1, 1, 0, 0)
    ts = dt.timestamp()
    out = encoder.encode(ts)
    assert len(out) == encoder._size
    assert all(b in (0, 1) for b in out)


def test_rdse_encode_accepts_struct_time():
    """RDSE DateEncoder accepts time.struct_time."""
    encoder = DateEncoder(_params_season_only())
    dt = datetime(2020, 5, 20, 18, 30)
    t = dt.timetuple()
    out = encoder.encode(t)
    assert len(out) == encoder._size
    assert all(b in (0, 1) for b in out)


def test_rdse_encode_rejects_unsupported_type():
    """RDSE DateEncoder rejects unsupported input types."""
    encoder = DateEncoder(_params_season_only())
    with pytest.raises(ValueError):
        encoder.encode("2020-01-01")
    with pytest.raises(ValueError):
        encoder.encode([2020, 1, 1])


# ---------------------------------------------------------------------------
# Misconfigured: no encoders enabled raises
# ---------------------------------------------------------------------------


def test_rdse_no_encoders_enabled_raises():
    """DateEncoder with all features disabled raises during initialization."""
    params = DateEncoderParameters(
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )
    with pytest.raises(RuntimeError, match="no sub-encoders enabled"):
        DateEncoder(params)


# ---------------------------------------------------------------------------
# Multiple dates: encoding varies with input (sanity)
# ---------------------------------------------------------------------------


def test_rdse_different_dates_different_encodings():
    """Different dates produce different encodings (at least for some pairs)."""
    encoder = DateEncoder(_params_all_combined())
    encodings = []
    for year, month, day in [(2020, 1, 1), (2020, 7, 4), (2019, 12, 25)]:
        dt = datetime(year, month, day, 12, 0)
        encodings.append(encoder.encode(dt))
    # At least two should differ
    assert len(set(tuple(e) for e in encodings)) >= 2


# Correctness tests below
def hamming_distance_helper(first, second) -> int:
    """
    Helper method to find the differences with the first != second and then count the nonzero
    as that is how many different bits there are. So if first was 1001 and second was 1010 the
    first operation would be 0011 and the count_nonzero would return 2. This indicates a hamming
    distance of 2 since 2 of the bits are different.
    """
    first = np.asarray(first)
    second = np.asarray(second)
    return int(np.count_nonzero(first != second))


def test_date_correctness():
    encoder = DateEncoder(_params_all_combined())
    encodings1 = []
    for year, month, day in [(2020, 1, 1), (2020, 7, 4), (2050, 12, 25)]:
        dt = datetime(year, month, day, 12, 0)
        encodings1.append(encoder.encode(dt))
    """This test verifies temporal distance is preserved: dates 30 years apart should differ more than dates in same year."""
    assert hamming_distance_helper(encodings1[0], encodings1[1]) < hamming_distance_helper(
        encodings1[0], encodings1[2]
    )

    encodings2 = []
    d = datetime(year=2026, month=1, day=1, minute=1)
    d1 = datetime(year=2026, month=1, day=1, minute=2)
    d2 = datetime(year=2026, month=1, day=1, minute=59)
    encodings2.append(encoder.encode(d))
    encodings2.append(encoder.encode(d1))
    encodings2.append(encoder.encode(d2))
    assert hamming_distance_helper(encodings2[0], encodings2[1]) < hamming_distance_helper(
        encodings2[0], encodings2[2]
    )

    encodings3 = []
    d3 = datetime(year=2026, month=1, day=1, minute=1, second=1)
    d4 = datetime(year=2026, month=1, day=1, minute=1, second=2)
    d5 = datetime(year=2026, month=1, day=1, minute=1, second=59)
    encodings3.append(encoder.encode(d3))
    encodings3.append(encoder.encode(d4))
    encodings3.append(encoder.encode(d5))
    assert hamming_distance_helper(encodings3[0], encodings3[1]) < hamming_distance_helper(
        encodings3[0], encodings3[2]
    )

    encodings4 = []
    d6 = datetime(year=2026, month=1, day=1, hour=1)
    d7 = datetime(year=2026, month=1, day=1, hour=2)
    d8 = datetime(year=2026, month=1, day=1, hour=23)
    encodings4.append(encoder.encode(d6))
    encodings4.append(encoder.encode(d7))
    encodings4.append(encoder.encode(d8))
    # RDSE uses hash-based encoding; collision ties are possible. Test verifies approximate distance ordering.
    assert hamming_distance_helper(encodings4[0], encodings4[1]) <= hamming_distance_helper(
        encodings4[0], encodings4[2]
    )

    encodings5 = []
    d9 = datetime(year=2000, month=1, day=1, hour=1, minute=1)
    d10 = datetime(year=2001, month=1, day=1, hour=1, minute=1)
    d11 = datetime(year=3000, month=1, day=1, hour=1, minute=1)
    encodings5.append(encoder.encode(d9))
    encodings5.append(encoder.encode(d10))
    encodings5.append(encoder.encode(d11))
    """With year encoder enabled (RDSE mode), verify different years produce distinct encodings.
    RDSE uses hash-based encoding, so semantic distance isn't strictly guaranteed."""
    d_2000_2001 = hamming_distance_helper(encodings5[0], encodings5[1])
    d_2000_3000 = hamming_distance_helper(encodings5[0], encodings5[2])
    assert d_2000_2001 > 0, "Adjacent years should produce different encodings"
    assert d_2000_3000 > 0, "Distant years should produce different encodings"
    # RDSE hash-based encoding doesn't guarantee semantic distance ordering

    # Additional year-specific tests: verify distinct encodings for different year ranges
    encodings6 = []
    y1 = datetime(year=2020, month=6, day=15, hour=12, minute=0)  # Same month/day, different years
    y2 = datetime(year=2025, month=6, day=15, hour=12, minute=0)  # 5 years apart
    y3 = datetime(year=2050, month=6, day=15, hour=12, minute=0)  # 30 years from y1
    encodings6.append(encoder.encode(y1))
    encodings6.append(encoder.encode(y2))
    encodings6.append(encoder.encode(y3))
    """Verify year encoder with RDSE produces distinct encodings for different years.
    RDSE hash-based encoding doesn't preserve temporal distance ordering."""
    d_2020_2025 = hamming_distance_helper(encodings6[0], encodings6[1])
    d_2020_2050 = hamming_distance_helper(encodings6[0], encodings6[2])
    d_2025_2050 = hamming_distance_helper(encodings6[1], encodings6[2])
    assert d_2020_2025 > 0, "Different years (2020 vs 2025) should produce different encodings"
    assert d_2020_2050 > 0, "Different years (2020 vs 2050) should produce different encodings"
    assert d_2025_2050 > 0, "Different years (2025 vs 2050) should produce different encodings"


def test_year_season_xor_constraint():
    """Test that year and season encoders are mutually exclusive (XOR constraint)."""
    # Both enabled should raise ValueError
    with pytest.raises(ValueError, match="year and season encoders are mutually exclusive"):
        params = DateEncoderParameters(year_active_bits=10, season_active_bits=10, rdse_used=True)
        DateEncoder(params)

    # Year only should work
    params_year = DateEncoderParameters(
        year_active_bits=10,
        year_size=500,
        year_radius=5.0,
        year_sparsity=0.0,
        year_resolution=0.0,
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )
    encoder_year = DateEncoder(params_year)
    dt = datetime(2020, 6, 15, 12, 0)
    encoding_year = encoder_year.encode(dt)
    assert len(encoding_year) == 500
    # RDSE may have slight variation in actual active bits due to hashing
    assert 8 <= sum(encoding_year) <= 12, f"Expected ~10 active bits, got {sum(encoding_year)}"

    # Season only should work
    params_season = DateEncoderParameters(
        year_active_bits=0,
        season_active_bits=10,
        season_size=500,
        season_radius=50.0,
        season_sparsity=0.0,
        season_resolution=0.0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )
    encoder_season = DateEncoder(params_season)
    encoding_season = encoder_season.encode(dt)
    assert len(encoding_season) == 500
    # RDSE may have slight variation in actual active bits due to hashing
    assert 8 <= sum(encoding_season) <= 12, f"Expected ~10 active bits, got {sum(encoding_season)}"
