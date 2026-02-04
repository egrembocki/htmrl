"""
Test suite for DateEncoder with rdse_used=True (RDSE version).

Covers: output format (binary 0/1, length), parameter conformance,
per-feature encoding, all-combined, determinism, seed, input types, and errors.
"""

from datetime import datetime

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
    with pytest.raises(TypeError, match="Unsupported type"):
        encoder.encode("2020-01-01")
    with pytest.raises(TypeError, match="Unsupported type"):
        encoder.encode([2020, 1, 1])


# ---------------------------------------------------------------------------
# Misconfigured: no encoders enabled raises
# ---------------------------------------------------------------------------


def test_rdse_no_encoders_enabled_raises():
    """DateEncoder with all features disabled raises when encoding."""
    params = DateEncoderParameters(
        season_active_bits=0,
        day_of_week_active_bits=0,
        weekend_active_bits=0,
        holiday_active_bits=0,
        time_of_day_active_bits=0,
        custom_active_bits=0,
        rdse_used=True,
    )
    encoder = DateEncoder(params)
    with pytest.raises(RuntimeError, match="no sub-encoders enabled"):
        encoder.encode(datetime(2020, 1, 1, 0, 0))


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
