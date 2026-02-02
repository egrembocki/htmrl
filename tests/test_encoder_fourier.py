"""Tests for the Fourier encoder's frequency locality behavior."""

import numpy as np

from psu_capstone.encoder_layer.fourier_encoder import FourierEncoder, FourierEncoderParameters
from psu_capstone.sdr_layer.sdr import SDR

_SIGNAL_LENGTH = 2048


def _build_encoder(**overrides) -> FourierEncoder:
    params = FourierEncoderParameters(
        frequency_ranges=[(0, 200)],
        resolutions_in_ranges=[1],
        active_bits_in_ranges=[20],
        size=2048,
        total_active_bits=40,
        seed=42,
    )

    for key, value in overrides.items():
        setattr(params, key, value)

    return FourierEncoder(params)


def _encode_frequency(encoder: FourierEncoder, frequency: float) -> list[int]:
    time = np.linspace(0, 1, _SIGNAL_LENGTH, endpoint=False)
    signal = np.sin(2 * np.pi * frequency * time)
    return encoder.encode(signal)


def _overlap_ratio(first: list[int], second: list[int]) -> float:
    sdr_one = SDR([len(first)])
    sdr_two = SDR([len(second)])
    sdr_one.set_dense(first)
    sdr_two.set_dense(second)
    overlap = sdr_one.get_overlap(sdr_two)
    return overlap / max(sum(first), 1)


def test_fourier_encoder_nearby_frequencies_share_bits() -> None:
    """59 Hz and 60 Hz signals should overlap heavily."""

    encoder = _build_encoder()
    sd60 = _encode_frequency(encoder, 60)
    sd59 = _encode_frequency(encoder, 59)

    assert _overlap_ratio(sd60, sd59) >= 0.9


def test_fourier_encoder_distant_frequencies_diverge() -> None:
    """60 Hz and 1 Hz signals should stay mostly orthogonal."""

    encoder = _build_encoder()
    sd60 = _encode_frequency(encoder, 60)
    sd_other = _encode_frequency(encoder, 1)

    assert _overlap_ratio(sd60, sd_other) <= 0.4

    # Safety check: nearby frequencies should overlap more than distant ones.
    encoder = _build_encoder()
    close_ratio = _overlap_ratio(
        _encode_frequency(encoder, 60),
        _encode_frequency(encoder, 59),
    )
    far_ratio = _overlap_ratio(
        _encode_frequency(encoder, 60),
        _encode_frequency(encoder, 1),
    )
    assert close_ratio > far_ratio


def test_overlap_falls_with_frequency_delta() -> None:
    """Configure decay so a 50 Hz delta produces ~50% overlap."""

    encoder = _build_encoder(locality_min_ratio=0.5, locality_decay_hz=50.0)

    overlap_close = _overlap_ratio(
        _encode_frequency(encoder, 100),
        _encode_frequency(encoder, 100),
    )
    overlap_mid = _overlap_ratio(
        _encode_frequency(encoder, 100),
        _encode_frequency(encoder, 50),
    )
    overlap_far = _overlap_ratio(
        _encode_frequency(encoder, 100),
        _encode_frequency(encoder, 10),
    )

    assert overlap_close > 0.9
    assert 0.45 <= overlap_mid <= 0.6
    assert overlap_far < overlap_mid
