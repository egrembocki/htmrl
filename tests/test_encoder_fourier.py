"""Tests for the Fourier encoder's frequency locality behavior."""

import numpy as np

from psu_capstone.encoder_layer.fourier_encoder import FourierEncoder, FourierEncoderParameters
from psu_capstone.sdr_layer.sdr import SDR

_SIGNAL_LENGTH = 2048


def _build_encoder(**overrides) -> FourierEncoder:
    """Instantiate a Fourier encoder tuned for 0-200 Hz evaluation, with optional overrides."""

    params = FourierEncoderParameters(
        frequency_ranges=[(0, 200)],
        resolutions_in_ranges=[1.0],
        sparsity_in_ranges=[0.02],
        size=4096,
    )

    for key, value in overrides.items():
        setattr(params, key, value)

    return FourierEncoder(params)


def _encode_signal(
    encoder: FourierEncoder,
    components: list[tuple[float, float, float]],
) -> list[int]:
    """Encode a composite signal from (frequency, amplitude, phase) tuples."""

    time = np.linspace(0, 1, _SIGNAL_LENGTH, endpoint=False)
    signal = np.zeros_like(time)
    for frequency, amplitude, phase in components:
        signal += amplitude * np.sin(2 * np.pi * frequency * time + phase)
    return encoder.encode(signal)


def _encode_frequency(
    encoder: FourierEncoder, frequency: float, amplitude: float = 1.0
) -> list[int]:
    """Encode a single sinusoid so tests can probe individual spectral behaviors."""

    return _encode_signal(encoder, [(frequency, amplitude, 0.0)])


def _encode_amplitude_modulated(
    encoder: FourierEncoder,
    carrier_hz: float,
    modulator_hz: float,
    depth: float = 0.5,
) -> list[int]:
    """Encode an amplitude-modulated carrier signal."""

    time = np.linspace(0, 1, _SIGNAL_LENGTH, endpoint=False)
    envelope = 1.0 + depth * np.sin(2 * np.pi * modulator_hz * time)
    signal = envelope * np.sin(2 * np.pi * carrier_hz * time)
    return encoder.encode(signal)


def _overlap_ratio(first: list[int], second: list[int]) -> float:
    """Return the overlap of two dense SDRs relative to the active bits in the first vector."""

    sdr_one = SDR([len(first)])
    sdr_two = SDR([len(second)])
    sdr_one.set_dense(first)
    sdr_two.set_dense(second)
    overlap = sdr_one.get_overlap(sdr_two)
    return overlap / max(sum(first), 1)


def test_identical_frequencies_overlap_completely() -> None:
    """A pure tone should map to the same SDR every time, proving determinism."""

    encoder = _build_encoder()
    sd_first = _encode_frequency(encoder, 75)
    sd_second = _encode_frequency(encoder, 75)

    assert _overlap_ratio(sd_first, sd_second) >= 0.99


def test_close_frequencies_share_more_bits_than_far_ones() -> None:
    """Neighbouring tones should collide more than mid or distant tones to prove locality."""

    encoder = _build_encoder()
    base = _encode_frequency(encoder, 60)
    close = _encode_frequency(encoder, 61)
    mid = _encode_frequency(encoder, 90)
    far = _encode_frequency(encoder, 5)

    close_ratio = _overlap_ratio(base, close)
    mid_ratio = _overlap_ratio(base, mid)
    far_ratio = _overlap_ratio(base, far)

    assert close_ratio >= 0.9
    assert close_ratio > mid_ratio > far_ratio
    assert far_ratio <= 0.38


def test_identical_frequency_with_different_magnitudes_remains_similar() -> None:
    """Amplitude changes alone should not scramble the SDR bits for a fixed frequency."""

    encoder = _build_encoder()
    loud = _encode_frequency(encoder, 75, amplitude=2.5)
    quiet = _encode_frequency(encoder, 75, amplitude=0.2)

    assert _overlap_ratio(loud, quiet) >= 0.99


def test_far_frequencies_remain_mostly_orthogonal() -> None:
    """Widely separated tones should produce low overlap, validating global coverage."""

    encoder = _build_encoder()
    low = _encode_frequency(encoder, 10)
    high = _encode_frequency(encoder, 180)

    assert _overlap_ratio(low, high) <= 0.35


def test_composite_signal_retains_component_information() -> None:
    """A sum of sinusoids should overlap strongly with each constituent tone."""

    encoder = _build_encoder()
    component_low = _encode_frequency(encoder, 30)
    component_high = _encode_frequency(encoder, 90)
    composite = _encode_signal(encoder, [(30, 1.0, 0.0), (90, 0.8, np.pi / 4)])
    unrelated = _encode_frequency(encoder, 5)

    overlap_low = _overlap_ratio(composite, component_low)
    overlap_high = _overlap_ratio(composite, component_high)
    overlap_unrelated = _overlap_ratio(composite, unrelated)

    assert overlap_low >= 0.55
    assert overlap_high >= 0.55
    assert overlap_low > overlap_unrelated
    assert overlap_high > overlap_unrelated


def test_amplitude_modulation_preserves_carrier_bits_more_than_modulator() -> None:
    """Amplitude modulation should keep the carrier SDR more intact than the slow envelope."""

    encoder = _build_encoder()
    carrier = _encode_frequency(encoder, 120)
    modulated = _encode_amplitude_modulated(encoder, carrier_hz=120, modulator_hz=5, depth=0.6)
    modulator = _encode_frequency(encoder, 5)

    overlap_carrier = _overlap_ratio(modulated, carrier)
    overlap_modulator = _overlap_ratio(modulated, modulator)

    assert overlap_carrier >= 0.55
    assert overlap_carrier > overlap_modulator
