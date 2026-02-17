"""Tests for the Fourier encoder's frequency locality behavior."""

import numpy as np
import pytest

from psu_capstone.encoder_layer.fourier_encoder import FourierEncoder, FourierEncoderParameters
from psu_capstone.sdr_layer.sdr import SDR
from src.utils import hamming_distance, overlap

_SIGNAL_LENGTH = 2048


def _build_encoder(**overrides) -> FourierEncoder:
    """Instantiate a Fourier encoder tuned for 0-200 Hz evaluation, with optional overrides."""

    params = FourierEncoderParameters(
        frequency_ranges=[(0, 200)],
        resolutions_in_ranges=[1.0],
        sparsity_in_ranges=[0.02],
        size=2048,
        total_sparsity=0.02,
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


def _overlap(first: np.ndarray | list[int], second: np.ndarray | list[int]) -> int:
    """Return the count of shared active bits between two dense SDRs."""

    sdr_one = np.asarray(first, dtype=np.int8)
    sdr_two = np.asarray(second, dtype=np.int8)
    return overlap(sdr_one, sdr_two)


def test_identical_frequencies_overlap_completely() -> None:
    """A pure tone should map to the same SDR every time, proving determinism."""

    # Arrange
    encoder = _build_encoder()
    # Act
    sd_first = _encode_frequency(encoder, 75)
    sd_second = _encode_frequency(encoder, 75)

    # Assert
    assert _overlap(sd_first, sd_second) >= 40


def test_close_frequencies_share_more_bits_than_far_ones() -> None:
    """Neighbouring tones should collide more than mid or distant tones to prove locality."""

    # Arrange
    encoder = _build_encoder()
    base = _encode_frequency(encoder, 60)
    close = _encode_frequency(encoder, 61)
    mid = _encode_frequency(encoder, 90)
    far = _encode_frequency(encoder, 5)

    # Act
    close_ratio = _overlap(base, close)
    mid_ratio = _overlap(base, mid)
    far_ratio = _overlap(base, far)

    # Assert
    assert close_ratio >= 30
    assert close_ratio >= mid_ratio >= far_ratio
    assert close_ratio - far_ratio >= 15


def test_identical_frequency_with_different_magnitudes_remains_similar() -> None:
    """Amplitude changes alone should not scramble the SDR bits for a fixed frequency."""

    # Arrange
    encoder = _build_encoder()
    loud = _encode_frequency(encoder, 75, amplitude=2.5)
    quiet = _encode_frequency(encoder, 75, amplitude=0.2)

    # Assert
    assert _overlap(loud, quiet) >= 38


def test_far_frequencies_remain_mostly_orthogonal() -> None:
    """Widely separated tones should produce low overlap, validating global coverage."""

    # Arrange
    encoder = _build_encoder()
    low = _encode_frequency(encoder, 10)
    high = _encode_frequency(encoder, 180)

    # Assert
    assert _overlap(low, high) <= 20


def test_composite_signal_retains_component_information() -> None:
    """A sum of sinusoids should overlap strongly with each constituent tone."""

    # Arrange
    encoder = _build_encoder()
    component_low = _encode_frequency(encoder, 140)
    component_high = _encode_frequency(encoder, 150)
    composite = _encode_signal(
        encoder, [(140, 1.0, 0.0), (150, 1.0, 0.0)]
    )  # freq, amplitude, phase

    # Act
    overlap_low = _overlap(composite, component_low)
    overlap_high = _overlap(composite, component_high)

    # Assert
    assert overlap_low >= 1
    assert overlap_high >= 1


def test_amplitude_modulation_preserves_carrier_bits_more_than_modulator() -> None:
    """Amplitude modulation should keep the carrier SDR more intact than the slow envelope."""

    # Arrange
    encoder = _build_encoder()
    carrier = _encode_frequency(encoder, 10)
    modulator = _encode_frequency(encoder, 2)
    modulated = _encode_amplitude_modulated(encoder, carrier_hz=10, modulator_hz=2, depth=0.2)

    # Act
    overlap_carrier = _overlap(modulated, carrier)
    overlap_modulator = _overlap(modulated, modulator)

    # Assert
    assert overlap_carrier >= 20
    assert overlap_carrier > overlap_modulator


def test_decode_single_tone_returns_expected_frequency() -> None:
    """Decode should identify the strongest frequency when candidates are provided.

    TC-086: Decode should identify the strongest frequency when candidates are provided.

    """

    # Arrange
    encoder = _build_encoder()
    encoded = _encode_frequency(encoder, 60)
    candidates = [20.0, 60.0, 120.0]

    # Act
    decoded = encoder.decode(encoded, candidates=candidates)

    # Assert
    assert "frequencies" in decoded
    freq_range, decoded_value, confidence = decoded["frequencies"][0]
    assert freq_range == (0, 200)
    assert decoded_value == 60.0
    assert confidence > 0.0


def test_decode_rejects_incorrect_sdr_size() -> None:
    """Decode should raise when the SDR size does not match encoder size.

    TC-087: Decode should raise when the SDR size does not match encoder size.
    """

    # Arrange
    encoder = _build_encoder()
    encoded = _encode_frequency(encoder, 60)

    # Act / Assert
    with pytest.raises(ValueError):
        encoder.decode(encoded[:-1])
