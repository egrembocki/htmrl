"""Tests for the Fourier encoder's frequency locality behavior."""

import numpy as np

from psu_capstone.encoder_layer.fourier_encoder import FourierEncoder, FourierEncoderParameters
from psu_capstone.sdr_layer.sdr import SDR

_SIGNAL_LENGTH = 2048


def _build_encoder() -> FourierEncoder:
    params = FourierEncoderParameters(
        frequency_ranges=[(0, 200)],
        resolutions_in_ranges=[1],
        active_bits_in_ranges=[20],
        size=2048,
        total_active_bits=40,
        seed=42,
    )

    return FourierEncoder(params)
