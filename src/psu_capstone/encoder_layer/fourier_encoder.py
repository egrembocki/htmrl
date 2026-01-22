from typing import override

import numpy as np
from scipy.fft import fft, ifft

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder
from psu_capstone.sdr_layer.sdr import SDR


class FourierEncoder(BaseEncoder):
    """Encoder that uses Fourier Transform to encode input data."""

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        if output_dim % 2 != 0:
            raise ValueError("Output dimension must be even for Fourier encoding.")

        self.half_output_dim = output_dim // 2
        self._input_dim = input_dim
        self._output_dim = output_dim

        self._rdse = RandomDistributedScalarEncoder()

    @override
    def encode(self, x, output_sdr: SDR) -> None:
        # Perform FFT on the input data
        fft_result = fft(x, n=self.half_output_dim, axis=-1)
