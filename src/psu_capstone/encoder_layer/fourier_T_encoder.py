import numpy as np
from scipy.ftt import fft, ifft

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomSparseRepresentation
from psu_capstone.sdr_layer.sdr import SDR


class FourierEncoder(BaseEncoder):
    """Encoder that uses Fourier Transform to encode input data."""

    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        if output_dim % 2 != 0:
            raise ValueError("Output dimension must be even for Fourier encoding.")
        self.half_output_dim = output_dim // 2

    def encode(self, x, output_sdr: SDR):
        # Perform FFT on the input data
        fft_result = fft(x, n=self.half_output_dim, axis=-1)

        # Separate real and imaginary parts
        real_part = np.real(fft_result)
        imag_part = np.imag(fft_result)

        # Concatenate real and imaginary parts
        encoded = np.concatenate([real_part, imag_part], axis=-1)

        return encoded
