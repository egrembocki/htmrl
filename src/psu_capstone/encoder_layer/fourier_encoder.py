"""
FFT encoder implementation for HTM core

General integral proof:  g_f = integral(g_t * exp(-2*pi*i*f*t))

Sum definition: X_k = sum(x_n * exp(-2*pi*i * k * n / N))


https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html





"""

import copy
import os
from dataclasses import dataclass
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.sdr_layer.sdr import SDR

plt.style.use("seaborn-v0_8-poster")


@dataclass
class FourierParameters:
    """Class to hold fourier encodder parameters

    parameters:
        samples : total number of samples or bucket size -> N
        time_step: current integer index of a time step  -> n
        freq_step: current integer index of freq step -> k


    """

    size: int = 2048
    """Number of total bits for encoder"""

    samples: int = 1000
    """Represents the total window or bucket size of the input data set. The size of x in scipy.fft"""

    time_step: int = 0
    """Start at index zero in the input data set. """

    freq_step: int = 1
    """The current frequency step. Which frequency are we looking for in Hz"""


class FourierEncoder(BaseEncoder[np.ndarray]):
    """Encoder that uses Fourier Transform to encode input data."""

    def __init__(
        self, parameters: FourierParameters = FourierParameters(), dimensions: list[int] = []
    ):

        self._params = copy.deepcopy(parameters)
        self._size = self._params.size

        # 2 pi f
        self._omega = 2 * np.pi * self._params.freq_step

        self._rdse = RandomDistributedScalarEncoder()

        self._fft_sdrs: list[SDR]

    def transform(self, input_data: Any) -> np.ndarray:
        """Manual FFT from

        https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html


        """

        input_data = self.trim(input_data)

        sample_size = len(input_data)

        if sample_size == 1:
            return input_data

        else:
            x_even = self.transform(input_data[::2])
            x_odd = self.transform(input_data[1::2])
            factor = np.exp(-2j * np.pi * np.arange(sample_size) / sample_size)

            magnitude = np.concatenate(
                [
                    x_even + factor[: int(sample_size / 2)] * x_odd,
                    x_even + factor[int(sample_size / 2) :] * x_odd,
                ]
            )

        return magnitude

    def normalize(self, input: np.ndarray) -> np.ndarray:

        return np.ones(4)

    def trim(self, input_data: np.ndarray) -> np.ndarray:
        """Trim input array into a power of 2 size"""
        sample_size = len(input_data)

        if sample_size & (sample_size - 1) != 0:
            target_size = 1 << (sample_size - 1).bit_length()
            padding = target_size - sample_size
            input_data = np.pad(input_data, (0, padding), mode="constant")

            self._params.samples = target_size

        return input_data

    @override
    def encode(self, input_value: np.ndarray, output_sdr: SDR) -> None:

        sdr_list: list[SDR] = []

        x_k = self.transform(input_value)

        # x_k = fft(input_value)

        x_k = np.asarray(x_k)

        for x in x_k:
            self._rdse.encode(x, output_sdr)

            sdr_list.append(output_sdr)

        self._fft_sdrs = sdr_list


if __name__ == "__main__":

    ih = InputHandler.get_instance()

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    data_path = os.path.join(project_root, "data", "sin_wave.csv")

    df = ih.input_data(input_source=data_path, required_columns=[])

    sample_rate = 5001
    time_step = 1.0 / sample_rate
    time = np.arange(0, 1, time_step)

    x = np.asarray(df)

    test_sdr = SDR([2048])

    fourier_encoder = FourierEncoder()

    # fourier_encoder.encode(x, test_sdr)

    mag = fourier_encoder.transform(x)

    total_samples = len(x)
    time_n = np.arange(total_samples)
    period = total_samples / sample_rate
    freq = time_n / period

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.stem(freq, abs(mag), "b", basefmt="-b", markerfmt=" ")
    plt.xlabel("Freq (Hz)")
    plt.ylabel("FFT Amplitude | X_k")

    # Get the one-sided specturm
    n_oneside = total_samples // 2
    # get the one side frequency
    f_oneside = freq[:n_oneside]

    # normalize the amplitude
    X_oneside = mag[:n_oneside] / n_oneside

    plt.subplot(122)
    plt.stem(f_oneside, abs(X_oneside), "b", markerfmt=" ", basefmt="-b")
    plt.xlabel("Freq (Hz)")
    plt.ylabel("Normalized FFT Amplitude |X(freq)|")
    plt.tight_layout()
    plt.show()
