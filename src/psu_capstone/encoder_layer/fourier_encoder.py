"""
FFT encoder implementation for HTM core

Indefinite integral proof:  g_f = integral(g_t * exp(-2*pi*i*f*t))

Sum definition: X_k = sum(x_n * exp(-2*pi*i * k * n / N))


https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html




"""

import copy
import os
from dataclasses import dataclass
from typing import Any, cast, override

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder
from psu_capstone.log import logger
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
    """The size the encoder"""

    start_time_: int = 0
    """Start index of the time interval"""

    stop_time: int = 1
    """Stop index of the time interval"""

    interval_size: int = 1024  # power of 2
    """The total number of buckets in the time interval. Samples -> N"""

    time_step_n: int = 0
    """The curreent value of x at time n. Which time step are we looking at in seconds"""

    freq_k: int = 10
    """The current frequency step in the X_k sum. Which frequency are we looking for in Hz"""


class FourierEncoder(BaseEncoder[np.ndarray]):
    """Encoder that uses Fourier Transform on time data. Build RDSE for each frequency component."""

    def __init__(
        self, parameters: FourierParameters = FourierParameters(), dimensions: list[int] = []
    ):
        """Initialize the encoder with optional Fourier parameters and SDR dimensions."""
        self._params = copy.deepcopy(parameters)
        self._size = self._params.size

        # 2 pi f
        self._omega = 2 * np.pi * self._params.freq_k

        self._rdse = RandomDistributedScalarEncoder()

        self._fft_sdrs: list[SDR]

    def transform(self, time_data: np.ndarray | pd.DataFrame) -> np.ndarray:
        """A recursive implementation of the 1D Cooley-Tukey FFT, the input should have a length of power of 2.

        https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html
        Args:
            time_data (np.ndarray | pd.DataFrame): Input time-domain data.

        Returns:
            np.ndarray: Transformed frequency-domain data.
        """

        # trim to a power of 2 size
        time_data = self._trim(time_data)
        total_samples = len(time_data)

        if total_samples == 1 if isinstance(time_data, np.ndarray) else 0:
            return cast(np.ndarray, time_data)

        else:
            t_even = self.transform(time_data[::2])
            t_odd = self.transform(time_data[1::2])
            omega = np.exp(-2j * np.pi * np.arange(total_samples) / total_samples)

            freq_data = np.concatenate(
                [
                    t_even + omega[: int(total_samples / 2)] * t_odd,
                    t_even + omega[int(total_samples / 2) :] * t_odd,
                ]
            )
            return freq_data

    def _normalize(self, input: np.ndarray) -> np.ndarray:
        """Normalize the FFT array to a unit vector."""

        # Calculate the normal vector norm (L2 norm)
        norm = np.linalg.norm(input)

        unit_vector = input / norm if norm != 0 else input

        return unit_vector

    def _trim(self, input_data: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Trim input array into a power of 2 size"""

        if isinstance(input_data, pd.DataFrame):

            # convert all pd.DataFrame columns to float types
            input_data = (
                input_data.select_dtypes(include=[np.number]).astype(float).to_numpy(copy=False)
            )

        elif isinstance(input_data, np.ndarray):
            input_data = input_data.astype(float)

        total_samples = len(input_data)

        if total_samples & (total_samples - 1) != 0:

            logger.info(
                f"Input size 0b{total_samples:b} is not a power of 2, padding to next power of 2."
            )

            target_size = 1 << (total_samples - 1).bit_length()
            padding = target_size - total_samples
            input_data = np.pad(input_data, (0, padding), mode="constant")

            self._params.samples = target_size

        return copy.deepcopy(input_data)

    @override
    def encode(self, input_value: np.ndarray, output_sdr: SDR) -> None:
        """Transform the input signal via FFT and populate the provided SDR.

        Args:
            input_value (np.ndarray): The input time-domain signal to be encoded.
            output_sdr (SDR): The SDR object to populate with the encoded data.
        """

        sdr_list: list[SDR] = []

        x_k = self.transform(input_value)

        # x_k = fft(input_value)

        for x in x_k:
            self._rdse.encode(x, output_sdr)

            sdr_list.append(output_sdr)

        self._fft_sdrs = sdr_list


if __name__ == "__main__":

    sin_wave = np.sin(2 * np.pi * 2 * np.linspace(0, 2, 256, dtype=float, endpoint=False))

    time_step = 1 / 128

    t = np.arange(0, 2, time_step, dtype=float)

    fourier_encoder = FourierEncoder()

    freq_data = fourier_encoder.transform(sin_wave)

    # print(freq_data.shape)
    print(freq_data)

    plt.figure(figsize=(12, 6))
    # plt.stem(np.abs(sin_wave), linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.plot(t, sin_wave, "r")  # Plot the sine wave, plot(x, y, 'r') means red line
    plt.title("Sine Wave in Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # plt.title("FFT Magnitude Spectrum")
    # plt.xlabel("Frequency Bin")
    # plt.ylabel("Magnitude")
    # plt.grid()
    # plt.show()
