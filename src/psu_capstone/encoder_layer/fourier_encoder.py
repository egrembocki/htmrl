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
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.log import logger
from psu_capstone.sdr_layer.sdr import SDR

plt.style.use("seaborn-v0_8-poster")


@dataclass
class FourierEncoderParameters:
    """Class to hold fourier encodder parameters

    parameters:
        samples : total number of samples or bucket size -> N
        size : size of the encoder output SDR
        start_time : start index of the time interval -> integral lower bound
        stop_time : stop index of the time interval -> integral upper bound
        interval_size : total number of buckets in the time interval -> N
        time_step_n : the curreent value of x at time n -> n
        freq_k : the current frequency step in the X_k sum -> k


    """

    size: int = 2048
    """The size the encoder"""

    start_time: int = 0
    """Start index of the time interval. -> Integral lower bound"""

    stop_time: int = 1
    """Stop index of the time interval. -> Integral upper bound"""

    period_size: int = 1
    """The total time period size in seconds."""

    total_samples: int = 2048
    """Total number of samples in the period -> N"""

    time_step_n: int = 0
    """The current value of x at time n. Which time step are we looking for -> n"""

    phase: float = 0.0
    """Phase shift of the signal in radians."""

    # TODO: do we want to track freq lower than 1Hz?
    freq_k: int = 10
    """The current frequency step in the X_k sum. Which frequency are we looking for in Hz"""


class FourierEncoder(BaseEncoder[np.ndarray]):
    """Encoder that uses Fourier Transform on time data. Build RDSE for each frequency component. Assume that time domain is in seconds.

    Args:
        parameters (FourierEncoderParameters, optional): Fourier encoder parameters. Defaults to FourierEncoderParameters().
        dimensions (list[int], optional): List of dimensions for the encoder. Defaults to [].
    """

    def __init__(
        self,
        parameters: FourierEncoderParameters = FourierEncoderParameters(),
        dimensions: list[int] = [],
    ):
        """Initialize the encoder with optional Fourier parameters and encoder dimensions."""

        super().__init__(dimensions)

        self._params = copy.deepcopy(parameters)
        """Fourier encoder local copy of passed parameters."""

        # Local copies of Parameters for easy access
        self._size = self._params.size
        """Size of the encoder."""

        self._start_time = self._params.start_time
        """Start time of the time interval."""
        self._stop_time = self._params.stop_time
        """Stop time of the time interval."""
        self._period_size = self._params.period_size
        """Total time period size in seconds."""
        self._total_samples = self._params.total_samples
        """Total number of samples in the period."""
        self._time_step = self._period_size / self._total_samples
        """Time step between samples."""

        # might be wrong in some cases
        self._frequency_resolution = 1.0 / self._period_size
        """Frequency resolution in Hz."""

        self._time_resolution = self._time_step
        """Time resolution in seconds."""
        self._time_step_n = self._params.time_step_n
        """The current value of x at time n."""
        self._phase = self._params.phase
        """Phase shift of the signal in radians."""
        self._freq_k = self._params.freq_k
        """The current frequency step in the X_k sum."""

        # 2 pi f
        self._omega = 2 * np.pi * self._params.freq_k
        """Angular frequency."""

        self._bucket_idx: int = 0
        """Current bucket index to track frequency resolution buckets."""

        self._buckets: list[int] = []
        """List to hold frequencies in each bucket."""

        # TODO: make sure we are still capturing the frequency components correctly in terms of the original signal
        self._rdse = RandomDistributedScalarEncoder()
        """RDSE encoder for encoding each frequency component."""

        self._fft_sdrs: list[SDR]
        """List to hold SDRs for each frequency component."""

    def transform(self, time_data: np.ndarray) -> np.ndarray:
        """A recursive implementation of the 1D Cooley-Tukey FFT, the input should have a length of power of 2.

            O(n log n) time complexity.

        Reference:
        https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html

        Args:
            time_data (np.ndarray | pd.DataFrame): Input time-domain data.

        Returns:
            np.ndarray: Transformed frequency-domain data. (complex-valued)
        """

        # trim to a power of 2 size
        time_data = self._trim(time_data)

        total_samples = len(time_data)

        if total_samples <= 1:
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
            input_data = input_data.astype(float, copy=False)

        else:
            input_data = np.asarray(input_data, dtype=float)

        # Always operate on a flattened 1D array so FFT math works correctly
        input_data = np.asarray(input_data, dtype=float).reshape(-1)

        total_samples = len(input_data)

        if total_samples & (total_samples - 1) != 0:

            target_size = 1 << (total_samples - 1).bit_length()
            padding = target_size - total_samples
            input_data = np.pad(input_data, (0, padding), mode="constant")

            logger.info(
                (
                    "FFT input adjusted:\n"
                    f"  • Original samples: {total_samples}\n"
                    f"  • Target samples:   {target_size}\n"
                    f"  • Zero padding:     {padding}\n"
                )
            )

        return copy.deepcopy(input_data)

    def get_params(self) -> FourierEncoderParameters:
        """Get the current Fourier encoder parameters.

        Returns:
            FourierEncoderParameters: The current Fourier encoder parameters.
        """

        return copy.deepcopy(self._params)

    def get_fft_sdrs(self) -> list[SDR]:
        """Get the list of SDRs corresponding to each frequency component after encoding.

        Returns:
            list[SDR]: List of SDRs for each frequency component.
        """

        return self._fft_sdrs

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

    # /home/millscb/repos/psu-capstone/data/test.csv

    from psu_capstone.utils import DATA_PATH, PROJECT_ROOT

    ih = InputHandler()

    df_sine = cast(pd.DataFrame, ih.input_data(os.path.join(PROJECT_ROOT, "data", "sine_wave.csv")))

    """
    sin_wave_60 = np.sin(
        60 * (2 * np.pi - 2) * np.linspace(0, 1, 1024, dtype=float, endpoint=False)
    )

    sin_wave_90 = np.sin(
        90 * (2 * np.pi - 3) * np.linspace(0, 1, 1024, dtype=float, endpoint=False)
    )
    """
    # sin_wave = df_sine.to_numpy(dtype=float, copy=False).flatten()

    sample_rate = 5001
    time_step = 1 / sample_rate
    t = np.arange(0, 1, time_step, dtype=float)
    f = 1
    sin_wave = np.sin(2 * np.pi * f * t)

    sin_wave = ih.input_data(os.path.join(PROJECT_ROOT, "data", "sine_wave.csv"))
    sin_wave = cast(pd.DataFrame, sin_wave).to_numpy(dtype=float, copy=False).flatten()

    fourier_encoder = FourierEncoder()
    freq_data = fourier_encoder.transform(sin_wave)

    sin_y = sin_wave

    plt.figure(figsize=(16, 8))
    plt.plot(t, sin_y, "r")  # Plot the sine wave, plot(x, y, 'r') means red line
    plt.title("Sine Wave in Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    samples = len(freq_data)
    samples_n = np.arange(samples)
    period = samples / sample_rate
    freq = samples_n / period

    # divide spectrum in half due to N/2 symmetry
    freq_data = freq_data[: len(freq_data) // 2]
    freq_data = fourier_encoder._normalize(freq_data)
    freq = freq[: samples // 2]

    plt.figure(figsize=(16, 8))
    plt.stem(freq, np.abs(freq_data), linefmt="b-", markerfmt="bo", basefmt="r-")
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(freq_data) // 10))
    plt.title("FFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(which="both", axis="both", linestyle="--", linewidth=0.8)
    plt.show()
