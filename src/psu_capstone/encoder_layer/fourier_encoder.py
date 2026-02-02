"""
FFT encoder implementation for HTM core

Indefinite integral proof:  g_f = integral(g_t * exp(-2*pi*i*f*t))

Sum definition: X_k = sum(x_n * exp(-2*pi*i * k * n / N))


https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html

"""

from __future__ import annotations

import copy
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, cast, override

import numpy as np
import pandas as pd
from matplotlib.pyplot import sca
from scipy.fft import fft, fftfreq, ifft

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.log import logger
from psu_capstone.sdr_layer.sdr import SDR


@dataclass
class FourierEncoderParameters:
    """Class to hold fourier encodder parameters

    parameters:
        samples : total number of samples or bucket size -> N
        size : size of the encoder output SDR
        start_time : start time of the time interval -> integral lower bound
        stop_time : stop time of the time interval -> integral upper bound
        interval_size : total number of buckets in the time interval -> N
        time_step_n : the curreent value of x at time n -> n



    """

    # encoder params
    size: int = 2048
    """The size the encoder"""

    total_active_bits: int = 40
    """Total number of active bits in the encoder output SDR."""

    frequency_ranges: list[tuple[int, int]] = field(default_factory=lambda: [(1, 100)])
    """List of frequency ranges to find."""

    magnitude_peaks: list[tuple[int, int]] = field(default_factory=lambda: [])
    """List of magnitudes corresponding to each frequency range or window."""

    active_bits_in_ranges: list[int] = field(default_factory=lambda: [5])
    """The number of active bits per frequency range in the encoder output SDR."""

    resolutions_in_ranges: list[float] = field(default_factory=lambda: [])
    """The resolution per frequency range in the encoder output SDR."""

    seed: int = 42
    """Random seed for reproducibility."""

    #  time domain params
    start_time: float = 0.0
    """Start time of the time interval."""

    stop_time: float = 1.0
    """Stop time of the time interval."""

    total_samples: int = 256
    """Total number of samples in the period."""

    sample_rate: int = 256
    """Sample rate in Hz."""


class FourierEncoder(BaseEncoder[np.ndarray], list[int]):
    """Encoder that uses Fourier Transform on time data. Build RDSE for each frequency component. Assume that time domain is in seconds.

    Args:
        parameters (FourierEncoderParameters, optional): Fourier encoder parameters. Defaults to FourierEncoderParameters().
        dimensions (list[int], optional): List of dimensions for the encoder. Defaults to [].
    """

    def __init__(self, parameters: FourierEncoderParameters = FourierEncoderParameters()):
        """Initialize the encoder with optional Fourier parameters and encoder dimensions."""

        # set the size of the base encoder
        super().__init__(dimensions=[], size=parameters.size)

        self._params = copy.deepcopy(parameters)
        """Fourier encoder local copy of passed parameters."""

        # encoder params
        self._frequency_ranges = self._params.frequency_ranges
        """List of frequency ranges to find."""
        self._size = self._params.size  # default to 2048
        """Size of the encoder."""
        self._total_active_bits = self._params.total_active_bits  # default to 40
        """Total number of active bits in the encoder output SDR."""
        self._active_bits_in_ranges = self._params.active_bits_in_ranges
        """The number of active bits per frequency range in the encoder output SDR."""
        self._resolutions_in_ranges = self._params.resolutions_in_ranges
        """The resolution per frequency range in the encoder output SDR."""
        self._seed = self._params.seed  # default to 42
        """Random seed for reproducibility."""
        self._bucket_sizes: list[int] = []
        """The frequency intervals."""

        # time domain params
        self._start_time = self._params.start_time  # default to 0.0
        """Start time of the time interval."""
        self._stop_time = self._params.stop_time  # default to 1.0
        """Stop time of the time interval."""
        self._period_size = self._stop_time - self._start_time
        """Total time period size in seconds."""
        self._total_samples = self._params.total_samples  # default to 256
        """Total number of samples in the period."""
        self._sample_rate = self._params.sample_rate  # default to 256
        """Sample rate in Hz."""
        self._time_step = self._period_size / self._total_samples
        """Time step between samples."""

    # table this for now, use scipy fft
    def _transform(self, time_data: np.ndarray) -> np.ndarray:
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
            t_even = self._transform(time_data[::2])
            t_odd = self._transform(time_data[1::2])
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

    def _hash_bucket(self, index: int) -> bytes:
        """Generate a hash value for the encoder parameters."""

        return hashlib.sha256(str(index).encode()).digest()

    def _calc_bucket_sizes(self) -> None:
        """Calculate the bucket sizes for each frequency range."""

        for i in range(len(self._frequency_ranges)):

            freq_range = self._frequency_ranges[i]
            start_freq = freq_range[0]
            stop_freq = freq_range[1]

            bucket_size = stop_freq - start_freq

            self._bucket_sizes.append(bucket_size)

    @override
    def encode(self, input_value: np.ndarray) -> list[int]:
        """Transform the input signal via FFT and populate the provided SDR.

        Args:
            input_value (np.ndarray | list[int] | list[float]): The input time-domain signal to be encoded.

        """

        self._calc_bucket_sizes()
        self._total_samples = len(input_value)
        self._sample_rate = len(input_value)
        self._time_step = self._period_size / self._total_samples

        scalar = ScalarEncoder(
            ScalarEncoderParameters(
                size=self._size // 4,
                active_bits=self._total_active_bits // 4,
                radius=0.0,
                resolution=1.0,
            )
        )
        rdse = RandomDistributedScalarEncoder(
            RDSEParameters(
                size=self._size // 4, active_bits=self._total_active_bits // 4, seed=self._seed
            )
        )

        sdr_res = self._size // 2
        sdr_list = [0] * sdr_res
        buffer = [0] * self._size

        start_freq = 0
        stop_freq = 0

        # freq_data = self._transform(input_value)

        freq_data = cast(np.ndarray, fft(input_value))
        freq_data = self._normalize(freq_data)

        # Nyquist frequency limit
        freq_data = freq_data[: self._total_samples // 2]
        freq_buckets = fftfreq(self._total_samples, self._time_step)[: self._total_samples // 2]

        peak_index = np.argmax(np.abs(freq_data))
        peak_freq = freq_buckets[peak_index]
        print(f"FFT Peak Frequency: {peak_freq} Hz")

        magnitude = np.abs(freq_data[int(peak_freq)])

        # TODO: false freq detection check when ranges exceed nyquist limit and peak bucket is invalid

        # TODO: handle multiple frequency ranges and magnitudes :: Each range gets its own frequency and magnitude encoding :: Each range will have active bits set by the user which is current active bits below. This will give N bits per frequency peak found in each range. The user can adjust size of encoder and total active bits to accommodate for sparisity needs.

        # TODO: find how many frequency peak magnitudes exceed thresholds (.1)

        if peak_freq <= self._time_step:
            peak_freq = 0
            magnitude = 0

            sdr_list = [0] * self._size

            return sdr_list
        else:

            sdr_freq = rdse.encode(peak_freq)
            sdr_magnitude = rdse.encode(magnitude)

            sdr_list.extend(sdr_freq)
            sdr_list.extend(sdr_magnitude)

        for freq_range in self._frequency_ranges:
            start_freq = freq_range[0]
            stop_freq = freq_range[1]

            if stop_freq > self._total_samples // 2:
                logger.info(
                    "Frequency range exceeds Nyquist frequency, adjusting to max allowable."
                )
                stop_freq = self._total_samples // 2

            assert stop_freq <= len(freq_data), "Frequency range exceeds FFT output size."
            assert start_freq >= 0, "Start frequency must be non-negative."

            assert (
                start_freq <= peak_freq <= stop_freq
            ), "Peak frequency not in any specified frequency range."

            assert (
                peak_freq <= self._total_samples // 2
            ), "Peak frequency exceeds Nyquist frequency (half the sample rate)."

            # slice freq data to current frequency range
            freq_data = freq_data[start_freq:stop_freq]
            freq_data = np.real(np.abs(freq_data))

            current_active_bits = self._active_bits_in_ranges[
                self._frequency_ranges.index(freq_range)
            ]
            current_res = self._resolutions_in_ranges[self._frequency_ranges.index(freq_range)]
            current_interval_size = self._bucket_sizes[self._frequency_ranges.index(freq_range)]

            assert len(freq_data) >= current_active_bits, "Active bits exceed frequency range size."
            assert len(freq_data) <= sdr_res, "Frequency must be less than encoder size."

            # encode the frequency range slice
            for f in range(current_interval_size):

                # index we want to hash
                buffer[f] = int((freq_data[f]) / current_res) + f

                # check if current freq magnitude exceeds threshold and is contributing to the waveform
                if freq_data[f] < 0.1:

                    continue

                # hash the bucket index value from freq data
                seeds = random.Random(self._hash_bucket(buffer[f]))

                # pick a set of active bits from range from start to fft size
                active_bits = seeds.sample(
                    population=range(start_freq, self._size), k=current_active_bits
                )

                # rdse
                # active_bits = rdse.encode(buffer[f])
                # sdr_list.extend(active_bits)

                # set sdr list with by index
                for bit in active_bits:
                    sdr_list[bit] = 1

        return sdr_list

    def check_params(self, parameters: FourierEncoderParameters) -> FourierEncoderParameters:
        """Check if the provided parameters are valid for the Fourier encoder.

        Args:
            parameters (FourierEncoderParameters): The Fourier encoder parameters to check.

            Returns:
            FourierEncoderParameters: The validated Fourier encoder parameters."""

        params = copy.deepcopy(parameters)

        # Check that frequency ranges and active bits are consistent
        if len(params.frequency_ranges) != len(params.active_bits_in_ranges):
            logger.error("Mismatch between number of frequency ranges and active bits in ranges.")
            return parameters

        if len(params.frequency_ranges) != len(params.resolutions_in_ranges):
            logger.error("Mismatch between number of frequency ranges and resolutions in ranges.")
            return parameters

        if sum(params.active_bits_in_ranges) != params.total_active_bits:
            logger.error("Sum of active bits in ranges does not equal total active bits.")
            return parameters
