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
from dataclasses import dataclass, field
from typing import Any, cast, override

import numpy as np
import pandas as pd
from scipy.fft import fft, ifft

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

    frequency_ranges: list[tuple[int, int]] = field(default_factory=lambda: [])
    """List of frequency ranges to find."""

    magnitude_peaks: list[tuple[int, int]] = field(default_factory=lambda: [])
    """List of magnitudes corresponding to each frequency range or window."""

    active_bits_in_ranges: list[int] = field(default_factory=lambda: [])
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

        super().__init__(dimensions=[], size=parameters.size)  # set the size of the base encoder

        self._params = copy.deepcopy(parameters)
        """Fourier encoder local copy of passed parameters."""

        # encoder params
        self._frequency_ranges = self._params.frequency_ranges
        """List of frequency ranges to find."""
        self._size = self._params.size
        """Size of the encoder."""
        self._active_bits_in_ranges = self._params.active_bits_in_ranges
        """The number of active bits per frequency range in the encoder output SDR."""
        self._resolutions_in_ranges = self._params.resolutions_in_ranges
        """The resolution per frequency range in the encoder output SDR."""
        self._seed = self._params.seed
        """Random seed for reproducibility."""
        self._bucket_sizes: list[int] = []
        """The number of frequency ranges or buckets."""

        # time domain params
        self._start_time = self._params.start_time
        """Start time of the time interval."""
        self._stop_time = self._params.stop_time
        """Stop time of the time interval."""
        self._period_size = self._stop_time - self._start_time
        """Total time period size in seconds."""
        self._total_samples = self._params.total_samples
        """Total number of samples in the period."""
        self._sample_rate = self._params.sample_rate
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

        self._bucket_sizes = []

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
        self._total_samples = len(input_value)
        self._sample_rate = len(input_value)  # assume sample rate equals number of samples for now

        scalar = ScalarEncoder(ScalarEncoderParameters(size=100, active_bits=2))
        rdse = RandomDistributedScalarEncoder(RDSEParameters(size=100, active_bits=2))

        self._calc_bucket_sizes()

        sdr_list = [0] * self._size

        buckets_idx = [0] * self._size

        start_freq = 0
        stop_freq = 0

        # freq_data = self._transform(input_value)
        freq_data = cast(np.ndarray, fft(input_value))
        freq_data = self._normalize(freq_data)
        freq = np.arange(self._total_samples, dtype=float) * (
            self._sample_rate / self._total_samples
        )

        print(f"FFT Peak Frequency: {freq[np.argmax(np.abs(freq_data))]} Hz")

        peak_freq = freq[np.argmax(np.abs(freq_data))]
        magnitude = np.abs(freq_data[np.argmax(np.abs(freq_data))])

        sdr_freq = scalar.encode(peak_freq)
        sdr_magnitude = scalar.encode(magnitude)
        sdr_list.extend(sdr_freq)
        sdr_list.extend(sdr_magnitude)

        for freq_range in self._frequency_ranges:
            start_freq = freq_range[0]
            stop_freq = freq_range[1]

            assert stop_freq <= len(freq_data), "Frequency range exceeds FFT output size."
            assert start_freq >= 0, "Start frequency must be non-negative."

            freq_data = freq_data[start_freq:stop_freq]

            freq_data = np.real(np.abs(freq_data))  # slice of fft data to encode

            # encode the frequency range slice
            for f in range(start_freq, len(freq_data)):

                # index we want to hash
                buckets_idx[f] = (
                    freq_data[f]
                    / self._resolutions_in_ranges[self._frequency_ranges.index(freq_range)]
                )

                # hash the bucket index value from freq data
                # seeds = random.Random(self._hash_bucket(buckets_idx[f]))

                # pick a set of active bits from range from start to stop frequency
                # active_bits = seeds.sample(
                #   population=range(start_freq, stop_freq),
                #   k=self._params.active_bits_in_ranges[self._frequency_ranges.index#(freq_range)],
                # )

                # rdse
                active_bits = rdse.encode(buckets_idx[f])
                sdr_list.extend(active_bits)

                #  extend the sdr list with the randomly dense selected active bits
                # for bit in active_bits:
                #   sdr_list[bit] = 1

        return sdr_list
