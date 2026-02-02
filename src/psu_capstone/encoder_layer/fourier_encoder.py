"""
FFT encoder implementation for HTM core

Indefinite integral proof:  g_f = integral(g_t * exp(-2*pi*i*f*t))

Sum definition: X_k = sum(x_n * exp(-2*pi*i * k * n / N))


https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html

"""

from __future__ import annotations

import copy
import random
import struct
from dataclasses import dataclass, field
from typing import Any, cast, override

import mmh3
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

    active_bits_in_ranges: list[int] = field(default_factory=lambda: [5])
    """The number of active bits per frequency range in the encoder output SDR."""

    resolutions_in_ranges: list[float] = field(default_factory=lambda: [1.0])
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
        self._magnitude_peaks: list[tuple[int, int]] = field(default_factory=lambda: [])
        """List of magnitudes corresponding to each frequency range or window."""

        # time domain params
        self._start_time = self._params.start_time  # default to 0.0
        """Start time of the time interval."""
        self._stop_time = self._params.stop_time  # default to 1.0
        """Stop time of the time interval."""
        self._period_size = self._stop_time - self._start_time
        """Total time period size in seconds."""
        self._total_samples = self._params.total_samples  # default to 256
        """Total number of samples in the period."""
        self._time_step = self._period_size / self._total_samples
        """Time step between samples."""
        self._sample_rate = float(1 / self._time_step)  # samples per second
        """Sample rate in Hz."""

        # sub encoders
        self._max_peak_encoder: RandomDistributedScalarEncoder | None = None
        """RDSE encoder for peak frequency."""
        self._magnitude_encoder: RandomDistributedScalarEncoder | None = None
        """RDSE encoder for peak magnitude."""
        self._freqs

        self._calc_bucket_sizes()

    # table this for now, use scipy fft
    def _transform(self, time_data: np.ndarray) -> np.ndarray:
        """A recursive implementation of the 1D Cooley-Tukey FFT, the input should have a length of power of 2.
            Depreciated: Use scipy.fft.fft instead.

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

    def _trim(self, input_data: np.ndarray | pd.DataFrame | list[float]) -> np.ndarray:
        """Trim input array into a power of 2 size"""

        self._total_samples = len(input_data)  # MIGHT NOT BE TRUE IN ALL CASES

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

    def _calc_bucket_sizes(self) -> None:
        """Calculate the bucket sizes for each frequency range."""

        for i in range(len(self._frequency_ranges)):

            freq_range = self._frequency_ranges[i]
            start_freq = freq_range[0]
            stop_freq = freq_range[1]

            bucket_size = stop_freq - start_freq

            self._bucket_sizes.append(bucket_size)

    def _p_stable_lsh(
        self, interval: int, resolution: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a p-stable LSH encoder for frequency buckets.
        Args:
            interval (int):size of set
            resolution (float): Resolution of the encoder.
            seed (int): Random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)

        # build random projection vector
        p = rng.normal(0, 1, interval)
        b = rng.uniform(0, resolution, interval)

        return p, b

    @override
    def encode(self, input_value: Any) -> list[int]:
        """Transform the input signal via FFT and populate the provided SDR.

        Encodes a single frequency peak into a SDR as list[int].
        Logic: h(x) = floor((a*x + b) / r)

        Args:
            input_value: Any: The input time-domain signal to be encoded.

        Returns:
            list[int]: List of active bit indices in the encoded SDR.

        Raises: # TODO: change these into different raised exceptions:
            AssertionError: If input signal length is less than total samples or exceeds encoder size.

        """

        input_value = self._trim(cast(np.ndarray, input_value))
        samples = self._total_samples
        sample_rate = self._sample_rate
        time_step = self._time_step
        size = self._size
        freq_ranges = self._frequency_ranges

        self._max_peak_encoder = RandomDistributedScalarEncoder(
            RDSEParameters(
                size=size,
                active_bits=self._total_active_bits,
                radius=0.1,
                seed=self._seed,
            )
        )

        self._magnitude_encoder = RandomDistributedScalarEncoder(
            RDSEParameters(
                size=size,
                active_bits=self._total_active_bits,
                radius=0.1,
                seed=self._seed,
            )
        )

        assert len(input_value) >= samples, "Input signal length is less than total samples."
        assert len(input_value) <= self._size, "Input signal length exceeds encoder size."
        assert samples <= sample_rate, "Total samples exceed sample rate."
        assert time_step > 0, "Time step must be positive."

        start_freq = 0
        stop_freq = 0

        active_bits: list[int] = [0] * size

        freq_data = cast(np.ndarray, fft(input_value))
        freq_data = self._normalize(freq_data)

        # Nyquist frequency limit
        freq_data = freq_data[: samples // 2]
        freq_buckets = fftfreq(samples, time_step)[: samples // 2]

        # find peak frequency in entire FFT data
        peak_index = np.argmax(np.abs(freq_data))
        peak_freq = freq_buckets[peak_index]
        print(f"FFT Peak Frequency: {peak_freq} Hz")

        # save magnitude at peak frequency
        magnitude = np.abs(freq_data[int(peak_freq)])

        # TODO: false freq detection check when ranges exceed nyquist limit and peak bucket is invalid

        # TODO: handle multiple frequency ranges and magnitudes :: Each range gets its own frequency and magnitude encoding :: Each range will have active bits set by the user which is current active bits below. This will give N bits per frequency peak found in each range. The user can adjust size of encoder and total active bits to accommodate for sparisity needs.

        # TODO: find how many frequency peak magnitudes exceed thresholds (.1)

        if peak_freq <= time_step:
            peak_freq = 0
            magnitude = 0

            active_bits = [0] * size

            return active_bits
        else:

            sdr_freq = self._max_peak_encoder.encode(peak_freq)
            sdr_magnitude = self._magnitude_encoder.encode(magnitude)

            sdr_inter: set[int] = set(sdr_freq).intersection(set(sdr_magnitude))

            sdr_inter_list = list(sdr_inter)

            # for bit in sdr_inter_list:
            #    if sdr_inter_list[bit] == 1:
            #        active_bits[bit] = 1

        # loop through each frequency range in list
        for freq_range in freq_ranges:
            start_freq = freq_range[0]
            stop_freq = freq_range[1]

            if stop_freq > samples // 2:
                logger.info(
                    "Frequency range exceeds Nyquist frequency, adjusting to max allowable."
                )
                stop_freq = samples // 2

            assert stop_freq <= len(freq_data), "Frequency range exceeds FFT output size."
            assert start_freq >= 0, "Start frequency must be non-negative."

            assert (
                start_freq <= peak_freq <= stop_freq
            ), "Peak frequency not in any specified frequency range."

            assert (
                peak_freq <= samples // 2
            ), "Peak frequency exceeds Nyquist frequency (half the sample rate)."

            # slice freq data to current frequency range
            freq_data = freq_data[start_freq : stop_freq + 1]
            freq_data = np.real(np.abs(freq_data))

            current_active_bits = self._active_bits_in_ranges[
                self._frequency_ranges.index(freq_range)
            ]
            current_res = self._resolutions_in_ranges[self._frequency_ranges.index(freq_range)]
            current_interval_size = self._bucket_sizes[self._frequency_ranges.index(freq_range)]

            assert len(freq_data) >= current_active_bits, "Active bits exceed frequency range size."

            projections, offsets = self._p_stable_lsh(
                interval=current_interval_size,
                resolution=current_res,
                # create seed based on the seed before it and the index of the frequency range
                seed=self._seed,
            )

            # encode the frequency range slice
            #  Logic: h(x) = floor((a*x + b) / r)
            for f in range(current_interval_size):

                # compute p-stable LSH
                a = projections[f]
                b = offsets[f]

                # hash value
                hash_value = (a * freq_data[f] + b) / current_res
                bucket_idx = int(np.floor(hash_value))

                # map to bucket index to bit position in active bits
                b_bytes = struct.pack("i", bucket_idx)
                bit_idx = int(mmh3.hash(b_bytes, self._seed)) % size

                active_bits[bit_idx] = 1

        return active_bits

    def _check_params(self, parameters: FourierEncoderParameters) -> FourierEncoderParameters:
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
