"""
FFT encoder implementation for HTM core

Indefinite integral proof:  g_f = integral(g_t * exp(-2*pi*i*f*t))

Sum definition: X_k = sum(x_n * exp(-2*pi*i * k * n / N))


https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.03-Fast-Fourier-Transform.html

"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, cast, override

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, ifft
from sklearn.utils import deprecated

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.log import logger


class FourierEncoder(BaseEncoder[np.ndarray], list[int]):
    """Encoder that uses Fourier Transform on time data. Build RDSE for each frequency component. Assume that time domain is in seconds.

    Args:
        parameters (FourierEncoderParameters, optional): Fourier encoder parameters. Defaults to FourierEncoderParameters().
        dimensions (list[int], optional): List of dimensions for the encoder. Defaults to [].
    """

    def __init__(self, parameters: FourierEncoderParameters | None = None):
        """Initialize the encoder with optional Fourier parameters and encoder dimensions."""

        if parameters is None:
            parameters = FourierEncoderParameters()

        # set the size of the base encoder
        super().__init__(parameters.size)

        self._params = copy.deepcopy(parameters)
        """Fourier encoder local copy of passed parameters."""

        # encoder params
        self._frequency_ranges = self._params.frequency_ranges
        """List of frequency ranges to find."""
        self._size = self._params.size  # default to 2048
        """Size of the encoder."""
        self._total_active_bits = self._params.total_active_bits  # default to 40
        """Total number of active bits in the encoder output SDR."""
        self._total_sparsity = self._params.total_sparsity  # default to 0.02
        """Total sparsity of the encoder."""
        self._total_resolution = self._params.total_resolution  # default to 1.0
        """Total resolution of the encoder."""
        self._active_bits_in_ranges = self._params.active_bits_in_ranges
        """The number of active bits per frequency range in the encoder output SDR."""
        self._sparsity_in_ranges = self._params.sparsity_in_ranges
        """The sparsity per frequency range in the encoder output SDR."""
        self._resolutions_in_ranges = self._params.resolutions_in_ranges
        """The resolution per frequency range in the encoder output SDR."""
        self._seed = self._params.seed  # default to 32
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
        self._time_step = self._period_size / self._total_samples
        """Time step between samples."""
        self._sample_rate = float(1 / self._time_step)  # samples per second
        """Sample rate in Hz."""

        self._validate_params(self._params)
        self._calc_bucket_sizes()

    @deprecated("Use scipy.fft.fft instead.")
    def __transform(self, time_data: np.ndarray) -> np.ndarray:
        """A recursive implementation of the 1D Cooley-Tukey FFT, the input should have a length of power of 2.
            Depreciated: Use scipy.fft.fft instead.

            Sum definition: X_k = sum(x_n * exp(-2*pi*i * k * n / N))

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
            t_even = self.__transform(time_data[::2])
            t_odd = self.__transform(time_data[1::2])
            omega = np.exp(-2j * np.pi * np.arange(total_samples) / total_samples)

            half = int(total_samples / 2)

            freq_data = np.concatenate(
                [
                    t_even + omega[np.arange(half)] * t_odd,
                    t_even + omega[np.arange(half, total_samples)] * t_odd,
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

        # verify sizes of input data
        self._total_samples = len(input_data)  # MIGHT NOT BE TRUE IN ALL CASES
        self._time_step = self._period_size / self._total_samples
        self._sample_rate = float(1 / self._time_step)  # samples per second

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

    @deprecated("Using rdse internal hasher.")
    def __p_stable_lsh(
        self, interval: int, resolution: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a p-stable LSH encoder for frequency buckets.

            Logic: h(x) = floor((a*x + b) / r)

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

    def _find_num_peaks(self, freq_data: np.ndarray) -> int:
        """Find the number of peaks in the given frequency data."""

        num_peaks = 0

        for i in range(len(freq_data)):
            if freq_data[i] > 0.1:
                num_peaks += 1
        return num_peaks

    def _set_equal_sparsity_in_ranges(self) -> None:
        """Set the sparsity in each frequency range based on total sparsity."""

        total_sparsity = self._total_sparsity
        num_ranges = len(self._frequency_ranges)

        if len(self._sparsity_in_ranges) != num_ranges:
            equal_sparsity = total_sparsity / num_ranges
            self._sparsity_in_ranges = [equal_sparsity] * num_ranges

        logger.info(f"Sparsity in ranges set to: {self._sparsity_in_ranges}")

    def _set_sparsity_at_range(self, range_index: int, sparsity: float) -> None:
        """Set the sparsity for a specific frequency range.

        Args:
            range_index (int): Index of the frequency range to set.
            sparsity (float): Sparsity value to set for the specified range.
        """

        if range_index < 0 or range_index >= len(self._frequency_ranges):
            raise IndexError("Range index out of bounds.")

        self._sparsity_in_ranges[range_index] = sparsity
        logger.info(f"Sparsity for range {self._frequency_ranges[range_index]} set to: {sparsity}")

    def _set_active_bits_equal_in_ranges(self) -> None:
        """Set the active bits in each frequency range based on total active bits."""

        total_active_bits = self._total_active_bits
        num_ranges = len(self._frequency_ranges)

        if len(self._active_bits_in_ranges) != num_ranges:
            equal_active_bits = total_active_bits // num_ranges
            self._active_bits_in_ranges = [equal_active_bits] * num_ranges

        logger.info(f"Active bits in ranges set to: {self._active_bits_in_ranges}")

    def _set_active_bits_at_range(self, range_index: int, active_bits: int) -> None:
        """Set the active bits for a specific frequency range.

        Args:
            range_index (int): Index of the frequency range to set.
            active_bits (int): Active bits value to set for the specified range.
        """

        if range_index < 0 or range_index >= len(self._frequency_ranges):
            raise IndexError("Range index out of bounds.")

        self._active_bits_in_ranges[range_index] = active_bits
        logger.info(
            f"Active bits for range {self._frequency_ranges[range_index]} set to: {active_bits}"
        )

    @override
    def encode(self, input_value: Any) -> list[int]:
        """Transform the input signal via FFT and populate the provided SDR.

        Encodes a single frequency peak into a SDR as list[int].

        Args:
            input_value: Any: The input time-domain signal to be encoded.

        Returns:
            list[int]: List of active bit indices in the encoded SDR.

        Raises: # TODO: change these into different raised exceptions:


        """

        # validate parameters

        self._validate_params(self._params)

        #  trim input data into a power of 2 size and reset internal params
        input_value = self._trim(cast(np.ndarray, input_value))
        samples = self._total_samples
        time_step = self._time_step
        size = self._size
        freq_ranges = self._frequency_ranges

        #  padd dense list with zeros to set dense bits by index later
        dense_bits = [0] * size

        #  list to hold values if we want to perform a mean later
        magnitudes = []
        frequencies = []

        # initial frequency range values of the current interval
        start_freq = 0
        stop_freq = 0

        #  perform FFT on input time data and normalize
        freq_data = cast(np.ndarray, fft(input_value))
        freq_data = self._normalize(freq_data)

        # Nyquist frequency limit and store freq buckets as indexes
        freq_data = freq_data[: samples // 2]

        # LOOP THROUGH FREQUENCY RANGES !!
        print("Looping through frequency ranges:")
        for freq_range in freq_ranges:
            start_freq = freq_range[0]
            stop_freq = freq_range[1]

            peak_freq = 0
            freq_buckets = fftfreq(samples, time_step)[start_freq:stop_freq]

            # slice freq data to current frequency range  :: assuming freq ranges are int in Hz
            freq_slice = freq_data[start_freq:stop_freq]
            freq_slice = np.real(np.abs(freq_slice))

            # find peak frequency in the current interval
            peak_index = np.argmax(np.abs(freq_slice))
            if freq_slice[peak_index] < 0.1:
                peak_freq = 0
            else:
                peak_freq = float(freq_buckets[peak_index])

            # if no peaks are present return empty SDR :: early exit
            if peak_freq <= 0:

                logger.info(f"No significant peaks found in range {freq_range}.")

                continue

            # current resolution for this frequency range
            current_res = self._resolutions_in_ranges[self._frequency_ranges.index(freq_range)]
            current_interval_size = self._bucket_sizes[self._frequency_ranges.index(freq_range)]
            current_sparsity = self._sparsity_in_ranges[self._frequency_ranges.index(freq_range)]

            # size = size // 2

            num_peaks = self._find_num_peaks(freq_slice)
            logger.info(f"Number of peaks found in range {freq_range}: {num_peaks}")

            # build each encoder for frequency and magnitude :: thank you GC
            freq_encoder = RandomDistributedScalarEncoder(
                RDSEParameters(
                    size=size,  # // num_peaks if num_peaks >= 1 else size
                    sparsity=current_sparsity,  # sparisity requested by user
                    active_bits=0,
                    resolution=current_res,
                )
            )

            # encode the frequency range slice
            for f in range(current_interval_size):

                # find frequencies that contributed to the fft output by magnitude threshold
                if freq_slice[f] < 0.1:
                    continue

                else:
                    logger.info(f"Encoding frequency bucket: {f + start_freq}")
                    # freq is the current bucket being evaluated
                    freq_value = float(f + start_freq)

                    sdr_freq = freq_encoder.encode(freq_value)
                    magnitudes.append(freq_slice[f])
                    frequencies.append(f + start_freq)
                    for ids in range(len(sdr_freq)):
                        if sdr_freq[ids] == 1:
                            dense_bits[ids] = 1

            # END of INNER for loop :: freq intervals

            #
            magnitude_encoder = RandomDistributedScalarEncoder(
                RDSEParameters(size=size, sparsity=time_step, active_bits=0, resolution=time_step)
            )

            magnitude = float(np.mean(magnitudes))

            print(f"Mean magnitude: {magnitude}")

            sdr_magnitude = magnitude_encoder.encode(magnitude)

            for idx in range(len(sdr_magnitude)):
                if sdr_magnitude[idx] == 1:
                    dense_bits[idx] = 1

        # END of OUTER for loop

        # freq that are closer together will have collissions in the SDR giving less active bits than expected
        # freq that are further apart will have less collissions giving more active bits than expected
        return dense_bits
        # END def encode

    def decode(
        self, encoded: list[int], candidates: Iterable[float] | None = None
    ) -> dict[str, list[tuple[tuple[int, int], float | None, float]] | tuple[float | None, float]]:
        """Decode a combined Fourier SDR into per-range frequency estimates and magnitude."""
        if len(encoded) != self.size:
            raise ValueError(
                f"Encoded input size ({len(encoded)}) does not match encoder size ({self.size})"
            )

        results: dict[
            str, list[tuple[tuple[int, int], float | None, float]] | tuple[float | None, float]
        ] = {"frequencies": []}

        for range_index, freq_range in enumerate(self._frequency_ranges):
            start_freq, stop_freq = freq_range
            current_res = self._resolutions_in_ranges[range_index]
            current_sparsity = self._sparsity_in_ranges[range_index]

            freq_encoder = RandomDistributedScalarEncoder(
                RDSEParameters(
                    size=self._size,
                    sparsity=current_sparsity,
                    active_bits=0,
                    resolution=current_res,
                )
            )

            if candidates is not None:
                range_candidates = [
                    candidate for candidate in candidates if start_freq <= candidate < stop_freq
                ]
                if not range_candidates:
                    range_candidates = list(range(start_freq, stop_freq))
            else:
                range_candidates = list(range(start_freq, stop_freq))
            decoded_value, confidence = freq_encoder.decode(encoded, range_candidates)
            results["frequencies"].append((freq_range, decoded_value, confidence))

        magnitude_candidates: list[float] = []
        if self._time_step > 0:
            magnitude_candidates = list(
                np.arange(0.0, 1.0 + self._time_step, self._time_step, dtype=float)
            )

        if magnitude_candidates:
            magnitude_encoder = RandomDistributedScalarEncoder(
                RDSEParameters(
                    size=self._size,
                    sparsity=self._time_step,
                    active_bits=0,
                    resolution=self._time_step,
                )
            )
            results["magnitude"] = magnitude_encoder.decode(encoded, magnitude_candidates)

        return results

    def _validate_params(self, parameters: FourierEncoderParameters) -> None:
        """Check if the provided parameters are valid for the Fourier encoder.

        Args:
            parameters (FourierEncoderParameters): The Fourier encoder parameters to check.

            Returns:
            FourierEncoderParameters: The validated Fourier encoder parameters."""

        params = copy.deepcopy(parameters)

        # xor active bits or sparsity checks
        if len(params.active_bits_in_ranges) > 0 and len(params.sparsity_in_ranges) > 0:
            raise ValueError(
                "Cannot specify both active_bits_in_ranges and sparsity_in_ranges. Choose one."
            )
        elif params.total_active_bits <= 0 and params.total_sparsity <= 0:
            raise ValueError(
                "Must specify either total_active_bits or total_sparsity greater than zero."
            )
        elif params.total_active_bits > 0 and params.total_sparsity > 0:
            raise ValueError(
                "Cannot specify both total_active_bits and total_sparsity. Choose one."
            )

        # total sparsity checks TODO: More edge cases to cover here
        if params.equal_sparsity and len(params.sparsity_in_ranges) <= 0:
            self._set_equal_sparsity_in_ranges()

        else:

            total_sparsity = sum(params.sparsity_in_ranges)

            if not np.isclose(self._total_sparsity, total_sparsity, atol=1e-3):
                raise ValueError(
                    f"Total sparsity {params.total_sparsity} does no` match sum of sparsities in ranges {total_sparsity}."
                )

        if (
            len(params.sparsity_in_ranges)
            != len(params.frequency_ranges)
            != len(params.resolutions_in_ranges)
        ):
            raise ValueError(
                "All lengths of sparsity_in_ranges, frequency_ranges, and resolutions_in_ranges must be equal."
            )

        # total active bits checks
        if len(params.active_bits_in_ranges) <= 0 and len(params.sparsity_in_ranges) <= 0:
            self._set_active_bits_equal_in_ranges()
        else:
            total_active_bits = sum(params.active_bits_in_ranges)

            if self._total_active_bits != total_active_bits:
                raise ValueError(
                    f"Total active bits {params.total_active_bits} does not match sum of active bits in ranges {total_active_bits}."
                )

        # frequency range checks
        for freq_range in params.frequency_ranges:
            if freq_range[0] < 0 or freq_range[1] < 0:
                raise ValueError("Frequency ranges must be non-negative.")
            if freq_range[0] >= freq_range[1]:
                raise ValueError("Frequency range start must be less than end.")

            if freq_range[1] > (self._sample_rate // 2):
                raise ValueError(
                    f"Frequency range end {freq_range[1]} exceeds Nyquist frequency {self._sample_rate / 2} Hz."
                )

        # resolution checks
        total_resolution = sum(params.resolutions_in_ranges)

        for resolution in params.resolutions_in_ranges:
            if resolution <= 0:
                raise ValueError("Resolutions must be positive.")

        if not np.isclose(self._total_resolution, total_resolution, atol=1e-3):
            raise ValueError(
                f"Total resolution {params.total_resolution} does not match sum of resolutions in ranges {total_resolution}."
            )


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

    # reference to Encoder
    encoder_class = FourierEncoder
    """Reference to the FourierEncoder class."""

    # encoder params
    equal_sparsity: bool = True
    """Flag to indicate if sparsity should be equal across frequency ranges."""

    size: int = 4096
    """The size the encoder"""

    total_active_bits: int = 0
    """Total number of active bits in the encoder output SDR."""

    total_sparsity: float = 0.02
    """Total sparsity of the encoder."""

    total_resolution: float = 1.0
    """Total resolution of the encoder."""

    frequency_ranges: list[tuple[int, int]] = field(default_factory=lambda: [(1, 100)])
    """List of frequency ranges to find."""

    active_bits_in_ranges: list[int] = field(default_factory=lambda: [])
    """The number of active bits per frequency range in the encoder output SDR."""

    sparsity_in_ranges: list[float] = field(default_factory=lambda: [0.02])
    """The sparsity per frequency range in the encoder output SDR."""

    resolutions_in_ranges: list[float] = field(default_factory=lambda: [1.0])
    """The resolution per frequency range in the encoder output SDR."""

    seed: int = 32
    """Random seed for reproducibility."""

    #  time domain params
    start_time: float = 0.0
    """Start time of the time interval."""

    stop_time: float = 1.0
    """Stop time of the time interval."""

    total_samples: int = 2048
    """Total number of samples in the period."""
