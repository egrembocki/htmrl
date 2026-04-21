"""Encoder to encode the difference between two values"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import override

import numpy as np

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.coordinate_encoder import CoordinateEncoder, CoordinateParameters
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.log import get_logger


class DeltaEncoder(BaseEncoder[tuple[float, float] | list[tuple[float, float]]]):
    """Encodes the difference between two values."""

    def __init__(self, encoder_params: DeltaEncoderParameters | None = None):

        params = (
            copy.deepcopy(encoder_params)
            if encoder_params is not None
            else DeltaEncoderParameters()
        )
        params = self._normalize_parameters(params)
        self._logger = get_logger("DeltaEncoder")
        self._size = params.size
        self._sparsity = params.sparsity
        self._active_bits = params.active_bits
        self._cached_encodings: dict[float, list[int]] = {}
        self._delta_value = 0.0

        self._rdse_encoder = RandomDistributedScalarEncoder(
            RDSEParameters(size=params.size, active_bits=params.active_bits, sparsity=0.0)
        )

        self._coordinate_encoder = CoordinateEncoder(
            CoordinateParameters(n=2048, w=42, seed=42, max_radius=10, use_all_neighbors=True)
        )

        super().__init__(params.size)

    def _normalize_parameters(self, params: DeltaEncoderParameters) -> DeltaEncoderParameters:
        """Normalize active-bit and sparsity inputs so both attributes stay meaningful."""

        if params.size <= 0:
            raise ValueError("Parameter 'size' must be positive.")

        has_active_bits = params.active_bits > 0
        has_sparsity = params.sparsity > 0.0

        if has_active_bits and has_sparsity:
            raise ValueError("Choose only one of 'active_bits' or 'sparsity'.")

        if not has_active_bits and not has_sparsity:
            raise ValueError("You must provide one of 'active_bits' or 'sparsity'.")

        if has_sparsity:
            if not 0.0 < params.sparsity <= 1.0:
                raise ValueError("sparsity must be between 0 and 1.")
            params.active_bits = int(round(params.size * params.sparsity))
            if params.active_bits <= 0:
                raise ValueError("Computed active bits must be greater than 0.")
        else:
            if params.active_bits > params.size:
                raise ValueError("active_bits cannot be greater than size.")
            params.sparsity = params.active_bits / float(params.size)

        return params

    @override
    def encode(self, input_value: tuple[float, float] | list[tuple[float, float]]) -> list[int]:
        """Encode the difference between value1 and value2."""

        if isinstance(input_value, list):
            return self._encode_pairs(input_value)

        value1, value2 = input_value
        if value1 > value2:
            self._delta_value = value1 - value2
        else:
            self._delta_value = value2 - value1

        # return early if we have already computed the encoding for this delta value
        if self._delta_value in self._cached_encodings:
            return self._cached_encodings[self._delta_value]

        encoding = self._rdse_encoder.encode(self._delta_value)
        self._logger.info(
            f"Encoded delta value {self._delta_value} to SDR with {sum(encoding)} active bits."
        )
        self._cached_encodings[self._delta_value] = encoding

        return encoding

    def decode(self, sdr: list[int]) -> tuple[float, float]:
        """Decode the SDR back to a value and confidence."""
        raise NotImplementedError("DeltaEncoder does not implement decode()")

    def _encode_pairs(self, pairs: list[tuple[float, float]]) -> list[int]:
        """Encode a list of value pairs into a single SDR."""

        if len(pairs) != 2:
            raise ValueError("Expected exactly 2 pairs of values to encode.")

        pair_one = pairs[0]
        pair_two = pairs[1]

        pair_one = (int(pair_one[0]), int(pair_one[1]))
        pair_two = (int(pair_two[0]), int(pair_two[1]))

        t_delta = (abs(pair_one[0] - pair_two[0]), abs(pair_one[1] - pair_two[1]))
        t_delta_distance: float = (t_delta[0] ** 2 + t_delta[1] ** 2) ** 0.5

        encoding_one = self._coordinate_encoder.encode((pair_one, 5))
        encoding_two = self._coordinate_encoder.encode((pair_two, 5))

        delta_encoding = self._rdse_encoder.encode((t_delta_distance))

        e_or: list[int] = np.logical_and(encoding_one, encoding_two).astype(int).tolist()
        if len(e_or) != len(delta_encoding):
            # CoordinateEncoder can produce concatenated block SDRs. Collapse blocks
            # back to the RDSE output width so representations are compatible.
            block_count, remainder = divmod(len(e_or), len(delta_encoding))
            if remainder != 0:
                raise ValueError(
                    "Coordinate encoding length must be divisible by RDSE encoding length."
                )

            reduced = np.array(e_or, dtype=int).reshape(block_count, len(delta_encoding))
            e_or = np.any(reduced, axis=0).astype(int).tolist()

        delta_encoding: list[int] = np.logical_or(e_or, delta_encoding).astype(int).tolist()

        return delta_encoding


@dataclass
class DeltaEncoderParameters:
    """Parameters for the DeltaEncoder."""

    encoder_class = DeltaEncoder
    """The class of the encoder to be used. Should be a subclass of BaseEncoder."""

    size: int = 2048
    """Number of bits in the vector output SDR."""

    sparsity: float = 0.02
    """The fraction of bits that are active in the output SDR."""

    active_bits: int = 0
    """The number of bits that are active in the output SDR."""

    encode_pairs: bool = True
    """Whether to encode pairs of values (value1, value2) or just the delta value."""


if __name__ == "__main__":

    encoder = DeltaEncoder()

    input_value = [(10.0, 5.0), (1.0, 3.5)]
    encoding = encoder.encode(input_value)

    print(f"Input: {input_value}, Encoding: {encoding}")
