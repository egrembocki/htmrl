"""Encoder to encode the difference between two values"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import override

from psu_capstone.encoder_layer.base_encoder import BaseEncoder, ParentDataClass
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
        self._logger = get_logger("DeltaEncoder")
        self._size = params.size
        self._sparsity = params.sparsity
        self._active_bits = params.active_bits
        self._cached_encodings: dict[float, list[int]] = {}
        self._delta_value = 0.0

        self._rdse_encoder = RandomDistributedScalarEncoder(
            RDSEParameters(size=params.size, sparsity=params.sparsity)
        )

        self._coordinate_encoder = CoordinateEncoder(
            CoordinateParameters(n=2048, w=42, seed=42, max_radius=10, use_all_neighbors=True)
        )

        super().__init__(params.size)

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
        # This is a placeholder implementation. You can implement this method to encode multiple pairs if needed.
        # TODO use coordinate encoder to encode pairs of values as coordinates and radius, then return the the difference between the two coordinates as the delta value to encode with the RDSE encoder. This will allow us to encode pairs of values in a way that captures their relationship, rather than just encoding the delta value directly.

        if len(pairs) != 2:
            raise ValueError("Expected exactly 2 pairs of values to encode.")

        raise NotImplementedError("Encoding pairs is not implemented yet.")


@dataclass
class DeltaEncoderParameters(ParentDataClass):
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

    test_encoder = RandomDistributedScalarEncoder(RDSEParameters())

    e1 = test_encoder.encode(10.0)
    e2 = test_encoder.encode(5.0)

    encoder = DeltaEncoder()
    input_value = (10.0, 5.0)
    encoding = encoder.encode(input_value)
    print(f"Input: {input_value}, Encoding: {encoding}")

    assert encoding == e2, "Encoding does not match expected value for 10.0"
