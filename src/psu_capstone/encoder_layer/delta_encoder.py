"""Encoder to encode the difference between two values"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import override

from psu_capstone.encoder_layer.base_encoder import BaseEncoder, ParentDataClass
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


class DeltaEncoder(BaseEncoder[float]):
    """Encodes the difference between two values."""

    def __init__(self, encoder_params: DeltaEncoderParameters | None = None):

        params = (
            copy.deepcopy(encoder_params)
            if encoder_params is not None
            else DeltaEncoderParameters()
        )

        self._size = params.size
        self._sparsity = params.sparsity
        self._active_bits = params.active_bits
        self._cached_encodings = {}

        super().__init__(params.size)

    @override
    def encode(self, input_value: float) -> list[int]:
        """Encode the difference between value1 and value2."""
        raise NotImplementedError("DeltaEncoder does not implement encode()")

    def decode(self, sdr: list[int]) -> tuple[float, float]:
        """Decode the SDR back to a value and confidence."""
        raise NotImplementedError("DeltaEncoder does not implement decode()")


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
