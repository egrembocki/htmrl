from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Iterable, override

import mmh3
import numpy as np

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


class CoordinateEncoder(BaseEncoder[tuple[float, float]]):

    def __init__(self, parameters: CoordinateParameters, dimensions: list[int] | None = None):
        self._parameters = copy.deepcopy(parameters)

        self._n = self._parameters.n
        self._w = self._parameters.w
        self._seed = self._parameters.seed
        self._dims = self._parameters.dims

        enc_params = RDSEParameters(
            size=self._n,
            active_bits=self._w,
            sparsity=0.0,
            radius=0.0,
            resolution=1.0 / self._n,
            category=False,
            seed=self._seed,
        )

        self._encoder = RandomDistributedScalarEncoder(enc_params)

        self._overlap = self._encoder._overlap
        self._encoding_cache: dict[tuple[tuple[int, ...], int], list[int]] = {}

        max_neighbors = (2 * self._parameters.max_radius + 1) ** self._dims

        # for all neighbors use this
        # self._size = max_neighbors * self._n

        # for winners use this
        self._size = self._w * self._n

        super().__init__(dimensions, self._size)

    @override
    def encode(self, input_value: tuple[tuple[int, ...] | list[int], int]) -> list[int]:
        return self.register_encoding(input_value)

    def _compute_encoding(self, key: tuple[tuple[int, ...], int]) -> list[int]:
        coordinate, radius = key
        neighbors = self._neighbors(coordinate, radius)
        winners = self._topWCoordinates(neighbors, self._w)

        out: list[int] = []
        for c in winners:
            v = self._coord_to_unit_float(c)
            out.extend(int(b) for b in self._encoder.encode(v))

        expected = self._size
        if len(out) < expected:
            out.extend([0] * (expected - len(out)))
        elif len(out) > expected:
            out = out[:expected]

        return out

    def register_encoding(
        self, input_value: tuple[tuple[int, ...] | list[int], int], encoded: list[int] | None = None
    ) -> list[int]:
        coordinate, radius = input_value
        key = (tuple(int(v) for v in coordinate), int(radius))

        vector = encoded if encoded is not None else self._compute_encoding(key)

        if len(vector) != self._size:
            raise ValueError("Stored encoding must match encoder size")

        self._encoding_cache[key] = vector
        return vector

    @staticmethod
    def _neighbors(coordinate, radius):
        ranges = (range(int(n) - radius, int(n) + radius + 1) for n in coordinate)
        return list(itertools.product(*ranges))

    @classmethod
    def _topWCoordinates(cls, coordinates, w):
        scored = [(cls._orderForCoordinate(c), c) for c in coordinates]

        scored.sort(key=lambda x: x[0])
        return [c for _, c in scored[-w:]]

    @staticmethod
    def _orderForCoordinate(coordinate) -> float:
        s = ",".join(str(int(v)) for v in coordinate)
        h = mmh3.hash(s, signed=False)
        return h / 2**32

    def _coord_to_unit_float(self, coordinate) -> float:
        s = ",".join(str(int(v)) for v in coordinate)
        h = mmh3.hash(s, seed=self._seed, signed=False)
        return h / 2**32

    def decode(
        self,
        encoded: list[int],
        candidates: Iterable[tuple[tuple[int, ...], int]] | None = None,
    ) -> tuple[tuple[tuple[int, ...], int] | None, float]:
        if len(encoded) != self.size:
            raise ValueError(
                f"Encoded input size ({len(encoded)}) does not match encoder size ({self.size})"
            )

        search_keys = (
            list(candidates) if candidates is not None else list(self._encoding_cache.keys())
        )
        if not search_keys:
            raise ValueError("No candidate encodings are available for decoding")

        best_key = None
        best_overlap = -1

        for key in search_keys:
            candidate_encoding = self._encoding_cache.get(key)
            if candidate_encoding is None:
                candidate_encoding = self.register_encoding(key)

            overlap = self._overlap(encoded, candidate_encoding)
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key

        expected_ones = (len(encoded) // self._n) * self._w
        confidence = best_overlap / expected_ones if expected_ones > 0 else 0.0
        return best_key, confidence


@dataclass
class CoordinateParameters:
    n: int = 2048
    w: int = 25
    seed: int = 42
    max_radius: int = 2
    dims: int = 2
    encoder_class = CoordinateEncoder


if __name__ == "__main__":
    params = CoordinateParameters(n=400, w=25, max_radius=6)
    enc = CoordinateEncoder(params)

    params1 = CoordinateParameters(n=2048, w=20, max_radius=2)
    enc2 = CoordinateEncoder(params1)

    a = enc2.encode(((10, 20), 2))
    b = enc.encode(((11, 20), 2))
    c = enc.encode(((0, 0), 5))

    print(enc2.decode(a))
    print(enc.decode(c))

    print()

    test_keys = [
        ((0, 0), 6),
        ((10, 20), 6),
        ((-5, 7), 6),
        ((100, 200), 6),
    ]

    for key in test_keys:
        encoded = enc.encode(key)
        print(enc.decode(encoded))
