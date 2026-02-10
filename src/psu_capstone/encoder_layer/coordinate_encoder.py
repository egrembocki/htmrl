from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import override

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
            radius=1.0,
            resolution=0.0,
            category=False,
            seed=self._seed,
        )

        self._encoder = RandomDistributedScalarEncoder(enc_params)

        max_neighbors = (2 * self._parameters.max_radius + 1) ** self._dims

        # for all neighbors use this
        self._size = max_neighbors * self._n

        # for winners use this
        # self._size = self._w * self._n

        super().__init__(dimensions, self._size)

    @override
    def encode(self, input_value: tuple[tuple[int, ...] | list[int], int]) -> list[int]:
        coordinate, radius = input_value

        if not isinstance(radius, int):
            raise TypeError(f"Expected integer radius, got: {radius!r} ({type(radius)})")

        neighbors = self._neighbors(coordinate, radius)
        winners = self._topWCoordinates(neighbors, self._w)

        out: list[int] = []
        for c in neighbors:
            v = self._coord_to_unit_float(c)
            block = self._encoder.encode(v)
            out.extend(int(b) for b in block)

        expected = self._size
        if len(out) < expected:
            out.extend([0] * (expected - len(out)))
        elif len(out) > expected:
            out = out[:expected]

        return out

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


@dataclass
class CoordinateParameters:
    n: int = 2048
    w: int = 25
    seed: int = 42
    max_radius: int = 2
    dims: int = 2
    encoder_class = CoordinateEncoder


if __name__ == "__main__":
    params = CoordinateParameters(n=40, w=25)
    enc = CoordinateEncoder(params)

    a = enc.encode(((10, 20), 2))
    b = enc.encode(((11, 20), 2))

    def _overlap_count(first: list[int], second: list[int]) -> int:
        return np.count_nonzero(first == second)

    print(len(a))
    print(_overlap_count(a, b))
    print(a)
