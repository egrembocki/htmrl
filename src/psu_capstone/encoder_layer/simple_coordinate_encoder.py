from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, cast, override

from psu_capstone.encoder_layer.base_encoder import BaseEncoder, ParentDataclass
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


class CoordinateEncoder2(BaseEncoder[tuple[float, ...]]):
    def __init__(self, parameters: "CoordinateParameters2"):
        self._p = copy.deepcopy(parameters)

        self._dims = self._p.dims
        self._n = self._p.n
        self._w = self._p.w

        # Per-dimension resolution support
        if self._p.resolutions is not None:
            if len(self._p.resolutions) != self._dims:
                raise ValueError("resolutions must match dims")
            resolutions = self._p.resolutions
        else:
            resolutions = (self._p.resolution,) * self._dims

        self._size = self._n * self._dims
        self._encoders: list[RandomDistributedScalarEncoder] = []

        base_seed = self._p.seed

        for d in range(self._dims):
            enc_params = RDSEParameters(
                size=self._n,
                active_bits=self._w,
                sparsity=0.0,
                radius=0.0,
                resolution=float(resolutions[d]),
                category=False,
                seed=(base_seed + d + 1),  # deterministic per dimension
            )
            self._encoders.append(RandomDistributedScalarEncoder(enc_params))

        super().__init__(self._size)

    @override
    def encode(self, input_value: tuple[float, ...]) -> list[int]:
        if len(input_value) != self._dims:
            raise ValueError(f"Expected {self._dims} dims, got {len(input_value)}")

        out: list[int] = []
        for d, v in enumerate(input_value):
            out.extend(self._encoders[d].encode(float(v)))
        return out

    def _split(self, encoded: list[int]) -> list[list[int]]:
        if len(encoded) != self._size:
            raise ValueError(f"Encoded length {len(encoded)} != expected {self._size}")

        chunks = []
        for d in range(self._dims):
            start = d * self._n
            end = (d + 1) * self._n
            chunks.append(encoded[start:end])
        return chunks

    def register_candidates(self, candidates: Iterable[tuple[float, ...]]) -> None:
        for tup in candidates:
            if len(tup) != self._dims:
                raise ValueError("Candidate tuple has wrong dimensionality")
            for d, v in enumerate(tup):
                self._encoders[d].register_encoding(float(v))

    def decode(
        self,
        encoded: list[int],
        candidates: Iterable[tuple[float, ...]] | None = None,
    ) -> tuple[tuple[float, ...], float]:
        chunks = self._split(encoded)

        # If candidates were provided, derive per-dim candidate lists
        per_dim_candidates: list[list[float]] | None = None
        if candidates is not None:
            cand_list = list(candidates)
            for t in cand_list:
                if len(t) != self._dims:
                    raise ValueError("Candidate tuple has wrong dimensionality")

            per_dim_candidates = [[float(t[d]) for t in cand_list] for d in range(self._dims)]

        decoded_vals: list[float] = []
        confidences: list[float] = []

        for d in range(self._dims):
            best, conf = self._encoders[d].decode(
                chunks[d],
                candidates=(per_dim_candidates[d] if per_dim_candidates is not None else None),
            )
            if best is None:
                raise ValueError("RDSE decode returned None (no candidates/registered encodings?)")
            decoded_vals.append(float(best))
            confidences.append(float(conf))

        combined_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return (tuple(decoded_vals), combined_conf)


@dataclass
class CoordinateParameters2(ParentDataclass):
    n: int = 2048
    w: int = 25
    dims: int = 2

    seed: int = 0
    resolution: float = 1.0
    resolutions: tuple[float, ...] | None = None

    encoder_class = CoordinateEncoder2


if __name__ == "__main__":
    params = CoordinateParameters2(n=400, w=25)
    enc = CoordinateEncoder2(params)

    params1 = CoordinateParameters2(n=2048, w=20)
    enc2 = CoordinateEncoder2(params1)

    a = enc2.encode((10, 20))
    b = enc.encode((11, 20))
    c = enc.encode((0, 0))

    print(enc2.decode(a))
    print(enc.decode(c))

    print()

    test_keys = [
        ((0, 0)),
        ((10, 20)),
        ((-5, 7)),
        ((100, 200)),
    ]

    for key in test_keys:
        encoded = enc.encode(key)
        print(enc.decode(encoded))

    params2 = CoordinateParameters2(n=400, w=20, dims=3)

    enc3 = CoordinateEncoder2(params2)

    d = enc3.encode((10, 20, 30))
    e = enc3.encode((10, 20, 30))
