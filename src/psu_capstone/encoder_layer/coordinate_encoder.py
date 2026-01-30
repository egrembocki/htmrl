from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, override

import numpy as np

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.sdr_layer.sdr import SDR


@dataclass
class CoordinateParameters:

    # size for each axis encoder which are x and y
    size_per_axis: int = 2048

    # active bits per axis encoder
    w: int = 40

    # units per bucket
    resolution: float = 10.0

    # optional multi-scale
    resolutions: Optional[list[float]] = None


class CoordinateEncoder(BaseEncoder[tuple[float, float]]):

    def __init__(self, parameters: CoordinateParameters, dimensions: list[int] | None = None):
        self._parameters = copy.deepcopy(parameters)
        self._parameters = self.check_parameters(self._parameters)

        self._size_per_axis = self._parameters.size_per_axis
        self._w = self._parameters.w

        # choose scales
        if self._parameters.resolutions is not None:
            self._resolutions = list(self._parameters.resolutions)
        else:
            self._resolutions = [self._parameters.resolution]

        # build RDSE pairs per scale
        self._x_encoders: list[RandomDistributedScalarEncoder] = []
        self._y_encoders: list[RandomDistributedScalarEncoder] = []

        for i, res in enumerate(self._resolutions):
            x_params = RDSEParameters(
                size=self._size_per_axis,
                active_bits=self._w,
                sparsity=0.0,
                radius=0.0,
                resolution=float(res),
                category=False,
                seed=0,
            )
            y_params = RDSEParameters(
                size=self._size_per_axis,
                active_bits=self._w,
                sparsity=0.0,
                radius=0.0,
                resolution=float(res),
                category=False,
                seed=0,
            )

            self._x_encoders.append(
                RandomDistributedScalarEncoder(x_params, dimensions=[x_params.size])
            )
            self._y_encoders.append(
                RandomDistributedScalarEncoder(y_params, dimensions=[y_params.size])
            )

        self._size = 2 * len(self._resolutions) * self._size_per_axis
        super().__init__(dimensions, self._size)

    @override
    def encode(self, input_value: tuple[float, float], output_sdr: SDR) -> None:
        assert output_sdr.size == self._size, "Output SDR size does not match encoder size."

        a, b = input_value

        if math.isnan(a) or math.isnan(b):
            output_sdr.zero()
            return

        x, y = float(a), float(b)

        dense_out: list[int] = []

        tmp_x = SDR([self._size_per_axis])
        tmp_y = SDR([self._size_per_axis])

        for xe, ye in zip(self._x_encoders, self._y_encoders):
            # x block
            tmp_x.zero()
            xe.encode(x, tmp_x)
            dense_out.extend(int(v) for v in tmp_x.get_dense())

            # y block
            tmp_y.zero()
            ye.encode(y, tmp_y)
            dense_out.extend(int(v) for v in tmp_y.get_dense())

        output_sdr.set_dense(dense_out)

    def check_parameters(self, p: CoordinateParameters) -> CoordinateParameters:
        if p.size_per_axis <= 0:
            raise ValueError("size_per_axis must be > 0")
        if p.w <= 0 or p.w > p.size_per_axis:
            raise ValueError("w must be > 0 and <= size_per_axis")
        if p.resolutions is not None:
            if len(p.resolutions) == 0:
                raise ValueError("resolutions cannot be empty if provided")
            if any(r <= 0 for r in p.resolutions):
                raise ValueError("all resolutions must be > 0")
        else:
            if p.resolution <= 0:
                raise ValueError("resolution must be > 0")
        return p


if __name__ == "__main__":

    params = CoordinateParameters(size_per_axis=256, w=20, resolution=10.0)

    enc = CoordinateEncoder(params)

    out1 = SDR([1, enc.size])
    out2 = SDR([1, enc.size])

    enc.encode((100.0, 200.0), out1)
    enc.encode((100.0, 200.0), out2)

    print(out1)
    assert out1.get_dense() == out2.get_dense()
    print("Determinism OK:", out1.get_sum(), "active bits")
