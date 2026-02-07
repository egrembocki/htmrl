from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, override

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.coordinate_encoder import CoordinateEncoder, CoordinateParameters
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters

# Web Mercator practical latitude limit (degrees)
_MAX_MERCATOR_LAT = 85.05112878


@dataclass
class GeospatialParameters:
    """
    Mercator (lat, lon) -> (x, y) meters, plus altitude meters -> z,
    then encode as [XY SDR | Z SDR].

    - XY uses CoordinateEncoder (two RDSEs per scale)
    - Z uses a dedicated RDSE (one per scale)
    """

    # XY encoder config
    size_per_axis: int = 2048
    w_xy: int = 40
    resolution_xy: float = 10.0
    resolutions_xy: Optional[list[float]] = None

    # Z encoder config
    size_alt: int = 2048
    w_alt: int = 40
    resolution_alt: float = 5.0
    resolutions_alt: Optional[list[float]] = None

    origin_lat: float = 0.0
    origin_lon: float = 0.0

    origin_alt: float = 0.0


class GeospatialEncoder(BaseEncoder[tuple[float, float, float]]):

    def __init__(self, parameters: GeospatialParameters, dimensions: list[int] | None = None):
        self._params = copy.deepcopy(parameters)
        self._params = self.check_parameters(self._params)

        xy_params = CoordinateParameters(
            size_per_axis=self._params.size_per_axis,
            w=self._params.w_xy,
            resolution=self._params.resolution_xy,
            resolutions=self._params.resolutions_xy,
        )
        self._xy = CoordinateEncoder(xy_params, dimensions=None)

        if self._params.resolutions_xy is not None:
            self._xy_scales = list(self._params.resolutions_xy)
        else:
            self._xy_scales = [float(self._params.resolution_xy)]

        if self._params.resolutions_alt is not None:
            self._z_scales = list(self._params.resolutions_alt)
        else:
            self._z_scales = [float(self._params.resolution_alt)]

        self._z_encoders: list[RandomDistributedScalarEncoder] = []
        for res in self._z_scales:
            z_params = RDSEParameters(
                size=self._params.size_alt,
                active_bits=self._params.w_alt,
                sparsity=0.0,
                radius=0.0,
                resolution=float(res),
                category=False,
                seed=0,
            )
            self._z_encoders.append(
                RandomDistributedScalarEncoder(z_params, dimensions=[z_params.size])
            )

        self._R = 6378137.0  # Web Mercator radius in meters
        self._lon0 = math.radians(self._params.origin_lon)
        self._y0 = self._mercator_y(math.radians(self._clamp_lat(self._params.origin_lat)))

        self._alt = float(self._params.origin_alt)

        # Total SDR size: XY size + (num_z_scales * size_alt)
        self._size = self._xy.size + (len(self._z_encoders) * self._params.size_alt)
        super().__init__(dimensions, self._size)

    @override
    def encode(self, input_value: tuple[float, float, float]) -> list[int]:
        lat, lon, alt = float(input_value[0]), float(input_value[1]), float(input_value[2])

        # Always normalize geo inputs
        lat = self._clamp_lat(lat)
        lon = self._wrap_lon(lon)

        # Project to meters
        x, y = self._project_mercator(lat, lon)

        # Center altitude
        z = alt - self._alt

        out: list[int] = []

        # XY block
        out.extend(int(v) for v in self._xy.encode((x, y)))

        # Z block
        for ze in self._z_encoders:
            out.extend(int(v) for v in ze.encode(z))

        return out

    def _project_mercator(self, lat_deg: float, lon_deg: float) -> tuple[float, float]:
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)

        x = self._R * (lon_rad - self._lon0)
        y = self._R * (self._mercator_y(lat_rad) - self._y0)
        return (x, y)

    @staticmethod
    def _mercator_y(lat_rad: float) -> float:
        # y = ln(tan(pi/4 + lat/2))
        return math.log(math.tan((math.pi / 4.0) + (lat_rad / 2.0)))

    @staticmethod
    def _clamp_lat(lat: float) -> float:
        if lat > _MAX_MERCATOR_LAT:
            return _MAX_MERCATOR_LAT
        if lat < -_MAX_MERCATOR_LAT:
            return -_MAX_MERCATOR_LAT
        return lat

    @staticmethod
    def _wrap_lon(lon: float) -> float:
        # Wrap longitude to [-180, 180)
        x = (lon + 180.0) % 360.0
        return x - 180.0

    def check_parameters(self, p: GeospatialParameters) -> GeospatialParameters:
        # XY
        if p.size_per_axis <= 0:
            raise ValueError("size_per_axis must be > 0")
        if p.w_xy <= 0 or p.w_xy > p.size_per_axis:
            raise ValueError("w_xy must be > 0 and <= size_per_axis")
        if p.resolutions_xy is not None:
            if len(p.resolutions_xy) == 0:
                raise ValueError("resolutions_xy_m cannot be empty if provided")
            if any(r <= 0 for r in p.resolutions_xy):
                raise ValueError("all resolutions_xy_m must be > 0")
        else:
            if p.resolution_xy <= 0:
                raise ValueError("resolution_xy_m must be > 0")

        # Z
        if p.size_alt <= 0:
            raise ValueError("size_alt must be > 0")
        if p.w_alt <= 0 or p.w_alt > p.size_alt:
            raise ValueError("w_alt must be > 0 and <= size_alt")
        if p.resolutions_alt is not None:
            if len(p.resolutions_alt) == 0:
                raise ValueError("resolutions_alt cannot be empty if provided")
            if any(r <= 0 for r in p.resolutions_alt):
                raise ValueError("all resolutions_alt_m must be > 0")
        else:
            if p.resolution_alt <= 0:
                raise ValueError("resolution_alt_m must be > 0")

        # Origins
        if not (-90.0 <= p.origin_lat <= 90.0):
            raise ValueError("origin_lat must be in [-90, 90]")

        # origin_lon / origin_alt_m can be any float
        return p


if __name__ == "__main__":

    params = GeospatialParameters(
        origin_lat=40.44,
        origin_lon=-79.99,
        origin_alt=300.0,
        resolution_xy=20.0,
        resolutions_xy=[20.0, 100.0, 500.0],
        resolution_alt=5.0,
        resolutions_alt=[5.0, 25.0, 100.0],
    )

    enc = GeospatialEncoder(params)

    out1 = enc.encode((40.4433, -79.9436, 320.0))
    out2 = enc.encode((40.4433, -79.9436, 320.0))

    print(out1)

    assert out1 == out2
