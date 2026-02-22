from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, Optional, override

import numpy as np
from pyproj import CRS, Transformer

from psu_capstone.encoder_layer.base_encoder import BaseEncoder, ParentDataclass
from psu_capstone.encoder_layer.simple_coordinate_encoder import (
    CoordinateEncoder2,
    CoordinateParameters2,
)

CRS_MERC = CRS.from_epsg(3857)
CRS_WGS84 = CRS.from_epsg(4326)
CRS_GEO = CRS.from_proj4("+proj=geocent +datum=WGS84 +units=m +no_defs")

T_WGS84_TO_GEO = Transformer.from_crs(CRS_WGS84, CRS_GEO, always_xy=True)
T_GEO_TO_WGS84 = Transformer.from_crs(CRS_GEO, CRS_WGS84, always_xy=True)
T_WGS84_TO_MERC = Transformer.from_crs(CRS_WGS84, CRS_MERC, always_xy=True)
T_MERC_TO_WGS84 = Transformer.from_crs(CRS_MERC, CRS_WGS84, always_xy=True)


class GeospatialEncoder2(
    BaseEncoder[tuple[float, float, float] | tuple[float, float, float, float]]
):
    def __init__(self, geo_params: "GeospatialParameters", coord2_params: "CoordinateParameters2"):
        self._geo_params = copy.deepcopy(geo_params)
        self._coord2_params = copy.deepcopy(coord2_params)

        dims = 4 if self._geo_params.use_altitude else 3

        # Build per-dim resolutions:
        # - x,y,(z) are already in "grid units" (meters/scale) then rounded -> resolution 1 grid
        # - radius is integer grid -> resolution 1
        if coord2_params.resolutions is None:
            resolutions = (1.0,) * dims
        else:
            resolutions = coord2_params.resolutions
            if len(resolutions) != dims:
                raise ValueError("CoordinateParameters2.resolutions must match dims")

        params = CoordinateParameters2(
            n=coord2_params.n,
            w=coord2_params.w,
            dims=dims,
            seed=coord2_params.seed,
            resolution=coord2_params.resolution,
            resolutions=resolutions,
        )

        self._encoder = CoordinateEncoder2(params)
        super().__init__(self._encoder.size)

    @override
    def encode(
        self, input_value: tuple[float, float, float] | tuple[float, float, float, float]
    ) -> list[int]:
        if len(input_value) == 3:
            speed, lon, lat = input_value
            alt = None
        elif len(input_value) == 4:
            speed, lon, lat, alt = input_value
        else:
            raise ValueError("Expected (speed, lon, lat) or (speed, lon, lat, alt).")

        lon = self._wrap_lon(float(lon))
        lat = self._clamp_lat(float(lat))
        speed = float(speed)
        alt = None if alt is None else float(alt)

        # Convert position to grid coords (ints), then encode as floats to RDSEs
        coord = self.coordinate_for_position(lon, lat, alt)
        r = float(self.radius_for_speed(speed))

        if self._geo_params.use_altitude and alt is not None:
            x, y, z = coord
            vec = (float(x), float(y), float(z), r)
        else:
            x, y = coord
            vec = (float(x), float(y), r)

        return self._encoder.encode(vec)

    def decode(
        self,
        encoded: list[int],
        candidates: Iterable[tuple[float, ...]] | None = None,
    ) -> tuple[tuple[float, float, Optional[float]] | None, float]:
        decoded_vec, conf = self._encoder.decode(encoded, candidates=candidates)

        if self._geo_params.use_altitude and len(decoded_vec) == 4:
            x, y, z, _r = decoded_vec
            coord = (int(round(x)), int(round(y)), int(round(z)))
        else:
            x, y, _r = decoded_vec
            coord = (int(round(x)), int(round(y)))

        lon, lat, alt = self.position_for_coordinate(coord)
        return (lon, lat, alt, _r), conf

    def coordinate_for_position(
        self, lon: float, lat: float, alt: Optional[float]
    ) -> tuple[int, ...]:
        scale = float(self._geo_params.scale)

        if self._geo_params.use_altitude and alt is not None:
            x, y, z = T_WGS84_TO_GEO.transform(lon, lat, float(alt))
            coord = np.array([x, y, z], dtype=float) / scale
            return tuple(int(round(v)) for v in coord)

        x_m, y_m = T_WGS84_TO_MERC.transform(lon, lat)
        coord = np.array([x_m, y_m], dtype=float) / scale
        return tuple(int(round(v)) for v in coord)

    def position_for_coordinate(
        self, coord: tuple[int, ...]
    ) -> tuple[float, float, Optional[float]]:
        scale = float(self._geo_params.scale)

        if self._geo_params.use_altitude and len(coord) == 3:
            x, y, z = (coord[0] * scale, coord[1] * scale, coord[2] * scale)
            lon, lat, alt = T_GEO_TO_WGS84.transform(x, y, z)
            return self._wrap_lon(float(lon)), self._clamp_lat(float(lat)), float(alt)

        x_m, y_m = (coord[0] * scale, coord[1] * scale)
        lon, lat = T_MERC_TO_WGS84.transform(x_m, y_m)
        return self._wrap_lon(float(lon)), self._clamp_lat(float(lat)), None

    def radius_for_speed(self, speed_mps: float) -> int:
        overlap = 1.5
        coords_per_timestep = speed_mps * float(self._geo_params.timestep)
        r = int(round((coords_per_timestep / 2.0) * overlap))

        min_r = int(math.ceil((math.sqrt(self._coord2_params.w) - 1.0) / 2.0))

        r = max(r, min_r)
        r = max(0, min(r, self._geo_params.max_radius))
        return r

    @staticmethod
    def _clamp_lat(lat: float) -> float:
        return max(-85.05112878, min(85.05112878, lat))

    @staticmethod
    def _wrap_lon(lon: float) -> float:
        return (lon + 180.0) % 360.0 - 180.0


@dataclass
class GeospatialParameters(ParentDataclass):
    scale: float = 5.0
    timestep: float = 1.0
    max_radius: int = 2
    use_altitude: bool = True
    encoder_class = GeospatialEncoder2


if __name__ == "__main__":
    coord_params = CoordinateParameters2(n=400, w=25)
    geo_params = GeospatialParameters(scale=0.01, timestep=2.0, max_radius=40, use_altitude=True)

    enc = GeospatialEncoder2(geo_params=geo_params, coord2_params=coord_params)

    a = enc.encode((5.0, -177.0365, 38.8977, 10.0))
    b = enc.encode((5.0, -77.0365, 38.897, 10.5))

    print(enc.decode(a))
    print(enc.decode(b))
