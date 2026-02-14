from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable, Optional, override

import numpy as np
from pyproj import CRS, Transformer

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.coordinate_encoder import CoordinateEncoder, CoordinateParameters

# Good for 2D
CRS_MERC = CRS.from_epsg(3857)

# Represents raw GPS coordinates
CRS_WGS84 = CRS.from_epsg(4326)

# Good for 3D
CRS_GEO = CRS.from_proj4("+proj=geocent +datum=WGS84 +units=m +no_defs")


T_WGS84_TO_GEO = Transformer.from_crs(CRS_WGS84, CRS_GEO, always_xy=True)
T_GEO_TO_WGS84 = Transformer.from_crs(CRS_GEO, CRS_WGS84, always_xy=True)

T_WGS84_TO_MERC = Transformer.from_crs(CRS_WGS84, CRS_MERC, always_xy=True)
T_MERC_TO_WGS84 = Transformer.from_crs(CRS_MERC, CRS_WGS84, always_xy=True)


class GeospatialEncoder(
    BaseEncoder[tuple[float, float, float] | tuple[float, float, float, float]]
):

    def __init__(
        self,
        geo_params: GeospatialParameters,
        coord_params: CoordinateParameters,
        dimensions: list[int] | None = None,
    ):

        self._geo_params = copy.deepcopy(geo_params)
        self._coord_params = copy.deepcopy(coord_params)

        coord_params = CoordinateParameters(
            n=coord_params.n,
            w=coord_params.w,
            seed=coord_params.seed,
            max_radius=self._geo_params.max_radius,
            dims=3 if self._geo_params.use_altitude else 2,
        )

        self._encoder = CoordinateEncoder(coord_params)

        super().__init__(dimensions, self._encoder.size)

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

        lon = float(lon)
        lat = float(lat)
        speed = float(speed)
        alt = None if alt is None else float(alt)

        lon = self._wrap_lon(lon)
        lat = self._clamp_lat(lat)

        coord = self.coordinate_for_position(lon, lat, alt)
        radius = self.radius_for_speed(speed)

        return self._encoder.encode((coord, radius))

    def decode(
        self,
        encoded: list[int],
        candidates: Iterable[tuple[tuple[int, ...], int]] | None = None,
    ) -> tuple[tuple[float, float, Optional[float]] | None, float]:

        best_key, conf = self._encoder.decode(encoded, candidates=candidates)
        if best_key is None:
            return None, 0.0

        grid_coord, _radius = best_key
        pos = self.position_for_coordinate(grid_coord)
        return pos, conf

    def coordinate_for_position(
        self, lon: float, lat: float, alt: Optional[float]
    ) -> tuple[int, ...]:

        scale = float(self._geo_params.scale)

        if self._geo_params.use_altitude and alt is not None:
            # Mercator meters and altitude to geocentric meters
            x, y, z = T_WGS84_TO_GEO.transform(lon, lat, float(alt))
            coord = np.array([x, y, z], dtype=float) / float(self._geo_params.scale)
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
            lon = self._wrap_lon(float(lon))
            lat = self._clamp_lat(float(lat))
            return lon, lat, alt

        x_m, y_m = (coord[0] * scale, coord[1] * scale)
        lon, lat = T_MERC_TO_WGS84.transform(x_m, y_m)
        lon = self._wrap_lon(float(lon))
        lat = self._clamp_lat(float(lat))
        return lon, lat, None

    def radius_for_speed(self, speed_mps: float) -> int:

        overlap = 1.5
        coords_per_timestep = speed_mps * float(self._geo_params.timestep)
        r = int(round((coords_per_timestep / 2.0) * overlap))

        min_r = int(math.ceil((math.sqrt(self._coord_params.w) - 1.0) / 2.0))

        r = max(r, min_r)
        r = max(0, min(r, self._geo_params.max_radius))
        return r

    @staticmethod
    def _clamp_lat(lat: float) -> float:
        # Web Mercator is undefinded at poles so we should clamp it
        return max(-85.05112878, min(85.05112878, lat))

    @staticmethod
    def _wrap_lon(lon: float) -> float:
        # wrap to [-180, 180)
        lon = (lon + 180.0) % 360.0 - 180.0
        return lon


@dataclass
class GeospatialParameters:
    # meters per grid unit
    scale: float = 5.0

    # seconds between readings
    timestep: float = 1.0

    # clamp output radius
    max_radius: int = 2

    use_altitude: bool = True

    encoder_class = GeospatialEncoder


if __name__ == "__main__":
    coord_params = CoordinateParameters(n=400, w=25)
    geo_params = GeospatialParameters(scale=0.2, timestep=2.0, max_radius=40, use_altitude=True)

    enc = GeospatialEncoder(geo_params=geo_params, coord_params=coord_params)

    a = enc.encode((5.0, -177.0365, 38.8977, 10.0))
    b = enc.encode((5.0, -77.0365, 38.897, 10.5))

    print(enc.decode(a))
    print(enc.decode(b))
