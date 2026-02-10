from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, override

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

# Turns GPS input into a flat metric space
T_WGS84_TO_MERC = Transformer.from_crs(CRS_WGS84, CRS_MERC, always_xy=True)

# Allows altitude to meaningfully affect the encoding
T_MERC_TO_GEO = Transformer.from_crs(CRS_MERC, CRS_GEO, always_xy=True)


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

        self._geo_params = geo_params

        coord_params = CoordinateParameters(
            n=coord_params.n,
            w=coord_params.w,
            seed=coord_params.seed,
            max_radius=self._geo_params.max_radius,
        )

        self._encoder = CoordinateEncoder(coord_params)

        super().__init__(dimensions, self._encoder.size)

    @override
    def encode(self, input_value):
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

    def coordinate_for_position(
        self, lon: float, lat: float, alt: Optional[float]
    ) -> tuple[int, ...]:
        # lon/lat to Mercator meters
        x_m, y_m = T_WGS84_TO_MERC.transform(lon, lat)

        if self._geo_params.use_altitude and alt is not None:
            # Mercator meters and altitude to geocentric meters
            x, y, z = T_MERC_TO_GEO.transform(x_m, y_m, alt)
            coord = np.array([x, y, z], dtype=float) / float(self._geo_params.scale)
            return tuple(int(round(v)) for v in coord)
        else:
            coord = np.array([x_m, y_m], dtype=float) / float(self._geo_params.scale)
            return tuple(int(round(v)) for v in coord)

    def radius_for_speed(self, speed_mps: float) -> int:

        overlap = 1.5
        coords_per_timestep = speed_mps * float(self._geo_params.timestep)
        r = int(round((coords_per_timestep / 2.0) * overlap))

        min_r = int(math.ceil((math.sqrt(self._encoder._w) - 1.0) / 2.0))

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
    scale: float = 10.0

    # seconds between readings
    timestep: float = 1.0

    # clamp output radius
    max_radius: int = 2

    use_altitude: bool = True

    encoder_class = GeospatialEncoder


if __name__ == "__main__":
    coord_params = CoordinateParameters(n=40, w=25, max_radius=2)
    geo_params = GeospatialParameters(scale=10.0, timestep=1.0, max_radius=2, use_altitude=True)

    enc = GeospatialEncoder(geo_params=geo_params, coord_params=coord_params)

    a = enc.encode((5.0, -177.0365, 38.8977, 10.0))
    b = enc.encode((5.0, -77.0365, 38.897, 10.0))

    lon = -1177.0365
    lon1 = (lon + 180.0) % 360.0 - 180.0
    print(lon1)

    def overlap_count(x: list[int], y: list[int]) -> int:
        return int(np.count_nonzero(np.array(x) == np.array(y)))

    print("len:", len(a))
    print("overlap:", overlap_count(a, b))

    print(a)
