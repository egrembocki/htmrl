import math
import random

import pytest

from psu_capstone.encoder_layer.coordinate_encoder import CoordinateParameters
from psu_capstone.encoder_layer.geospatial_encoder import GeospatialEncoder, GeospatialParameters


def test_decode_round_trip_3d():
    coord_params = CoordinateParameters(n=400, w=25)
    geo_params = GeospatialParameters(
        scale=5.0,
        timestep=1.0,
        max_radius=10,
        use_altitude=True,
    )

    enc = GeospatialEncoder(geo_params, coord_params)

    original = (3.0, -77.0365, 38.8977, 15.0)  # speed, lon, lat, alt
    sdr = enc.encode(original)

    decoded_pos, conf = enc.decode(sdr)

    assert decoded_pos is not None
    lon, lat, alt = decoded_pos

    assert math.isclose(lon, original[1], abs_tol=1e-4)
    assert math.isclose(lat, original[2], abs_tol=1e-4)
    assert math.isclose(alt, original[3], abs_tol=1.0)

    assert conf > 0.5


def test_decode_round_trip_2d():
    coord_params = CoordinateParameters(n=400, w=25)
    geo_params = GeospatialParameters(
        scale=5.0,
        timestep=1.0,
        max_radius=10,
        use_altitude=False,
    )

    enc = GeospatialEncoder(geo_params, coord_params)

    original = (2.0, -122.4194, 37.7749)  # speed, lon, lat
    sdr = enc.encode(original)

    decoded_pos, conf = enc.decode(sdr)

    assert decoded_pos is not None
    lon, lat, alt = decoded_pos

    assert math.isclose(lon, original[1], abs_tol=1e-4)
    assert math.isclose(lat, original[2], abs_tol=1e-4)
    assert alt is None
    assert conf > 0.5


def test_decode_respects_wrap_and_clamp():
    coord_params = CoordinateParameters(n=400, w=25)
    geo_params = GeospatialParameters(scale=5.0, use_altitude=False)

    enc = GeospatialEncoder(geo_params, coord_params)

    original = (2.0, 190.0, 95.0)

    encoded = enc.encode(original)
    decoded, conf = enc.decode(encoded)

    assert decoded is not None
    lon, lat, _ = decoded

    assert -180.0 <= lon < 180.0

    assert -85.05112878 <= lat <= 85.05112878
