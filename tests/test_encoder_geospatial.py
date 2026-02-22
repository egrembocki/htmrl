import pytest

from psu_capstone.encoder_layer.coordinate_encoder import CoordinateParameters
from psu_capstone.encoder_layer.geospatial_encoder import GeospatialEncoder, GeospatialParameters


def _build_encoder(
    *, use_altitude: bool, scale: float = 5.0, timestep: float = 1.0, max_radius: int = 10
):
    coord_params = CoordinateParameters(
        n=400, w=25, seed=123, max_radius=max_radius, dims=3 if use_altitude else 2
    )
    geo_params = GeospatialParameters(
        scale=scale, timestep=timestep, max_radius=max_radius, use_altitude=use_altitude
    )
    return GeospatialEncoder(geo_params=geo_params, coord_params=coord_params)


def _overlap(a: list[int], b: list[int]) -> int:
    return sum(1 for x, y in zip(a, b) if x and y)


def _active_count(a: list[int]) -> int:
    return sum(1 for x in a if x)


def test_encode_wrap_lon_equivalence_over_dateline():
    enc = _build_encoder(use_altitude=False, scale=5.0)

    # same physical longitude, different representations
    a = enc.encode((1.0, 179.9, 10.0))
    b = enc.encode((1.0, 539.9, 10.0))  # 179.9 + 360
    c = enc.encode((1.0, -180.1, 10.0))  # should wrap close to 179.9

    assert a == b
    assert a == c


def test_encode_clamp_lat_at_poles_is_stable():
    enc = _build_encoder(use_altitude=False, scale=5.0)

    # absurd latitudes should clamp to +/-85.05112878
    a = enc.encode((0.0, 0.0, 9999.0))
    b = enc.encode((0.0, 0.0, 85.05112878))
    c = enc.encode((0.0, 0.0, -9999.0))
    d = enc.encode((0.0, 0.0, -85.05112878))

    assert a == b
    assert c == d


def test_encode_speed_increases_locality_radius_and_tends_to_increase_overlap():
    enc = _build_encoder(use_altitude=False, scale=2.0, timestep=2.0, max_radius=100)

    # two nearby positions
    p1 = (0.0, -77.0365, 38.8977)
    p2 = (0.0, -77.0367, 38.9423)

    slow1 = enc.encode((0.5, p1[1], p1[2]))
    slow2 = enc.encode((0.5, p2[1], p2[2]))

    fast1 = enc.encode((30.0, p1[1], p1[2]))
    fast2 = enc.encode((30.0, p2[1], p2[2]))

    overlap_slow = _overlap(slow1, slow2)
    overlap_fast = _overlap(fast1, fast2)

    assert overlap_fast >= overlap_slow


def test_encode_small_position_change_has_more_overlap_than_large_change():
    enc = _build_encoder(use_altitude=False, scale=10.0, timestep=1.0, max_radius=10)

    base = enc.encode((2.0, -77.0365, 38.8977))
    near = enc.encode((2.0, -77.0366, 38.89775))
    far = enc.encode((2.0, -80.0, 41.0))

    overlap_near = _overlap(base, near)
    overlap_far = _overlap(base, far)

    assert overlap_near > overlap_far


def test_encode_altitude_mode_changes_encoding_when_altitude_changes():
    enc = _build_encoder(use_altitude=True, scale=1.0, timestep=1.0, max_radius=10)

    a = enc.encode((1.0, -77.0365, 38.8977, 10.0))
    b = enc.encode((1.0, -77.0365, 38.8977, 200.0))

    assert a != b
    assert _overlap(a, b) < _active_count(a)
