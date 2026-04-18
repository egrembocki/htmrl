"""Tests for structural ParameterMarker compatibility across encoder parameter classes."""

# Test Suite: TS-23 (Encoder-ParameterMarker)

import pytest

from psu_capstone.encoder_layer.base_encoder import ParameterMarker
from psu_capstone.encoder_layer.category_encoder import CategoryParameters
from psu_capstone.encoder_layer.coordinate_encoder import CoordinateParameters
from psu_capstone.encoder_layer.date_encoder import DateEncoderParameters
from psu_capstone.encoder_layer.delta_encoder import DeltaEncoderParameters
from psu_capstone.encoder_layer.fourier_encoder import FourierEncoderParameters
from psu_capstone.encoder_layer.geospatial_encoder import GeospatialParameters
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoderParameters


@pytest.mark.parametrize(
    "params",
    [
        CategoryParameters(),
        CoordinateParameters(),
        DateEncoderParameters(),
        DeltaEncoderParameters(),
        FourierEncoderParameters(),
        GeospatialParameters(),
        RDSEParameters(),
        ScalarEncoderParameters(),
    ],
    ids=[
        "category",
        "coordinate",
        "date",
        "delta",
        "fourier",
        "geospatial",
        "rdse",
        "scalar",
    ],
)
# Test Type: unit test
def test_encoder_parameter_is_parameter_marker(params):
    # TS-23 TC-204
    """All encoder parameter classes should satisfy the ParameterMarker protocol."""
    assert isinstance(params, ParameterMarker)
