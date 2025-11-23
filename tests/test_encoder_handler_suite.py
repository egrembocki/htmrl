"""Test cases for EncoderHandler to build union SDRs"""

import copy
from datetime import datetime
from typing import List

import pandas as pd
import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def handler() -> EncoderHandler:
    """Fixture to create an EncoderHandler with multiple encoders"""

    df = pd.DataFrame(
        [
            {
                "float_col": float(3.14),  # rdse
                "int_col": int(42),  # scalar
                "str_col": str("B"),  # category
                "date_col": datetime(2023, 12, 25),  # date
            }
        ]
    )

    handler = EncoderHandler(df)

    return handler


def test_handler_singleton(handler: EncoderHandler):
    """Test that EncoderHandler enforces singleton pattern"""

    # Arrange
    test_input = handler._data_frame

    # Act
    h1 = handler
    h2 = EncoderHandler(test_input)

    # Assert
    assert h1 is h2


def test_copy_deepcopy_sdr(handler: EncoderHandler):
    """Test copying and deep copying SDRs from multiple encoders"""

    # Arrange
    test_data = handler._data_frame
    rows = test_data.iloc[0]
    sdrs = []

    # Act
    handler.build_composite_sdr(test_data)

    # Assert that a deep copy occurs
    for i, encoder in enumerate(handler._encoders):
        input_value = rows.iloc[i]

        output_sdr = SDR([encoder.size])

        output_sdr.zero()

        assert output_sdr.get_sparse() == []

        try:
            encoder.encode(input_value, output_sdr)
            assert output_sdr.get_sparse() != []
        except Exception as e:
            pytest.fail(f"Encoding failed with exception: {e}")

        copied_sdr = copy.deepcopy(output_sdr)
        sdrs.append(copied_sdr)
        assert sdrs[i].get_sparse() == output_sdr.get_sparse()
        assert sdrs[i].get_sparse() != []
        output_sdr.zero()
        assert sdrs[i].get_sparse() != output_sdr.get_sparse()
