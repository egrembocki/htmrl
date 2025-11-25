import pandas as pd
import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler


class DummyEncoder(BaseEncoder):

    def __init__(self, dimensions=None):
        if dimensions is None:
            dimensions = [8, 1]  # arbitrary SDR size for the test

    def attach_input(self, df: pd.DataFrame):
        self.input_df = df

    def encode(self, input_value: float, output_sdr: SDR) -> None:
        pass


# Note: This is expected to fail because we do not have an HTM interface yet
# from psu_capstone.htm.interface import HTMinterface


class HTMinterface:
    """
    Mock Class for HTM interface


    """

    def __init__(self, sdr: SDR):
        self.last_received_sdr = sdr

    def consume_sdr(self, sdr: SDR):
        self.last_received_sdr = sdr


def test_encoder_to_htm_receives_sdr_object():

    # Arrange
    fib_sequence = [0, 1, 1, 2, 3, 5, 8, 13]

    handler = InputHandler()
    df = handler._to_dataframe(fib_sequence)

    encoder = DummyEncoder()
    encoder.attach_input(df)

    # HTM interface (not implemented yet)
    htm = HTMinterface(sdr=SDR([8, 1]))

    # Act
    # Encode a single value
    last_value = df.iloc[3].values[0]
    sdr = SDR([8, 1])
    encoder.encode(last_value, sdr)

    # Mock interface call: Would accept an SDR instance as input.
    htm.consume_sdr(sdr)

    # Assert
    # Once HTMinterface is implemented, give it some observable state
    assert isinstance(df, pd.DataFrame)
    assert last_value == 2
    assert df.shape == (8, 1)
    assert isinstance(sdr, SDR)
    assert htm.last_received_sdr is sdr
