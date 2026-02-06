import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.input_layer.input_handler import InputHandler


class DummyEncoder(BaseEncoder):

    def __init__(self, dimensions=None):
        if dimensions is None:
            dimensions = [8, 1]  # arbitrary SDR size for the test

    def attach_input(self, records: list[dict[str, float]]):
        self.input_records = records

    def encode(self, input_value: float) -> None:
        pass


# Note: This is expected to fail because we do not have an HTM interface yet
# from psu_capstone.htm.interface import HTMinterface


class HTMinterface:
    """
    Mock Class for HTM interface


    """

    def __init__(self, sdr: list[int]):
        self.last_received_sdr = sdr

    def consume_sdr(self, sdr: list[int]):
        self.last_received_sdr = sdr


def test_encoder_to_htm_receives_sdr_object():

    # Arrange
    fib_sequence = [0, 1, 1, 2, 3, 5, 8, 13]

    encoder = DummyEncoder()
    handler = InputHandler()
    records = handler.input_data(fib_sequence, required_columns=["value"])
    buffered_records = encoder.buffer_data(records)
    encoder.attach_input(buffered_records)

    # HTM interface (not implemented yet)
    htm = HTMinterface([8, 1])

    # Act
    # Encode a single value
    last_value = buffered_records[3]["value"]
    # sdr = SDR([8, 1])
    e = encoder.encode(last_value)

    # Mock interface call: Would accept an SDR instance as input.
    htm.consume_sdr(e)

    # Assert
    # Once HTMinterface is implemented, give it some observable state
    assert isinstance(buffered_records, list)
    assert last_value == 2
    assert len(buffered_records) == 8
