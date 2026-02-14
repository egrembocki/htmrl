import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.input_layer.input_handler import InputHandler


class DummyEncoder(BaseEncoder):
    def __init__(self, size: int = 8):
        super().__init__(size=size)

    def attach_input(self, records: dict[str, list[float]]):
        self.input_records = records

    def encode(self, input_value: float) -> list[int]:
        return [0] * self.size


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
    encoder.attach_input(records)

    # HTM interface (not implemented yet)
    htm = HTMinterface([8, 1])

    # Act
    # Encode a single value
    last_value = records["value"][3]
    # sdr = SDR([8, 1])
    e = encoder.encode(last_value)

    # Mock interface call: Would accept an SDR instance as input.
    htm.consume_sdr(e)

    # Assert
    # Once HTMinterface is implemented, give it some observable state
    assert isinstance(records, dict)
    assert last_value == 2
    assert len(records["value"]) == 8
