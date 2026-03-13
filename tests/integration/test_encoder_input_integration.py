import datetime
from typing import Any

import pandas as pd
import pytest

from psu_capstone.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.input_layer.input_handler import InputHandler


@pytest.fixture
def input_handler() -> InputHandler:
    return InputHandler()


@pytest.fixture
def rdse_params() -> RDSEParameters:
    return RDSEParameters(
        size=256,
        active_bits=21,
        sparsity=0.0,
        radius=0.0,
        resolution=1.0,
        category=False,
        seed=123,
    )


@pytest.fixture
def rdse(rdse_params: RDSEParameters) -> RandomDistributedScalarEncoder:
    return RandomDistributedScalarEncoder(rdse_params)


def count_ones(vec: list[int]) -> int:
    return sum(1 for b in vec if b)


class TestInputRDSEIntegration:

    # Had errors where I had seq_map["values"] instead of seq_map["value"]
    def test_input_dict_numeric_column_encodes_all_values(
        self,
        input_handler: InputHandler,
        rdse: RandomDistributedScalarEncoder,
        rdse_params: RDSEParameters,
    ):
        payload = {"value": [0.1, 1.2, 5.5, 10.0]}

        seq_map = input_handler.input_data(payload)
        assert list(seq_map.keys()) == ["value"]
        values = input_handler.get_column_data("value")
        assert values == [0.1, 1.2, 5.5, 10.0]

        for v in values:
            vec = rdse.encode(float(v))
            assert len(vec) == rdse.size
            assert count_ones(vec) == rdse._active_bits

    # Had errors because I forgot to include self
    def test_input_bytes_csv_parses_then_encodes(
        self,
        input_handler: InputHandler,
        rdse: RandomDistributedScalarEncoder,
        rdse_params: RDSEParameters,
    ):
        payload = bytearray(b"value\n0.0\n1.0\n2.0\n")

        seq_map = input_handler.input_data(payload)
        assert list(seq_map.keys()) == ["value"]
        values = input_handler.get_column_data("value")

        assert values == ["0.0", "1.0", "2.0"]

        for v in values:
            vec = rdse.encode(float(v))
            assert len(vec) == rdse.size
            assert count_ones(vec) == rdse._active_bits

    def test_input_filters_none_values_before_encoding(
        self,
        input_handler: InputHandler,
        rdse: RandomDistributedScalarEncoder,
        rdse_params: RDSEParameters,
    ):
        payload = [
            {"value": 1.0},
            {"value": None},
            {"value": 2.0},
        ]

        seq_map = input_handler.input_data(payload)
        assert seq_map["value"] == [1.0, 2.0]

        for v in seq_map["value"]:
            vec = rdse.encode(float(v))
            assert len(vec) == rdse.size

    def test_inputs_requires_column_when_multiple_columns_present(
        self, input_handler: InputHandler
    ):
        payload = [{"a": 1.0, "b": 2.0}]

        input_handler.input_data(payload, required_columns=["a", "b"])

        with pytest.raises(ValueError, match="Column must be specified"):
            input_handler.get_column_data(None)

    def test_datetime_column_present_does_not_break_numeric_extraction(
        self,
        input_handler: InputHandler,
        rdse: RandomDistributedScalarEncoder,
        rdse_params: RDSEParameters,
    ):
        payload = [
            {"t": "2026-03-01T12:00:00", "value": 1.0},
            {"t": "2026-03-01T12:00:01", "value": 2.0},
        ]

        seq_map = input_handler.input_data(payload)
        assert seq_map["value"] == [1.0, 2.0]

        for v in seq_map["value"]:
            vec = rdse.encode(float(v))
            assert len(vec) == rdse.size

    def test_roundtrip_encode_then_decode_from_input_sequence(
        self, input_handler: InputHandler, rdse: RandomDistributedScalarEncoder
    ):
        payload = {"value": [0.0, 1.0, 2.0, 3.0]}
        input_handler.input_data(payload)
        values = input_handler.get_column_data("value")

        encoded = []
        for v in values:
            encoded.append((v, rdse.encode(v)))

        for original, vec in encoded[::2]:
            decoded_value, overlap = rdse.decode(vec)
            assert decoded_value == original
            assert overlap > 0


@pytest.fixture
def date_params() -> DateEncoderParameters:
    return DateEncoderParameters()


@pytest.fixture
def date_encoder(date_params: DateEncoderParameters) -> DateEncoder:
    return DateEncoder(date_params)


class TestInputDateIntegration:

    def test_datetime_objects_from_dict_rows_encoder(
        self, input_handler: InputHandler, date_encoder: DateEncoder
    ):
        payload = [
            {"t": datetime.datetime(2026, 3, 1, 12, 0, 0)},
            {"t": datetime.datetime(2026, 3, 1, 12, 0, 1)},
        ]

        print(datetime.datetime(2026, 3, 1, 12, 0, 0))

        seq_map = input_handler.input_data(payload)
        values = seq_map["t"]
        assert len(values) == 2

        for v in values:

            print(v)

            if isinstance(v, str):
                v = datetime.datetime.fromisoformat(v)

            sdr = date_encoder.encode(v)

            assert len(sdr) == date_encoder.size
            assert sum(sdr) > 0

    def test_epoch_seconds_from_input_encode(
        self, input_handler: InputHandler, date_encoder: DateEncoder
    ):
        payload = {"t": [1700000000.0, 1700003600.0, 1700086400.0]}

        input_handler.input_data(payload)
        values = input_handler.get_column_data("t")
        assert values == [1700000000.0, 1700003600.0, 1700086400.0]

        for v in values:
            vec = date_encoder.encode(v)
            assert len(vec) == date_encoder.size

    def test_input_filters_none_values_before_date_encoding(
        self, input_handler: InputHandler, date_encoder: DateEncoder
    ):
        payload = [
            {"t": datetime.datetime(2026, 3, 1, 12, 0, 0)},
            {"t": None},
            {"t": datetime.datetime(2026, 3, 1, 12, 0, 2)},
        ]

        input_handler.input_data(payload)
        values = input_handler.get_column_data("t")
        print(values)
        assert values == [
            pd.Timestamp("2026-03-01 12:00:00"),
            pd.Timestamp("2026-03-01 12:00:02"),
        ]

        for v in values:
            vec = date_encoder.encode(v)
            assert len(vec) == date_encoder.size

    def test_iso_strings_are_accepted(self, input_handler: InputHandler, date_encoder: DateEncoder):
        payload = [
            {"t": "2026-03-01T12:00:00"},
            {"t": "2026-03-01T12:00:01"},
        ]

        input_handler.input_data(payload)
        values = input_handler.get_column_data("t")
        assert values == [
            pd.Timestamp("2026-03-01 12:00:00"),
            pd.Timestamp("2026-03-01 12:00:01"),
        ]

        for v in values:
            vec = date_encoder.encode(v)
            assert len(vec) == date_encoder.size


@pytest.fixture
def category_params() -> CategoryParameters:
    return CategoryParameters(
        w=4,
        category_list=["red", "green", "blue"],
        rdse_used=True,
    )


@pytest.fixture
def category_encoder(category_params: CategoryParameters) -> CategoryEncoder:
    return CategoryEncoder(category_params)


class TestInputCategoryIntegration:
    def test_input_dict_category_column_encodes_all_values(
        self,
        input_handler: InputHandler,
        category_encoder: CategoryEncoder,
        category_params: CategoryParameters,
    ):
        payload = {"color": ["red", "green", "blue"]}

        input_handler.input_data(payload)
        values = input_handler.get_column_data("color")

        assert values == ["red", "green", "blue"]

        for v in values:
            vec = category_encoder.encode(v)
            assert len(vec) == category_encoder.size
            assert count_ones(vec) == category_params.w

    def test_none_category_rows_are_removed(
        self,
        input_handler: InputHandler,
        category_encoder: CategoryEncoder,
    ):
        payload = [
            {"color": "red"},
            {"color": None},
            {"color": "blue"},
        ]

        input_handler.input_data(payload)
        values = input_handler.get_column_data("color")

        assert values == ["red", "blue"]

        for v in values:
            vec = category_encoder.encode(v)
            assert len(vec) == category_encoder.size

    # Fails currently
    def test_input_bytes_csv_parses_then_category_encodes(
        self,
        input_handler: InputHandler,
        category_encoder: CategoryEncoder,
        category_params: CategoryParameters,
    ):
        payload = bytearray(b"color\nred\ngreen\nblue\n")

        input_handler.input_data(payload)
        values = input_handler.get_column_data("color")

        assert values == ["red", "green", "blue"]

        for v in values:
            vec = category_encoder.encode(v)
            print(vec)
            assert len(vec) == category_encoder.size
            assert count_ones(vec) == category_params.w

    def test_roundtrip_known_category_encode_then_decode(
        self,
        input_handler: InputHandler,
        category_encoder: CategoryEncoder,
    ):
        payload = {"color": ["red", "green", "blue"]}

        input_handler.input_data(payload)
        values = input_handler.get_column_data("color")

        for v in values:
            vec = category_encoder.encode(v)
            decoded_value, confidence = category_encoder.decode(vec)
            assert decoded_value == v
            assert confidence > 0
