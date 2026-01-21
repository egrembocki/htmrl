from datetime import datetime
from typing import Any, cast

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.date_encoder import DateEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.sdr_layer.sdr import SDR

if __name__ == "__main__":

    rdse = RandomDistributedScalarEncoder(RDSEParameters())

    encoder = RandomDistributedScalarEncoder(RDSEParameters())

    encoder.active_bits = 50
    encoder.size = 2058

    value = 3.14
    output_sdr = SDR([encoder.size])
    encoder.encode(value, output_sdr)

    date_encoder = DateEncoder()

    date_value = datetime(2023, 10, 5, 14, 30, 0)

    date_output_sdr = SDR([date_encoder.size])
    date_encoder.encode(date_value, date_output_sdr)

    season_encoder = cast(RandomDistributedScalarEncoder, date_encoder.season_encoder)
    print(
        f"Season Encoder Output SDR: {season_encoder.size}, active bits: {season_encoder.active_bits}"
    )

    season_encoder.active_bits = 56

    print(f"RDSE Encoder Output SDR: {output_sdr.size}")
    print(output_sdr.get_sparse())

    print(f"Date Encoder Output SDR: {date_output_sdr.size}")
    print(date_output_sdr.get_sparse())
