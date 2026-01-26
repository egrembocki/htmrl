from datetime import datetime
from typing import Any, cast

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.sdr_layer.sdr import SDR

if __name__ == "__main__":

    rdse_encoder = RandomDistributedScalarEncoder()  # default RDSE encoder test

    print(f"RDSE Encoder Size: {rdse_encoder.size}")

    scalar_encoder = ScalarEncoder()  # default scalar encoder test

    rdse_encoder.size = 2058

    print(f"RDSE Encoder Size after set: {rdse_encoder.size}")

    rdse_encoder.active_bits = 50

    print(f"RDSE Encoder Active Bits: {rdse_encoder.active_bits}")

    rdse_encoder.resolution = 0.5

    print(f"RDSE Encoder Resolution: {rdse_encoder.resolution}")

    rdse_encoder.sparsity = 0.25

    print(f"RDSE Encoder Sparsity: {rdse_encoder.sparsity}")

    rdse_encoder.category = True

    print(f"RDSE Encoder Category: {rdse_encoder.category}")

    rdse_encoder.seed = 12345

    print(f"RDSE Encoder Seed: {rdse_encoder.seed}")

    date_encoder = DateEncoder()  # default date encoder test

    season_encoder = cast(RandomDistributedScalarEncoder, date_encoder.season_encoder)
    day_of_week_encoder = cast(RandomDistributedScalarEncoder, date_encoder.dayofweek_encoder)
    weekend_encoder = cast(RandomDistributedScalarEncoder, date_encoder.weekend_encoder)
    holiday_encoder = cast(RandomDistributedScalarEncoder, date_encoder.holiday_encoder)
    time_of_day_encoder = cast(RandomDistributedScalarEncoder, date_encoder.timeofday_encoder)
    custom_day_encoder = cast(RandomDistributedScalarEncoder, date_encoder.customdays_encoder)

    print(f"Date Encoder Size: {date_encoder.size}")

    date_encoder.rdse_sizes = {
        "season": 16,
        "dayOfWeek": 32,
        "weekend": 64,
        "holiday": 128,
        "timeOfDay": 256,
        "customDays": 512,
    }

    print(f"Season Encoder Size after set: {season_encoder.size}")
    print(f"Day of Week Encoder Size after set: {day_of_week_encoder.size}")
    print(f"Weekend Encoder Size after set: {weekend_encoder.size}")
    print(f"Holiday Encoder Size after set: {holiday_encoder.size}")
    print(f"Time of Day Encoder Size after set: {time_of_day_encoder.size}")
    print(f"Custom Days Encoder Size after set: {custom_day_encoder.size}")

    print(f"Date Encoder Size after setting individual encoders: {date_encoder.size}")

    test_date = datetime(2023, 3, 15, 14, 30)  # March 15, 2023, 14:30
    output_sdr = SDR([date_encoder.size])
    date_encoder.encode(test_date, output_sdr)
    print(f"Encoded SDR for {test_date}: {output_sdr.get_sparse()}")

    # Change individual encoder sizes
    date_encoder.season_encoder.size = 20

    print(f"Season Encoder Size 20 :: {date_encoder.rdse_sizes['season']}")

    output_sdr.size = date_encoder.size  # Update SDR size after changing encoder sizes

    date_encoder.encode(test_date, output_sdr)

    print(
        f"Encoded SDR after changing individual encoder sizes for {test_date}: {output_sdr.get_sparse()}"
    )
    print(f"Date Encoder Size after changing individual encoder sizes: {date_encoder.size}")
