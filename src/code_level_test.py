from datetime import datetime
from typing import Any, cast

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.sdr_layer.sdr import SDR

if __name__ == "__main__":

    date_encoder = DateEncoder()  # default date encoder test

    date_encoder.rdse_sizes = {
        "season": 16,
        "dayOfWeek": 32,
        "weekend": 64,
        "holiday": 128,
        "timeOfDay": 256,
        "customDays": 512,
    }

    print(f"Season Encoder Size after set: {date_encoder.season_size}")
    print(f"Day of Week Encoder Size after set: {date_encoder.dayofweek_size}")
    print(f"Weekend Encoder Size after set: {date_encoder.weekend_size}")
    print(f"Holiday Encoder Size after set: {date_encoder.holiday_size}")
    print(f"Time of Day Encoder Size after set: {date_encoder.timeofday_size}")
    print(f"Custom Days Encoder Size after set: {date_encoder.customdays_size}")

    print(f"Date Encoder Size after setting individual encoders: {date_encoder.size}")

    test_date = datetime(2023, 3, 15, 14, 30)  # March 15, 2023, 14:30
    output_sdr = SDR([date_encoder.size])
    date_encoder.encode(test_date, output_sdr)
    print(f"Encoded SDR for {test_date}: {output_sdr.get_sparse()}")

    # Change individual encoder sizes
    date_encoder.season_size = 20

    print(f"Season Encoder Size 20 :: {date_encoder.season_encoder.size}")

    output_sdr.size = date_encoder.size  # Update SDR size after changing encoder sizes

    date_encoder.encode(test_date, output_sdr)

    print(
        f"Encoded SDR after changing individual encoder sizes for {test_date}: {output_sdr.get_sparse()}"
    )
    print(f"Date Encoder Size after changing individual encoder sizes: {date_encoder.size}")
