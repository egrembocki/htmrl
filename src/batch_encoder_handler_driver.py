import time

import pandas as pd

from psu_capstone.encoder_layer.batch_encoder_handler import BatchEncoderHandler
from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler


def main():
    handler = InputHandler.get_instance()
    # skipping the timestamp req in input handler
    handler._prepend_timestamp_column = lambda df: df

    excel_file_path = r"C:\Users\Josh\Downloads\rsi_data\RSI_column.xlsx"

    input_data = handler.input_data(input_source=excel_file_path, required_columns=["RSI"])

    print(input_data)

    encoder = BatchEncoderHandler(input_data)
    start_time = time.perf_counter()
    sdrs = encoder.build_composite_sdr(input_data, threads_per_column=16)
    end_time = time.perf_counter()
    print(f"Encoded {len(sdrs)} SDRs for RSI column")
    print(f"Encoding took {end_time - start_time:.4f} seconds")
    print("Sample SDRs:")
    for i, sdr in enumerate(sdrs[:5]):
        print(sdr.get_sparse())

    print("Deterministic check with new encoders and same seed. First 5 data points:")
    params = RDSEParameters(
        size=2048, active_bits=40, sparsity=0.0, radius=1, resolution=0, category=False, seed=1
    )
    e1 = RandomDistributedScalarEncoder(params)
    o1 = SDR([e1.size])
    e1.encode(42.44, o1)
    print(o1.get_sparse())

    e1 = RandomDistributedScalarEncoder(params)
    o1 = SDR([e1.size])
    e1.encode(64.44, o1)
    print(o1.get_sparse())

    e1 = RandomDistributedScalarEncoder(params)
    o1 = SDR([e1.size])
    e1.encode(65.32, o1)
    print(o1.get_sparse())

    e1 = RandomDistributedScalarEncoder(params)
    o1 = SDR([e1.size])
    e1.encode(73.89, o1)
    print(o1.get_sparse())

    e1 = RandomDistributedScalarEncoder(params)
    o1 = SDR([e1.size])
    e1.encode(73.89, o1)
    print(o1.get_sparse())

    print("Non-threading encoder handler next:")
    encoder1 = EncoderHandler(input_data)
    start_time1 = time.perf_counter()
    sdrs1 = encoder1.build_composite_sdr(input_data)
    end_time1 = time.perf_counter()
    print(f"Encoded {len(sdrs1)} SDRs for RSI column")
    print(f"Encoding took {end_time1 - start_time1:.4f} seconds")
    print("Sample SDRs:")
    for i, sdr in enumerate(sdrs1[:5]):
        print(sdr.get_sparse())


if __name__ == "__main__":
    main()
