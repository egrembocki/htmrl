import time
import warnings

import numpy as np
import pandas as pd

from psu_capstone.encoder_layer.batch_encoder_handler import BatchEncoderHandler
from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler

warnings.simplefilter(action="ignore", category=FutureWarning)


def main():
    handler = InputHandler.get_instance()
    # skipping the timestamp req in input handler
    # handler._prepend_timestamp_column = lambda df: df

    excel_file_path = r"C:\Users\Josh\Downloads\rsi_data\small_data.xlsx"

    input_data = handler.input_data(input_source=excel_file_path, required_columns=[])
    input_data = input_data.loc[:, ~input_data.columns.duplicated()]

    print(input_data)

    # Composite sdr
    start_time = time.perf_counter()
    encoder = BatchEncoderHandler(input_data)
    sdrs = encoder.build_composite_sdr(input_data, 8)
    end_time = time.perf_counter()
    print(f"Encoded {len(sdrs)} SDRs for all columns")
    print(f"Encoding took {end_time - start_time:.4f} seconds")
    print("Sample SDRs:")
    for i, sdr in enumerate(sdrs[:1]):  # first 5 SDRs
        print(f"Composite SDR index: {i}")
        print(f"Sparse indices: {sdr.get_sparse()}")
        print(f"Length of SDR: {len(sdr.get_dense())}\n")

    # Tests the dict list of column and sdr.

    """encoder = BatchEncoderHandler(input_data)
    start_time = time.perf_counter()
    sdrs = encoder._build_dict_list_sdr(input_data, threads_per_column=8)
    end_time = time.perf_counter()
    print(f"Encoded {len(sdrs)} SDRs for all columns")
    print(f"Encoding took {end_time - start_time:.4f} seconds")
    print("Sample SDRs:")
    for col, sdr_list in sdrs.items():
        print(f"Column: {col}")
        for i, sdr in enumerate(sdr_list[:1]):
            print(sdr.get_sparse())
            print(f"Length of sdr: {len(sdr.get_dense())} ")
            print("\n")"""

    # Tests the composite sdr where 1 sdr exists per row.

    """ encoder2 = BatchEncoderHandler(input_data)
    composite_sdrs = encoder2.build_composite_sdr(sdrs, threads_per_column=8)
    for i, sdr in enumerate(composite_sdrs[:1]):
        print(f"Row {i} composite SDR sparse indices: {sdr.get_sparse()}")
        print(f"Length of sdr: {len(sdr.get_dense())} ")
    """
    """
    #This test was to double check that rdse seed works

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
    print(o1.get_sparse())"""

    # Tested the speed on 1 column of data of the other encoder handler
    # tested here on full dataset and it took over 18 minutes, this is on 4.5 billion bits roughly
    print("Non-threading encoder handler next:")
    start_time1 = time.perf_counter()
    encoder1 = EncoderHandler(input_data)
    sdrs1 = encoder1.build_composite_sdr(input_data)
    end_time1 = time.perf_counter()
    print(f"Encoded {len(sdrs1)} SDRs for RSI column")
    print(f"Encoding took {end_time1 - start_time1:.4f} seconds")
    print("Sample SDRs:")
    for i, sdr in enumerate(sdrs1[:1]):
        print(sdr.get_sparse())
        print(f"Length of SDR: {len(sdr.get_dense())}\n")


if __name__ == "__main__":
    main()
