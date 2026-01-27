import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from psu_capstone.agent_layer.htm.spatial_pooler import SpatialPooler
from psu_capstone.agent_layer.htm.temporal_memory import TemporalMemory
from psu_capstone.encoder_layer.batch_encoder_handler import BatchEncoderHandler
from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.input_layer.improved_input_handler import InputHandler
from psu_capstone.sdr_layer.sdr import SDR

warnings.simplefilter(action="ignore", category=FutureWarning)

"""
    Why do I have so much here? A lot of this can be turned into integration tests and or unit tests.
    I am keeping this for future test writing.
"""


def main():
    handler = InputHandler.get_instance()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "rec-center-hourly.csv")

    print(data_path)

    input_data = handler.input_data(input_source=data_path, required_columns=[])
    input_data = input_data.loc[:, ~input_data.columns.duplicated()]

    encoder = RandomDistributedScalarEncoder()
    values = pd.to_numeric(input_data["kw_energy_consumption"], errors="coerce").dropna().tolist()
    train_values, test_values = train_test_split(values, test_size=0.2, shuffle=False)
    for v in train_values:
        encoder.encode(v)

    """
    print("__value_____predicted___error")
    y_true = []
    y_pred = []

    for v in test_values:
        test_sdr = SDR([encoder.size])
        encoder.encode(v, test_sdr)
        pred = encoder.decode(test_sdr)
        y_true.append(v)
        y_pred.append(pred)
        error = pred - v
        print(f"{v:7.3f}   {pred:9.3f}   {error:+7.3f}")
    """
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print("RMSE:", rmse)

    print(input_data)

    encoder = BatchEncoderHandler(input_data)
    sdrs = encoder.build_composite_sdr(input_data, 8)
    for sdr in sdrs:
        print(sdr.get_sparse())
    #sdrs, knn = encoder.create_knn_composite_sdr(input_data, 8)
    """
    """
    # Composite sdr
    start_time = time.perf_counter()
    encoder = BatchEncoderHandler(input_data)
    # encoder.choose_custom_column_encoding(dict["Wave", "category"]) #force the numerical to be encoded as category test
    sdrs = encoder.build_composite_sdr(input_data, 8)
    end_time = time.perf_counter()
    print(f"Encoded {len(sdrs)} SDRs for all columns")
    print(f"Encoding took {end_time - start_time:.4f} seconds")
    print("Sample SDRs:")
    for i, sdr in enumerate(sdrs[:5]):  # first 5 SDRs
        print(f"Composite SDR index: {i}")
        print(f"Sparse indices: {sdr.get_sparse()}")
        print(f"Length of SDR: {len(sdr.get_dense())}\n")
    """
    """
    # test on htm proposed flow
    sdrs = encoder._sdrs_encoded
    sdr0 = encoder._sdrs_encoded[0]     #sdrs[0]
    dense_input = np.asarray(sdr0, dtype=np.int8)
    input_vector = [dense_input]
    size = len(sdr0)
    sp = SpatialPooler(size, 40, 100)
    mask, active_cols = sp.compute_active_columns(input_vector, inhibition_radius=10)
    print(mask)
    print(active_cols)
    sp.learning_phase(active_cols, dense_input)
    tm = TemporalMemory(sp.columns, 5)
    tm_out = tm.step(active_cols)
    active_cells = tm_out["active_cells"]
    predictive_cells = tm_out["predictive_cells"]
    learning_cells = tm_out["learning_cells"]
    #print("Active cells: ", active_cells)
    #print("Predictive cells: ", predictive_cells)
    #print("Learning cells: ", learning_cells)
    predicted_columns_mask = tm.get_predictive_columns_mask()
    print(predicted_columns_mask)

    # test on multiple steps
    num_t = 1930
    tm_outputs = []
    tm_prediction_masks = []
    sdr0 = sdrs[0]
    size = len(sdr0)
    sp = SpatialPooler(size, 200, 50)
    cells_per_column = 5
    tm = TemporalMemory(columns=sp.columns, cells_per_column=cells_per_column)

    tm_prediction_masks = [None] * len(sdrs)

    for t in range(num_t):
        idx = t % len(sdrs)
        sdr = sdrs[idx]
        dense_input = np.asarray(sdr, dtype=np.int8)
        mask, active_cols = sp.compute_active_columns([dense_input], inhibition_radius=10)
        tm_out = tm.step(active_cols)
        m = tm.get_predictive_columns_mask()
        tm_prediction_masks[idx] = m
        tm_outputs.append(tm_out)

    knn = encoder.build_knn_from_tm_predictions(
        tm_prediction_masks, input_data, n_neighbors=1, weights="distance", distance="hamming"
    )
    # try out the hot gym
    predictions = []
    actual_values = []
    for t in range(193):
        idx = t % len(sdrs)
        sdr = sdrs[idx]
        dense_input = np.asarray(sdr, dtype=np.int8)
        mask, active_cols = sp.compute_active_columns([dense_input], inhibition_radius=10)

        tm.step(active_cols)
        m = tm.get_predictive_columns_mask()
        p = knn.predict(m.reshape(1, -1))[0][0]

        predictions.append(p)
        next_idx = (idx + 1) % len(sdrs)
        target_col = input_data.columns[1]
        actual_values.append(input_data.loc[input_data.index[next_idx], target_col])

    for pred, actual in zip(predictions, actual_values):
        print(f"Actual next step: {actual}, Prediction: {pred}")
    """
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


"""
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
"""

if __name__ == "__main__":
    main()
