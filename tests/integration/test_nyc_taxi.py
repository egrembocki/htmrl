import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics

from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.HTM import ColumnField, InputField
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoderParameters
from psu_capstone.input_layer.input_handler import InputHandler


def build_dataset(csv_path):
    ih = InputHandler()
    data = ih.input_data(csv_path)

    return {
        "passenger_count_input": [float(v) for v in data["passenger_count"]],
        "timeofday_input": [int(v) for v in data["timeofday"]],
        "dayofweek_input": [int(v) for v in data["dayofweek"]],
        "timestamp": pd.to_datetime(data["timestamp"]),
    }


def build_brain(encoder_type):
    if encoder_type == "rdse":
        passenger_input = InputField(
            size=2048,
            encoder_params=RDSEParameters(
                size=1024,
                sparsity=0.4,
                resolution=1.0,
                category=False,
                seed=42,
            ),
        )

        timeofday_input = InputField(
            size=2048,
            encoder_params=RDSEParameters(
                size=2048,
                sparsity=0.2,
                resolution=1.0,
                category=False,
                seed=43,
            ),
        )

        dayofweek_input = InputField(
            size=2048,
            encoder_params=RDSEParameters(
                size=2048,
                sparsity=0.5,
                resolution=1.0,
                category=False,
                seed=44,
            ),
        )

    elif encoder_type == "scalar":
        passenger_input = InputField(
            size=2048,
            encoder_params=ScalarEncoderParameters(
                size=2048,
                minimum=0,
                maximum=26000,
                periodic=False,
                category=False,
                sparsity=0.2,
                resolution=1.0,
                active_bits=0,
                radius=0,
            ),
        )

        timeofday_input = InputField(
            size=2048,
            encoder_params=ScalarEncoderParameters(
                size=2048,
                minimum=0,
                maximum=23,
                periodic=True,
                category=False,
                sparsity=0.2,
                resolution=1.0,
                active_bits=0,
                radius=0,
            ),
        )

        dayofweek_input = InputField(
            size=2048,
            encoder_params=ScalarEncoderParameters(
                size=2048,
                minimum=0,
                maximum=6,
                periodic=False,
                category=True,
                sparsity=0.2,
                resolution=1.0,
                active_bits=0,
                radius=0,
            ),
        )
    else:
        raise ValueError("Need a correct encoder type.")

    inputfieldlist = [passenger_input, timeofday_input, dayofweek_input]

    cf = ColumnField(inputfieldlist, num_columns=2048, cells_per_column=50)
    brain = Brain(
        {
            "passenger_count_input": passenger_input,
            "timeofday_input": timeofday_input,
            "dayofweek_input": dayofweek_input,
            "column_field": cf,
        }
    )
    return brain


def run_experiment(
    dataset,
    train_steps,
    learn,
    start_idx=0,
    end_idx=None,
    encoder_type="rdse",
    window=960,
):
    passenger = dataset["passenger_count_input"]
    timeofday = dataset["timeofday_input"]
    dayofweek = dataset["dayofweek_input"]
    timestamps = dataset["timestamp"]

    if end_idx is None:
        end_idx = len(passenger)

    passenger = passenger[start_idx:end_idx]
    timeofday = timeofday[start_idx:end_idx]
    dayofweek = dayofweek[start_idx:end_idx]
    timestamps = timestamps[start_idx:end_idx]

    brain = build_brain(encoder_type)
    n = len(passenger)

    if n == 0:
        raise ValueError("Dataset is empty.")

    train_steps = min(train_steps, n)

    predictions_pass = [np.nan] * n
    mape_curve = [np.nan] * n

    eval_actuals = []
    eval_predictions = []

    for i in range(n):
        brain.step(
            {
                "passenger_count_input": passenger[i],
                "timeofday_input": timeofday[i],
                "dayofweek_input": dayofweek[i],
            },
            learn=(learn and i < train_steps),
        )

        preds = brain.prediction()
        pred_pass = preds["passenger_count_input"]
        predictions_pass[i] = pred_pass

        if i >= train_steps:
            eval_actuals.append(passenger[i])
            eval_predictions.append(pred_pass)

            if len(eval_actuals) >= window:
                mape_curve[i] = metrics.mean_absolute_percentage_error(
                    eval_actuals[-window:],
                    eval_predictions[-window:],
                )

    first_mape_idx = min(train_steps + window - 1, n - 1)

    return {
        "timestamps": timestamps,
        "predictions_passenger": predictions_pass,
        "mape_curve": mape_curve,
        "train_steps": train_steps,
        "first_mape_idx": first_mape_idx,
        "window": window,
    }


def test_taxi_dataset():
    plt.close("all")

    csv_path = "data/nyc_taxi.csv"
    dataset = build_dataset(csv_path)

    sp_learning = run_experiment(
        dataset,
        learn=True,
        train_steps=1000,
        start_idx=0,
        end_idx=2000,
        encoder_type="rdse",
        window=400,
    )

    print("SP Learning result:", sp_learning)

    first_valid_idx = sp_learning["first_mape_idx"]

    timestamps = sp_learning["timestamps"][first_valid_idx:]
    mape = sp_learning["mape_curve"][first_valid_idx:]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(timestamps, mape, label="SP Learning")

    ax1.set_ylabel("Mean Absolute Percent Error (MAPE)")
    ax1.set_xlabel("Timestamp")
    ax1.legend()

    plt.tight_layout()
    plt.show()
