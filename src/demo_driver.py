"""Driver script to demonstrate the capabilities of the Brain and Trainer classes.
Author: Chris Mills @millscb

Date: 2025-02-21

This script includes several demonstration functions that can be run to visualize and understand different components of the Brain and Trainer classes. Each function focuses on a specific aspect, such as data ingestion, encoding, brain structure, and training on datasets. The main function at the bottom can be modified to call any of these demonstration functions as needed.

Note: Some of the demonstration functions may require specific datasets to be available in the DATA_PATH directory. Ensure that the necessary datasets are in place before running those demos.

"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import grapher
from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.train import Trainer
from psu_capstone.encoder_layer.base_encoder import ParentDataClass
from psu_capstone.encoder_layer.category_encoder import CategoryParameters
from psu_capstone.encoder_layer.date_encoder import DateEncoderParameters
from psu_capstone.encoder_layer.fourier_encoder import FourierEncoderParameters
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.log import logger
from utils import DATA_PATH, PROJECT_ROOT, hamming_distance, overlap

ESD = os.path.join(DATA_PATH, "concat_ESData.xlsx")
REC_CENTER = os.path.join(DATA_PATH, "rec_center.csv")
DATA_COLUMN_LOG_MESSAGE = "Data column '%s': %d records"
DEFAULT_OUTPUT_DIR = Path(PROJECT_ROOT) / "artifacts" / "demo"

DATA_DICT = {}


def build_demo_records() -> list[dict[str, Any]]:
    """Keep a compact sample dataset used by integration tests."""

    return [
        {
            "temp_c": 21.5,
            "visits": 3,
            "country": "US",
            "timestamp": datetime(2023, 12, 25, 8, 30),
        },
        {
            "temp_c": 4.5,
            "visits": 12,
            "country": "MX",
            "timestamp": datetime(2015, 3, 25, 8, 30),
        },
        {
            "temp_c": 15.0,
            "visits": 7,
            "country": "CA",
            "timestamp": datetime(2022, 7, 4, 14, 0),
        },
        {
            "temp_c": 21.5,
            "visits": 3,
            "country": "US",
            "timestamp": datetime(2023, 12, 25, 8, 30),
        },
    ]


def show_input_data_demo() -> None:
    """Demonstrate the input handler can digest large datasets.
    Use Case Input Data --> Validate Data
    """

    ih = InputHandler()
    data = ih.input_data(ESD)

    for name, value in data.items():
        logger.info(DATA_COLUMN_LOG_MESSAGE, name, len(value))


def show_input_to_encoder_demo(s: int = 5) -> None:
    """Demonstrate the InputHandler's ability to convert raw input data to encoder-ready format."""

    ih = InputHandler()
    brain = Brain()
    trainer = Trainer(brain)

    columns: list[str] = []
    data = ih.input_data(ESD)

    for key in data.keys():
        logger.debug(DATA_COLUMN_LOG_MESSAGE, key, len(data[key]))

        columns.append(key)

    column_data = ih.get_column_data(column=columns[1])

    trainer.main_brain = trainer.build_brain(
        [(f"{columns[1]}_input", 2048, RDSEParameters(resolution=0.01))]
    )  # fin markets

    values = column_data[:s]  # Take the first 's' values for demonstration

    field = trainer.main_brain._input_fields[f"{columns[1]}_input"]
    encoder = field.encoder

    sdr = []

    for i, value in enumerate(values):
        encoded: list[int] = field.encode(value)
        decoded_value, confidence = field.decode("active", field, encoder._encoding_cache)  # type: ignore
        print(f"Decoded: {decoded_value} (Confidence: {confidence})")
        grapher.plot_sdr(encoded, title=f"RDSE Encoding: {columns[1]}={value:.2f} (SDR {i + 1})")

        sdr.append(encoded)

    for i, encoded in enumerate(sdr):
        print(
            f"Hamming distance from first SDR: {hamming_distance(sdr[0], encoded)} (Step {i + 1})"
        )
        print(f"Overlap with first SDR: {overlap(sdr[0], encoded)} (Step {i + 1})")


def show_brain_creation_demo() -> None:
    """Demonstrate creating a Brain and inspecting its structure.
    Use Case Input Parameters
    """

    brain = Brain()
    trainer = Trainer(brain)

    input_fields: list[tuple[str, int, ParentDataClass]] = []

    # Example of building a brain with specific input fields
    input_fields = [
        ("temperature_input", 2048, RDSEParameters()),
        ("humidity_input", 2048, RDSEParameters()),
        ("energy_consumption_input", 2048, RDSEParameters()),
        ("date_input", 2048, DateEncoderParameters()),
        ("category_input", 2048, CategoryParameters()),
        ("wave_input", 2048, FourierEncoderParameters()),
    ]

    trainer.main_brain = trainer.build_brain(input_fields)

    print("Trainer Fields:")
    for field in trainer._trainer_input_fields:
        print(f"- {field.name}")

    print("Brains: Input Fields:")
    for field in trainer.main_brain._input_fields.values():
        print(f"- {field.name}")
    print("Brains: Column Fields:")
    for field in trainer.main_brain._column_fields.values():
        print(f"- {field.name}")


def sine_wave_demo(steps: int = 100) -> None:
    """Demonstrate encoding and learning on a simple sine wave dataset.

    Use Case Train Model -> Train Brain -> Use HTM -> Encode SDR / Decode SDR

    """

    import numpy as np

    # Generate a sine wave dataset
    x = np.linspace(0, 1, 2048, endpoint=False)
    y = np.sin(2 * np.pi * 1 * x)
    data = {"sine_wave_input": y}
    brain = Brain()
    trainer = Trainer(brain)
    trainer.main_brain = trainer.build_brain(
        [("sine_wave_input", 2048, RDSEParameters(resolution=0.001))]
    )
    brain = trainer.main_brain

    for name, value in data.items():
        logger.debug(DATA_COLUMN_LOG_MESSAGE, name, len(value))

    column = {"sine_wave_input": y.tolist()}

    trainer.train_column(brain, column, steps)

    # show predicted vs actual values for the last 100 steps
    test_results = trainer.test(brain, column, steps)

    trainer.show_active_columns(brain, dataset_name="sine wave")
    trainer.show_heat_map(brain, dataset_name="sine wave")

    report_path = os.path.join(PROJECT_ROOT, "docs/reports/sine_wave_training_stats.txt")
    trainer.print_train_stats(report_path, test_results=test_results, training_steps=steps)


def fin_data_demo(column: str | None = None, steps: int = 100) -> None:
    """Demonstrate loading and visualizing data from the dataset.

    Use Cace Input Data -> Validate Data -> Input Parameters -> Train Model -> Train Brain -> Use HTM -> Encode/Decode SDR
    """

    ih = InputHandler()
    data = ih.input_data(ESD)

    brain = Brain()
    trainer = Trainer(brain)
    # resolution changes the spatial pooler representation
    trainer.main_brain = trainer.build_full_brain(data, 2048, RDSEParameters(resolution=0.01))
    brain = trainer.main_brain

    if not column:
        for name, value in data.items():
            logger.debug(DATA_COLUMN_LOG_MESSAGE, name, len(value))

            trainer.train_column(brain, column={f"{name}_input": value}, steps=steps)

    elif column not in data:
        logger.error("Specified column '%s' not found in dataset.", column)
        return
    else:
        trainer.train_column(brain, column={f"{column}_input": data[column]}, steps=steps)

    brain.print_stats()

    # trainer.test(trainer._main_brain, {f"{column}_input": data[column]}, steps=steps)

    trainer.show_active_columns(brain, dataset_name="financial data")
    trainer.show_heat_map(brain, dataset_name="financial data")


def rec_center_demo(steps: int = 100) -> None:
    """Demonstrate loading and visualizing the rec_center dataset."""

    columns: list[str] = []
    ih = InputHandler()
    brain = Brain()
    trainer = Trainer(brain)
    data = ih.input_data(REC_CENTER)

    for key in data.keys():
        logger.debug(DATA_COLUMN_LOG_MESSAGE, key, len(data[key]))

        columns.append(key)

    trainer.main_brain = trainer.build_brain(
        [
            (f"{columns[0]}_input", 2048, DateEncoderParameters()),
            (f"{columns[1]}_input", 2048, RDSEParameters(resolution=0.1)),
        ]
    )

    brain = trainer.main_brain

    for field in brain._input_fields.values():
        logger.debug(
            f"Brain input field: {field.name} with encoder: {type(field.encoder).__name__}"
        )

        trainer.train_column(
            brain, column={field.name: data[field.name.replace("_input", "")]}, steps=steps
        )

    brain.print_stats()
    trainer.show_active_columns(brain, dataset_name="recreation center data")
    trainer.show_heat_map(brain, dataset_name="recreation center data")


def show_field_single_encoding_demo() -> None:
    """Demonstrate encoding a sample input through the Brain's input fields."""

    brain = Brain()
    trainer = Trainer(brain)

    input_fields: list[tuple[str, int, ParentDataClass]] = []
    # Create a simple brain with one input field and one column field
    input_fields = [
        ("temperature_input", 2048, RDSEParameters(resolution=0.1)),
    ]
    trainer.main_brain = trainer.build_brain(input_fields)

    # Example input value to encode
    sample_input = {"temperature_input": 25.8}

    # Step the brain with the sample input
    output = trainer.main_brain._input_fields["temperature_input"].encode(
        sample_input["temperature_input"]
    )

    grapher.plot_sdr(
        output, title=f"RDSE Encoding: temperature_input={sample_input['temperature_input']}°C"
    )


def _import_pyplot():
    """Import matplotlib lazily so non-visual tests can import this module."""

    try:
        import matplotlib

        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to render demo visuals.") from exc
    return plt


def _save_heatmap(matrix: Any, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    """Save a heatmap visualization to a file."""
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(14, 6))
    image = ax.imshow(matrix, interpolation="nearest", aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_time_wave(records: list[dict[str, object]], path: Path) -> None:
    """Save a time-wave plot of consumption over time."""
    plt = _import_pyplot()
    times = [record["timestamp"] for record in records]
    usage = [float(record["consumption"]) for record in records]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, usage, marker="o", linewidth=1.5)
    ax.set_title("Hot Gym Time Wave")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("consumption")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _column_input_connectivity(column_field: Any) -> Any:
    """Build connectivity matrix showing column connections to input cells."""
    import numpy as np

    input_cells = list(column_field.input_field.cells)
    cell_index = {cell: idx for idx, cell in enumerate(input_cells)}

    connectivity = np.zeros((len(column_field.columns), len(input_cells)), dtype=int)
    for col_idx, column in enumerate(column_field.columns):
        for syn in column.connected_synapses:
            connectivity[col_idx, cell_index[syn.source_cell]] = 1
    return connectivity


def run_full_demo(row_limit: int = 96, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Run the manual-test pipeline end-to-end and emit visualization artifacts.

    Args:
        row_limit: Number of Hot Gym records to run through the pipeline
        output_dir: Directory where PNG artifacts will be saved

    Returns:
        Dictionary mapping artifact names to their file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Start with manual_test flow (input normalization + encoder prep + brain preparation)
    import numpy as np

    from manual_test import (
        build_brain,
        load_demo_records,
        normalize_for_brain,
        prepare_encoder_records,
    )
    from psu_capstone.agent_layer.HTM import ColumnField
    from psu_capstone.encoder_layer.encoder_handler import EncoderHandler

    input_handler = InputHandler()
    records = load_demo_records(input_handler, limit=row_limit)
    if not records:
        raise ValueError("No records loaded for demo.")

    encoder_records = prepare_encoder_records(records)
    encoder_handler = EncoderHandler(encoder_records)
    composite_sdrs = encoder_handler.build_composite_sdr(encoder_records)

    brain = build_brain({"consumption": 256, "timestamp": 256})
    brain_inputs = normalize_for_brain(records)
    column_field = brain["cortex"]
    if not isinstance(column_field, ColumnField):
        raise TypeError("Brain cortex field is not a ColumnField.")

    # 2) Collect HTM learning traces
    active_columns = np.zeros((len(brain_inputs), len(column_field.columns)), dtype=int)
    active_cells = np.zeros((len(brain_inputs), len(column_field.cells)), dtype=int)
    predictive_cells = np.zeros((len(brain_inputs), len(column_field.cells)), dtype=int)

    for step_idx, row in enumerate(brain_inputs):
        brain.step(row, learn=True)
        for col_idx, column in enumerate(column_field.columns):
            active_columns[step_idx, col_idx] = int(column.active)
        for cell_idx, cell in enumerate(column_field.cells):
            active_cells[step_idx, cell_idx] = int(cell.active)
            predictive_cells[step_idx, cell_idx] = int(cell.predictive)

    # 3) Build SDR matrix for encoder outputs
    sdr_matrix = np.array([sdr.get_dense() for sdr in composite_sdrs], dtype=int)

    # 4) Save visuals
    outputs = {
        "time_wave": output_dir / "hot_gym_time_wave.png",
        "encoder_sdr": output_dir / "encoder_composite_sdr.png",
        "htm_active_columns": output_dir / "htm_active_columns.png",
        "htm_active_cells": output_dir / "htm_active_cells.png",
        "htm_predictive_cells": output_dir / "htm_predictive_cells.png",
        "column_input_map": output_dir / "htm_column_input_connectivity.png",
    }

    _save_time_wave(records, outputs["time_wave"])
    _save_heatmap(
        sdr_matrix,
        outputs["encoder_sdr"],
        title="Composite SDR Encodings (Rows x Bits)",
        xlabel="SDR Bit Index",
        ylabel="Row Index",
    )
    _save_heatmap(
        active_columns,
        outputs["htm_active_columns"],
        title="HTM Active Columns Over Time",
        xlabel="Column Index",
        ylabel="Step",
    )
    _save_heatmap(
        active_cells,
        outputs["htm_active_cells"],
        title="HTM Active Cells Over Time",
        xlabel="Cell Index",
        ylabel="Step",
    )
    _save_heatmap(
        predictive_cells,
        outputs["htm_predictive_cells"],
        title="HTM Predictive Cells Over Time",
        xlabel="Cell Index",
        ylabel="Step",
    )

    connectivity = _column_input_connectivity(column_field)
    _save_heatmap(
        connectivity,
        outputs["column_input_map"],
        title="Column Connected Synapses to Input Space",
        xlabel="Input Bit Index",
        ylabel="Column Index",
    )

    return outputs


if __name__ == "__main__":
    # Ensure we're in the project root directory for consistent file operations
    os.chdir(PROJECT_ROOT)

    # Example usage of the Brain and Trainer classes

    # show_input_data_demo()
    # show_input_to_encoder_demo(3)
    show_brain_creation_demo()
    # sine_wave_demo(200)
    # fin_data_demo(steps=3)

    # rec_center_demo(200)
    # show_field_single_encoding_demo()
