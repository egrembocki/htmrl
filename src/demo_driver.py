"""End-to-end HTM demo driver with visualization outputs.

This module builds on ``manual_test`` by running the same input -> encoder -> HTM
pipeline and producing saved visual artifacts for:
- Hot Gym time-wave data
- Composite SDR encodings
- HTM learning dynamics (active columns / active & predictive cells)
- Column-to-input connected synapse map
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "demo"


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


def _save_heatmap(matrix, path: Path, title: str, xlabel: str, ylabel: str) -> None:
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
    plt = _import_pyplot()
    times = [record["timestamp"] for record in records]
    usage = [float(record["kw_energy_consumption"]) for record in records]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, usage, marker="o", linewidth=1.5)
    ax.set_title("Hot Gym Time Wave")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("kw_energy_consumption")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _column_input_connectivity(column_field):
    input_cells = list(column_field.input_field.cells)
    cell_index = {cell: idx for idx, cell in enumerate(input_cells)}
    import numpy as np

    connectivity = np.zeros((len(column_field.columns), len(input_cells)), dtype=int)
    for col_idx, column in enumerate(column_field.columns):
        for syn in column.connected_synapses:
            connectivity[col_idx, cell_index[syn.source_cell]] = 1
    return connectivity


def run_full_demo(row_limit: int = 96, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    """Run the manual-test pipeline end-to-end and emit visualization artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Start with manual_test flow (input normalization + encoder prep + brain preparation)
    from manual_test import (
        REQUIRED_COLUMNS,
        build_brain,
        load_demo_records,
        normalize_for_brain,
        prepare_encoder_records,
    )
    from psu_capstone.agent_layer.HTM import ColumnField
    from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
    from psu_capstone.input_layer.input_handler import InputHandler

    input_handler = InputHandler()
    records = load_demo_records(input_handler, limit=row_limit)
    if not records:
        raise ValueError("No records loaded for demo.")

    encoder_records = prepare_encoder_records(records)
    encoder_handler = EncoderHandler(encoder_records)
    composite_sdrs = encoder_handler.build_composite_sdr(encoder_records)

    brain = build_brain({"kw_energy_consumption": 256, "timestamp": 256})
    brain_inputs = normalize_for_brain(records)
    column_field = brain["cortex"]
    if not isinstance(column_field, ColumnField):
        raise TypeError("Brain cortex field is not a ColumnField.")

    # 2) Collect HTM learning traces
    import numpy as np

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full HTM demo with visualization outputs.")
    parser.add_argument("--rows", type=int, default=96, help="Number of Hot Gym rows to run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG artifacts are saved.",
    )
    args = parser.parse_args()

    outputs = run_full_demo(row_limit=args.rows, output_dir=args.output_dir)
    print("\nGenerated demo visuals:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    print("\nRequired columns: [timestamp, kw_energy_consumption]")


if __name__ == "__main__":
    main()
