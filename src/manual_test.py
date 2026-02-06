"""Manual demo runner for the full HTM pipeline.

This module stitches together the input layer, encoder layer, and HTM brain engine
into a single runtime demo suitable for live presentations.
"""

from __future__ import annotations

from datetime import datetime

from psu_capstone.agent_layer.HTM import ColumnField, InputField
from psu_capstone.agent_layer.brain import Brain
from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.input_layer.input_handler import InputHandler


def build_demo_payload() -> list[dict[str, object]]:
    """Construct a small, presentation-friendly dataset."""
    return [
        {
            "timestamp": datetime(2024, 7, 4, 9, 0),
            "sensor_id": "alpha",
            "temp_c": 21.5,
            "usage_kw": 3,
        },
        {
            "timestamp": datetime(2024, 7, 4, 10, 0),
            "sensor_id": "alpha",
            "temp_c": 22.1,
            "usage_kw": 5,
        },
        {
            "timestamp": datetime(2024, 7, 4, 11, 0),
            "sensor_id": "beta",
            "temp_c": 23.2,
            "usage_kw": 8,
        },
        {
            "timestamp": datetime(2024, 7, 4, 12, 0),
            "sensor_id": "beta",
            "temp_c": 24.0,
            "usage_kw": 13,
        },
    ]


def build_brain(field_sizes: dict[str, int]) -> Brain:
    """Create a Brain wired to four input fields and one column field."""
    temp_field = InputField(
        RDSEParameters(size=field_sizes["temp_c"], active_bits=8, resolution=0.5, seed=11)
    )
    usage_field = InputField(
        RDSEParameters(size=field_sizes["usage_kw"], active_bits=8, resolution=1.0, seed=17)
    )
    sensor_field = InputField(
        RDSEParameters(size=field_sizes["sensor_id"], active_bits=1, category=True, resolution=0, seed=23)
    )
    timestamp_field = InputField(
        RDSEParameters(size=field_sizes["timestamp"], active_bits=8, resolution=3600.0, seed=29)
    )

    column_field = ColumnField(
        input_fields=[temp_field, usage_field, sensor_field, timestamp_field],
        num_columns=128,
        cells_per_column=4,
    )

    return Brain(
        {
            "temp_c": temp_field,
            "usage_kw": usage_field,
            "sensor_id": sensor_field,
            "timestamp": timestamp_field,
            "cortex": column_field,
        }
    )


def normalize_for_brain(records: list[dict[str, object]]) -> list[dict[str, float]]:
    """Convert records into numeric values consumable by RDSE input fields."""
    sensor_ids = sorted({record["sensor_id"] for record in records})
    sensor_lookup = {sensor_id: idx for idx, sensor_id in enumerate(sensor_ids)}

    normalized: list[dict[str, float]] = []
    for record in records:
        timestamp = record["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        normalized.append(
            {
                "timestamp": float(timestamp.timestamp()),
                "sensor_id": float(sensor_lookup[str(record["sensor_id"])]),
                "temp_c": float(record["temp_c"]),
                "usage_kw": float(record["usage_kw"]),
            }
        )
    return normalized


def prepare_encoder_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Ensure datetime values are restored for the encoder layer demo."""
    hydrated: list[dict[str, object]] = []
    for record in records:
        timestamp = record["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        hydrated.append({**record, "timestamp": timestamp})
    return hydrated


def run_demo() -> None:
    """Run the full demo: input layer -> encoder layer -> HTM brain."""
    print("\n=== PSU Capstone Manual Demo ===")

    input_handler = InputHandler()
    raw_payload = build_demo_payload()
    normalized_records = input_handler.input_data(raw_payload)

    print("\nInput Layer Output (normalized records):")
    print(f"Loaded {len(normalized_records)} rows from the input layer.")
    for row in normalized_records:
        print(f"  {row}")

    encoder_records = prepare_encoder_records(normalized_records)
    encoder_handler = EncoderHandler(encoder_records)
    composite_sdrs = encoder_handler.build_composite_sdr(encoder_records)

    print("\nEncoder Layer Output (composite SDR summary):")
    for idx, composite in enumerate(composite_sdrs, start=1):
        print(
            f"  Row {idx}: size={composite.size}, sparsity={composite.get_sparsity():.4f}"
        )

    brain = build_brain(
        {
            "temp_c": 128,
            "usage_kw": 128,
            "sensor_id": 64,
            "timestamp": 128,
        }
    )

    print("\nBrain HTM Engine Output:")
    brain_inputs = normalize_for_brain(normalized_records)
    for step_idx, row in enumerate(brain_inputs, start=1):
        brain.step(row, learn=True)
        prediction = brain.prediction()
        print(f"  Step {step_idx}: input={row} prediction={prediction}")

    print("\nHTM Column Statistics:")
    brain.print_stats()


if __name__ == "__main__":
    run_demo()
