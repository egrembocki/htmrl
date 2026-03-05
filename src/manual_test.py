"""Manual demo runner for the full HTM pipeline.

This module stitches together the input layer, encoder layer, and HTM brain engine
into a single runtime demo suitable for live presentations.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.HTM import ColumnField, InputField
from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.input_layer.input_handler import InputHandler

DEMO_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "hot_gym_short.csv"
REQUIRED_COLUMNS = ["timestamp", "consumption"]
DEMO_ROW_LIMIT = 8
SEQUENCE_SAMPLE = 5


def load_demo_records(handler: InputHandler, limit: int | None = None) -> list[dict[str, object]]:
    """Load and normalize demo data via the InputHandler singleton."""

    handler.input_data(DEMO_DATA_FILE, required_columns=REQUIRED_COLUMNS)
    if limit is not None and limit <= 0:
        return []
    return handler.to_records(limit=limit)


def sample_usage_sequence(handler: InputHandler, sample_size: int) -> list[float]:
    """Use InputHandler helper output to retrieve a single-column encoder sequence sample."""

    if sample_size <= 0:
        return []

    sequence = handler.to_encoder_sequence(
        handler.data,
        required_columns=REQUIRED_COLUMNS,
        column="consumption",
    )
    values = [float(value) for value in sequence["consumption"]]
    return values[:sample_size]


def build_brain(field_sizes: dict[str, int]) -> Brain:
    """Create an HTM Brain wired to timestamp and energy usage input fields."""

    usage_field = InputField(
        RDSEParameters(
            size=field_sizes["consumption"],
            active_bits=16,
            sparsity=0.0,
            resolution=0.5,
            seed=11,
        )
    )
    timestamp_field = InputField(
        RDSEParameters(
            size=field_sizes["timestamp"],
            active_bits=16,
            sparsity=0.0,
            resolution=3600.0,
            seed=17,
        )
    )

    column_field = ColumnField(
        input_fields=[usage_field, timestamp_field],
        num_columns=128,
        cells_per_column=4,
    )

    return Brain(
        {
            "consumption": usage_field,
            "timestamp": timestamp_field,
            "cortex": column_field,
        }
    )


def normalize_for_brain(records: list[dict[str, object]]) -> list[dict[str, float]]:
    """Convert records into numeric values consumable by RDSE input fields."""

    normalized: list[dict[str, float]] = []
    for record in records:
        timestamp = record["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        usage = record["consumption"]
        normalized.append(
            {
                "timestamp": float(timestamp.timestamp()),
                "consumption": float(usage),
            }
        )

    return normalized


def prepare_encoder_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Ensure the encoder layer receives datetime timestamps and numeric values."""

    hydrated: list[dict[str, object]] = []
    for record in records:
        timestamp = record["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        usage = float(record["consumption"])
        hydrated.append(
            {
                "timestamp": timestamp,
                "consumption": usage,
            }
        )

    return hydrated


def run_demo() -> None:
    """Run the full demo: input layer -> encoder layer -> HTM brain."""
    print("\n=== PSU Capstone Manual Demo ===")

    input_handler = InputHandler()
    normalized_records = load_demo_records(input_handler, limit=DEMO_ROW_LIMIT)

    print("\nInput Layer Output (normalized records):")
    print(f"Loaded {len(normalized_records)} rows from the input layer.")
    for row in normalized_records:
        print(f"  {row}")

    usage_sequence = sample_usage_sequence(
        input_handler, min(SEQUENCE_SAMPLE, len(normalized_records))
    )
    print("\nInput Handler sequence sample (consumption):")
    print(f"  {usage_sequence}")

    encoder_records = prepare_encoder_records(normalized_records)
    encoder_handler = EncoderHandler(encoder_records)
    composite_sdrs = encoder_handler.build_composite_sdr(encoder_records)

    print("\nEncoder Layer Output (composite SDR summary):")
    for idx, composite in enumerate(composite_sdrs, start=1):
        print(f"  Row {idx}: size={composite.size}, sparsity={composite.get_sparsity():.4f}")

    brain = build_brain(
        {
            "consumption": 256,
            "timestamp": 256,
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
