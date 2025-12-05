"""
demo_driver.py
@Alex
@Chris

This script demonstrates the encoding and visualization of various data types using the encoder layer
of the psu-capstone project. It showcases scalar, category, date, and RDSE encoders, builds composite
SDRs for sample and Excel data, and visualizes the resulting SDRs as 2D grids. The script is intended
for demonstration and testing purposes, providing insight into how input data is transformed into SDRs.
"""

import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from psu_capstone.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler

matplotlib.use("TkAgg")  # Ensure Tk backend for window positioning


# Set the path to the Excel file relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easyData.xlsx")


def visualize_sdr_all_rows(
    sdrs: list[SDR], title: str = "Composite SDR – Grid View", row_labels: list[str] = None
):
    """
    Visualize a list of SDRs (one per row) as individual 2D square/rectangular grids.
    Optionally display a label for each row in the plot title.

    Allows interactive navigation between SDRs using left/right arrow keys.
    """

    # Precompute grids and labels
    grids: list[np.ndarray] = []
    labels: list[str] = []

    for idx, sdr in enumerate(sdrs):
        dense = np.array(sdr.get_dense())
        n = dense.size

        # Compute smallest square that fits all bits
        side = int(np.ceil(np.sqrt(n)))

        # Pad if necessary
        padded = np.zeros(side * side, dtype=int)
        padded[:n] = dense

        # Reshape into a 2D grid
        grid = padded.reshape(side, side)
        grids.append(grid)

        label = ""
        if row_labels is not None and idx < len(row_labels):
            label = f" ({row_labels[idx]})"
        labels.append(label)

    cmap = ListedColormap(["white", "blue"])

    # Set up a single figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Try to move window to top-left
    try:
        mng = plt.get_current_fig_manager()
        mng.window.wm_geometry("+0+0")  # type: ignore # TkAgg-specific
    except Exception:
        pass

    class IndexTracker:
        def __init__(self, num_items: int):
            self.idx = 0
            self.num_items = num_items

        def update(self):
            ax.clear()
            ax.imshow(grids[self.idx], cmap=cmap, interpolation="nearest")
            ax.set_title(f"{title} – Row {self.idx}{labels[self.idx]}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            fig.canvas.draw_idle()

        def on_key(self, event):
            if event.key in ("right"):  # next
                self.idx = (self.idx + 1) % self.num_items
                self.update()
            elif event.key in ("left"):  # previous
                self.idx = (self.idx - 1) % self.num_items
                self.update()
            elif event.key in ("escape", "up"):
                plt.close(fig)

    tracker = IndexTracker(len(grids))
    tracker.update()

    # Connect key-press events
    fig.canvas.mpl_connect("key_press_event", tracker.on_key)

    plt.show(block=True)


def build_demo_dataframe() -> pd.DataFrame:
    """
    Build a simple demonstration DataFrame with scalar, category, and datetime columns.

    Returns:
        pd.DataFrame: DataFrame containing sample rows for encoding demonstration.
    """

    scalar_rows = [
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
    return pd.DataFrame(scalar_rows)


def main():
    """
    Main driver function for the demo.

    - Initializes encoders for scalar, category, date, and RDSE types.
    - Encodes sample values and visualizes their SDRs.
    - Loads and preprocesses Excel data, builds composite SDRs for both sample and Excel data.
    - Visualizes composite SDRs for each row, displaying relevant labels.
    - Prints SDR statistics for inspection.
    """

    print("Beginning Demo...")
    ih = InputHandler()

    scalar_encoder = ScalarEncoder(
        ScalarEncoderParameters(
            minimum=0.0,
            maximum=100.0,
            size=100,
            active_bits=10,  # sparisty of 10% for visualization
            clip_input=True,
            periodic=False,
            category=False,
            sparsity=0.0,
            radius=0.0,
            resolution=0.0,
        )
    )

    rdse_encoder = RandomDistributedScalarEncoder(
        RDSEParameters(
            active_bits=10,  # sparisty of 10% for visualization
            size=100,
            sparsity=0.0,
            radius=0.1,  # small radius for demo, all us to see a float encoding
            resolution=0.0,
            category=False,
            seed=42,
        )
    )

    category_encoder = CategoryEncoder(
        CategoryParameters(w=5, category_list=["US", "CA", "MX", "UK", "FR", "DE"])
    )

    date_encoder = DateEncoder(
        DateEncoderParameters(
            season_width=0,
            season_radius=91.5,
            day_of_week_width=7,
            day_of_week_radius=1.0,
            weekend_width=2,
            holiday_width=4,
            holiday_dates=[[12, 25], [1, 1], [7, 4], [11, 11]],
            time_of_day_width=24,
            time_of_day_radius=1.0,
            custom_width=0,
            custom_days=[],
            rdse_used=False,
        )
    )

    required_columns_excel = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "RSI",
        "MACD",
        "MACDState",
        "HAColor",
        "HABodyRangeRatio",
        "HALongWick",
        "HAGreenConsec",
        "HARedConsec",
        "EMAState",
        "HAHighToEMALong",
        "HACloseToEMALong",
        "HALowToEMALong",
        "HAHighToEMAShort",
        "HACloseToEMAShort",
        "HALowToEMAShort",
        "MyWAZLTTrend",
    ]

    scalar_values = [25, 50, 75, 25]
    date_values = [
        datetime(2024, 6, 15, 10, 30),
        datetime(2024, 12, 25, 8, 0),
        datetime(2024, 7, 4, 21, 15),
        datetime(2024, 6, 15, 10, 30),
    ]
    category_values = ["US", "CA", "MX", "US"]  # Added two more category values
    scalar_sdrs = []
    rdse_sdrs = []
    category_sdrs = []
    date_sdrs = []
    for i, value in enumerate(scalar_values):
        sdr1 = SDR([scalar_encoder.size])
        sdr2 = SDR([rdse_encoder.size])
        sdr3 = SDR(category_encoder.dimensions)
        sdr4 = SDR([date_encoder.size])
        scalar_encoder.encode(value, sdr1)
        rdse_encoder.encode(value, sdr2)
        category_encoder.encode(category_values[i], sdr3)
        date_encoder.encode(date_values[i], sdr4)
        scalar_sdrs.append(sdr1)
        rdse_sdrs.append(sdr2)
        category_sdrs.append(sdr3)
        date_sdrs.append(sdr4)

    # Visualize individual SDRs with value in title
    visualize_sdr_all_rows(
        scalar_sdrs, title="Scalar Encoder SDR", row_labels=[str(v) for v in scalar_values]
    )
    visualize_sdr_all_rows(
        rdse_sdrs, title="RDSE Encoder SDR", row_labels=[str(v) for v in scalar_values]
    )
    visualize_sdr_all_rows(category_sdrs, title="Category Encoder SDR", row_labels=category_values)
    visualize_sdr_all_rows(date_sdrs, title="Date Encoder SDR", row_labels=date_values)  # type: ignore

    # Test composite SDR building
    demo_sample_df = build_demo_dataframe()
    sub_set_df = ih.input_data(DATA_PATH, required_columns=required_columns_excel)

    # change values to all float to trigger rdse in sub_set_df
    for col in sub_set_df.columns:
        if sub_set_df[col].dtype == int:
            try:
                sub_set_df[col] = sub_set_df[col].astype(float)
            except ValueError:
                pass  # Ignore columns that cannot be converted to float

    # encoder parameters are hardcoded in EncoderHandler for demo purposes
    handler = EncoderHandler(demo_sample_df)
    composite_list = handler.build_composite_sdr(demo_sample_df)

    print("Composite SDR count:", len(composite_list))
    for idx, composite in enumerate(composite_list):
        print(f"Composite SDR {idx} dimensions:", composite.dimensions)
        print(f"Composite SDR {idx} size:", composite.size)
        print(f"Composite SDR {idx} Sparsity:", composite.get_sparsity())
        print(f"Composite SDR {idx} Density:", composite.get_dense())

    # For demo_sample_df, show all values being encoded for each row, but skip any datetime/Timestamp in the label
    def all_row_label(row, max_len=60):
        vals = list(row.values)
        # Only include non-datetime values

        def is_not_datetime(val):
            return not hasattr(val, "strftime")

        filtered_vals = [str(v) for v in vals if is_not_datetime(v)]
        label = ", ".join(filtered_vals)
        # Insert newlines every max_len characters for readability
        if len(label) > max_len:
            parts = []
            current = ""
            for v in filtered_vals:
                if len(current) + len(v) + 2 > max_len:
                    parts.append(current.rstrip(", "))
                    current = ""
                current += v + ", "
            if current:
                parts.append(current.rstrip(", "))
            label = "\n".join(parts)
        return label

    row_labels = [all_row_label(row) for _, row in demo_sample_df.iterrows()]
    visualize_sdr_all_rows(composite_list, title="Composite SDR", row_labels=row_labels)

    composite_list_excel = handler.build_composite_sdr(sub_set_df)

    print("Composite SDR count (Excel data):", len(composite_list_excel))
    for idx, composite in enumerate(composite_list_excel):
        print(f"Composite SDR {idx} dimensions:", composite.dimensions)
        print(f"Composite SDR {idx} size:", composite.size)
        print(f"Composite SDR {idx} Sparsity:", composite.get_sparsity())

    row_labels_excel = [all_row_label(row) for _, row in sub_set_df.iterrows()]
    visualize_sdr_all_rows(composite_list_excel, title="Composite SDR", row_labels=row_labels_excel)


if __name__ == "__main__":
    main()
