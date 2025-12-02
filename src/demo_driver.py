"""
demo_driver.py

"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from matplotlib.colors import ListedColormap

from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler

# Set the path to the Excel file relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easyData.xlsx")


def visualize_sdr_all_rows(sdrs: list[SDR], title: str = "Composite SDR – Grid View"):
    """
    Visualize a list of SDRs (one per row) as individual 2D square/rectangular grids.
    """
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

        # Colormap: white (0) and blue (1)
        cmap = ListedColormap(["white", "blue"])

        plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap=cmap, interpolation="nearest")
        plt.title(f"{title} – Row {idx}")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.show(block=True)


def build_demo_dataframe() -> pd.DataFrame:

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
            "country": "US",
            "timestamp": datetime(2015, 3, 25, 8, 30),
        },
    ]
    return pd.DataFrame(scalar_rows)


def main():

    ih = InputHandler()

    required_columns_manual = ["temp_c", "visits", "country", "timestamp"]

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

    sub_set_df = ih.input_data(DATA_PATH, required_columns=required_columns_excel)

    print("Raw DataFrame first ten rows from Excel:")

    print(sub_set_df.head())

    handler = EncoderHandler(sub_set_df)
    composite_list = handler.build_composite_sdr(sub_set_df)

    print("Composite SDR count:", len(composite_list))
    for idx, composite in enumerate(composite_list):
        print(f"Composite SDR {idx} dimensions:", composite.dimensions)
        print(f"Composite SDR {idx} size:", composite.size)
        print(f"Composite SDR {idx} Sparsity:", composite.get_sparsity())

    visualize_sdr_all_rows(composite_list, title="Composite SDR – Rows")


if __name__ == "__main__":
    main()
