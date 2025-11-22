"""
demo_driver.py

"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler


def visualize_sdr_all_rows(sdr: SDR, title: str = "Composite SDR – Grid View"):
    """
    Visualize a composite SDR (multiple rows) as a single 2D square/rectangular grid.
    """
    # Flatten SDR to 1D dense bit array
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

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, interpolation="nearest")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show(block=True)


def build_demo_dataframe() -> pd.DataFrame:

    rows = [
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
    return pd.DataFrame(rows)


def main():

    ih = InputHandler()

    excel_path = (
        r"C:\Users\alexb\Desktop\SWENG 480-481 Final Project\psu-capstone\data\concat_ESData.xlsx"
    )
    full_df = ih.load_data(excel_path)

    print("Raw DataFrame from Excel:")
    print(full_df.head())

    # pick as many rows as you want here
    demo_df = full_df.iloc[0:10]  # first 10 rows
    # demo_df = full_df                # or ALL rows
    # demo_df = full_df.sample(5)      # or a random 5 rows

    df = ih.to_dataframe(demo_df)

    handler = EncoderHandler(df)
    composite: SDR = handler.build_composite_sdr(df)

    print("Composite SDR dimensions:", composite.dimensions)
    print("Composite SDR size:", composite.size)
    print("Composite Sparsity:", composite.get_sparsity())

    visualize_sdr_all_rows(composite, title="Composite SDR – Rows")


if __name__ == "__main__":
    main()
