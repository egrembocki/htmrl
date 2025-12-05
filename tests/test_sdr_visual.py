"""Visual tests for SDR class."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import ListedColormap

from psu_capstone.encoder_layer.sdr import SDR


@pytest.fixture
def sdr_visualization(debug=False):
    """Fixture for SDR visualization tests."""
    pass  # No setup needed for this simple visualization test


@pytest.mark.visual
def test_sdr_visualization(sdr_visualization):
    """Test the visualization of an SDR."""

    # sdr.get_dense() should return a list or 1D array of 0/1
    sdr = SDR([64, 32])
    sdr.randomize(0.02)
    sdr.add_noise(0.01)
    dense = np.array(sdr.get_dense())

    # compute square grid size
    n = dense.size
    side = int(np.ceil(np.sqrt(n)))  # smallest square that fits SDR

    # pad if needed
    padded = np.zeros(side * side, dtype=int)
    padded[:n] = dense

    grid = padded.reshape(side, side)

    # colormap: white for 0, blue for 1
    cmap = ListedColormap(["white", "blue"])

    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, interpolation="nearest")
    title = "SDR Visualization"
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show(block=True)


@pytest.mark.visual
def test_sdr_one_row_visual():
    """Test visualization of a single row SDR."""
    # Arrange
    sdr = SDR([100])
    sdr.randomize(0.05)

    dense = np.array(sdr.get_dense())
    arr2d = dense.reshape(1, -1)  # one row, N columns

    # ON bits = blue, OFF bits = white
    cmap = ListedColormap(["white", "blue"])

    plt.figure(figsize=(12, 2))
    plt.imshow(arr2d, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.yticks([])  # Remove y-axis
    plt.xlabel("Bit Index")
    plt.title("SDR (1D One-Row Visual)")
    plt.show()


@pytest.mark.visual
def test_sdr_union_visual():
    print(">> running test_sdr_union_layout_visual")  # sanity check

    rows, cols = 20, 50  # size of each small SDR grid

    # --- Create three SDRs (each 16x16) ---
    sdr1 = SDR([rows, cols])
    sdr1.randomize(0.02)

    sdr1.set_sparse_inplace()

    sdr2 = SDR([rows, cols])
    sdr2.randomize(0.05)

    sdr2.set_sparse_inplace()

    sdr3 = SDR([rows, cols])
    sdr3.randomize(0.03)

    sdr3.set_sparse_inplace()

    # --- Union SDR big enough to hold all three stacked vertically ---
    # shape: (rows * 3, cols)
    sdr_union = SDR([rows * 3, cols])
    sdr_union.concatenate([sdr1, sdr2, sdr3], axis=0)

    # --- Convert SDRs to 2D dense numpy arrays ---
    grid1 = np.array(sdr1.get_dense(), dtype=int).reshape(rows, cols)
    grid2 = np.array(sdr2.get_dense(), dtype=int).reshape(rows, cols)
    grid3 = np.array(sdr3.get_dense(), dtype=int).reshape(rows, cols)

    union_grid = np.array(sdr_union.get_dense(), dtype=int).reshape(rows * 3, cols)

    # --- Colormap: 0 -> white, 1 -> blue ---
    cmap = ListedColormap(["white", "#1f77b4"])

    # --- Figure layout that matches your screenshot ---
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax_union = fig.add_subplot(gs[1, 0])

    ax1.imshow(grid1, cmap=cmap, interpolation="nearest")
    ax1.set_title("SDR One")
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.imshow(grid2, cmap=cmap, interpolation="nearest")
    ax2.set_title("SDR Two")
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.imshow(grid3, cmap=cmap, interpolation="nearest")
    ax3.set_title("SDR Three")
    ax3.set_xticks([])
    ax3.set_yticks([])

    ax_union.imshow(union_grid, cmap=cmap, interpolation="nearest", aspect="auto")
    ax_union.set_title("Union")
    ax_union.set_xticks([])
    ax_union.set_yticks([])

    plt.tight_layout()
    plt.show(block=True)

    print(">> finished test_sdr_union_layout_visual")
