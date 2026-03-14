"""Visualization utilities for SDRs and encoder analysis.

This module provides plotting functions for visualizing sparse distributed
representations, FFT analysis of time-series data, and encoder behaviors.
Includes tools for comparing encodings, analyzing frequency spectra, and
displaying SDR patterns as 2D grids.
"""

import os
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import ListedColormap, PowerNorm
from scipy.fft import fft, fftfreq, ifft

# Use layer-level imports here because this module crosses package boundaries often
# and benefits the most from shorter, more consistent import statements.
import psu_capstone.encoder_layer as en
import psu_capstone.input_layer as il
from psu_capstone.log import logger
from psu_capstone.sdr_layer.sdr import SDR
from utils import DATA_PATH, PROJECT_ROOT

plt.style.use("seaborn-v0_8-poster")


def plot_sdr(data: list[int], title: str | None = None) -> None:
    """Plot a visual representation of an SDR as a 2D grid.

    Converts the 1D binary SDR into a square grid visualization where
    active bits are shown in blue and inactive bits in white.

    Args:
        data: Binary list representing the SDR (0s and 1s).
        title: Optional title for the plot.
    """

    sdr = SDR([len(data)])
    sdr.set_dense(data)
    dense = np.array(sdr.get_dense(), dtype=int)

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
    plot_title = title or "SDR Visualization"
    plt.title(plot_title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show(block=True)


def plot_heat_map(
    heat_map: np.ndarray,
    title: str | None = None,
    norm: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot a heat map visualization of the given 2D array data.

    Args:
        heat_map: 2D numpy array of values to visualize
        title: Title for the plot
        norm: Matplotlib normalization object (e.g., PowerNorm)
        vmin: Minimum value for color mapping (if norm is None)
        vmax: Maximum value for color mapping (if norm is None)
    """
    plt.figure(figsize=(12, 12))
    plt.imshow(heat_map, cmap="hot", interpolation="nearest", norm=norm, vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.colorbar(label="Duty Cycle")
    plt.xticks([])
    plt.yticks([])
    plt.show(block=True)


def visualize_signal_fft(dataset: str, sample_rate: int) -> None:
    """Plot time-domain data and FFT magnitude spectrum for the specified dataset."""
    ih = il.InputHandler()

    signal = ih.input_data(os.path.join(PROJECT_ROOT, "data", dataset))

    columns: list[str] = []

    for k in signal.keys():
        if k.lower() == "timestamp":
            continue
        columns.append(k)

        print(f"Columns: {columns}")

    for column in columns:
        values = signal[column]
        values = np.array(values, dtype=float)
        values[0] = 0.0  # remove DC component by zeroing the first value
        # values = values - np.mean(values)  # remove DC component
        values = values[:4096]

        print(f"Plotting column: {column}")

        time_axis = np.arange(len(values), dtype=float)
        plt.figure(figsize=(16, 8))
        plt.plot(time_axis, values, "r")
        plt.title(f"Sine Wave in Time Domain - {column}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

        # frequency domain
        freq_data = cast(np.ndarray, fft(values))
        samples = len(freq_data)
        freq_data = freq_data[1 : samples // 2]
        freq_bin = fftfreq(samples, 1 / sample_rate)[1 : samples // 2]
        plt.figure(figsize=(16, 8))
        plt.plot(freq_bin, np.abs(freq_data))
        peak_index = np.argmax(np.abs(freq_data))
        peak_freq = freq_bin[peak_index]
        print(f"Plot Peak Frequency: {peak_freq} Hz")

        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        plt.title(f"FFT Magnitude Spectrum - {column}")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.grid(which="both", axis="both", linestyle="--", linewidth=0.8)
        plt.show()

        fft_encoder = en.FourierEncoder(en.FourierEncoderParameters())

        sdr = fft_encoder.encode(values)

        plot_sdr(sdr)


if __name__ == "__main__":

    fft_encoder = en.FourierEncoder(
        en.FourierEncoderParameters(
            resolutions_in_ranges=[1.0, 1.0],
            frequency_ranges=[(0, 100), (100, 500)],
            size=2048,
            # active bits in range times number of ranges
            sparsity_in_ranges=[0.02, 0.02],
            sensitivity_threshold=0.01,
        )
    )

    a, b, c, d, e, f = 10, 2, 30, 2, 50, 60
    y1 = np.sin(2 * np.pi * a * np.linspace(0, 1, 2048, endpoint=False))
    y1 *= np.sin(2 * np.pi * b * np.linspace(0, 1, 2048, endpoint=False))
    # y1 += np.sin(2 * np.pi * c * np.linspace(0, 1, 2048, endpoint=False))
    # y1 += np.sin(2 * np.pi * d * np.linspace(0, 1, 2048, endpoint=False))
    y2 = np.sin(2 * np.pi * d * np.linspace(0, 1, 2048, endpoint=False))
    # y2 += np.sin(2 * np.pi * a * np.linspace(0, 1, 2048, endpoint=False))
    # y2 += np.sin(2 * np.pi * e * np.linspace(0, 1, 2048, endpoint=False))
    # y2 += np.sin(2 * np.pi * f * np.linspace(0, 1, 2048, endpoint=False))

    """
    fft_one = fft_encoder.encode(y1)
    fft_two = fft_encoder.encode(y2)

    print(f"SDR One: {len(fft_one)}")
    print(f"SDR active bits One: {sum(fft_one)}")
    print(f"SDR Two: {len(fft_two)}")
    print(f"SDR active bits Two: {sum(fft_two)}")

    overlap_bits = overlap(fft_one, fft_two)
    hamming_dist = hamming_distance(fft_one, fft_two)
    print(f"Overlap: {overlap_bits} bits")
    print(f"Hamming Distance: {hamming_dist} bits")

    #plot_sdr(fft_one)
    #plot_sdr(fft_two)

    """
    visualize_signal_fft("fin_test.csv", sample_rate=4096)
