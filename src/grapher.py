import os
from cmath import phase
from dataclasses import dataclass, field
from importlib import simple
from random import sample
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from scipy.fft import fft, fftfreq, ifft

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.fourier_encoder import FourierEncoder, FourierEncoderParameters
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.log import logger
from psu_capstone.sdr_layer.sdr import SDR
from utils import DATA_PATH, PROJECT_ROOT

plt.style.use("seaborn-v0_8-poster")


def plot_sdr(data: list[int]) -> None:
    """Plot a visual representation of the given SDR data."""

    print(f"SDR data length: {len(data)}")
    print(f"SDR active bits: {sum(data)}")

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
    title = "SDR Visualization"
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show(block=True)


def plot_hot_gym_fft(sample_rate: int = 256, dataset: str = "hot_gym_short.csv") -> None:
    """Plot time-domain data and FFT magnitude spectrum for the specified dataset."""
    ih = InputHandler()
    hot_gym = ih.input_data(os.path.join(PROJECT_ROOT, "data", dataset))

    signal = (
        cast(pd.DataFrame, hot_gym)
        .drop(columns="timestamp")
        .to_numpy(dtype=float, copy=False)
        .flatten()
    )

    # time domain example: sine waves
    phase_shift = np.cos(2 * np.pi * 10 * np.linspace(0, 1, 2048, endpoint=False))

    signal = 0.5 * np.sin(2 * np.pi * (100) * np.linspace(0, 1, 2048, endpoint=False) + phase_shift)
    # signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 2048, endpoint=False))

    sample_rate = len(signal)  # samples per second

    time_axis = np.arange(signal.size, dtype=float) / sample_rate
    plt.figure(figsize=(16, 8))
    plt.plot(time_axis, signal, "r")
    plt.title("Sine Wave in Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    # frequency domain

    freq_data = cast(np.ndarray, fft(signal))
    samples = len(freq_data)
    freq_data = freq_data[: samples // 2]
    freq_bin = fftfreq(samples, 1 / sample_rate)[: samples // 2]
    plt.figure(figsize=(16, 8))
    plt.plot(freq_bin, np.abs(freq_data))
    peak_index = np.argmax(np.abs(freq_data))
    peak_freq = freq_bin[peak_index]
    print(f"FFT Peak Frequency: {peak_freq} Hz")

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(freq_data) // 10 or 1))
    plt.title("FFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(which="both", axis="both", linestyle="--", linewidth=0.8)
    plt.show()


if __name__ == "__main__":

    fft_encoder = FourierEncoder(
        FourierEncoderParameters(
            resolutions_in_ranges=[1],
            frequency_ranges=[(0, 200)],
            # search for frequencies peaks between 0 and 200 Hz
            active_bits_in_ranges=[5],
            # every contributing frequency gets 5 active bits
            size=2048,
            # use size and total active bits to set sparsity
            total_active_bits=40,
        )
    )

    p = np.cos(2 * np.pi * 10 * np.linspace(0, 1, 2048, endpoint=False))
    y = np.sin(2 * np.pi * 56 * np.linspace(0, 1, 2048, endpoint=False) + p)

    fft_data = fft_encoder.encode(y)

    plot_sdr(fft_data)

    # plot_hot_gym_fft(sample_rate=256)
