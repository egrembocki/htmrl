import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from scipy.fft import fft, fftfreq

from psu_capstone.encoder_layer.fourier_encoder import FourierEncoder, FourierEncoderParameters
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.sdr_layer.sdr import SDR
from utils import DATA_PATH, PROJECT_ROOT, hamming_distance, overlap

plt.style.use("seaborn-v0_8-poster")


def plot_sdr(data: list[int]) -> None:
    """Plot a visual representation of the given SDR data."""

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
    hot_gym_records = ih.input_data(os.path.join(PROJECT_ROOT, "data", dataset))
    signal_values: list[float] = []
    for record in hot_gym_records:
        for key, value in record.items():
            if key == "timestamp":
                continue
            signal_values.append(float(value))
    signal = np.asarray(signal_values, dtype=float)

    # signal = 0.5 * np.sin(2 * np.pi * (100) * np.linspace(0, 1, 2048, endpoint=False) + phase_shift)
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

    fft_encoder = FourierEncoder()

    sdr_hot_gym = fft_encoder.encode(signal)

    plot_sdr(sdr_hot_gym)


if __name__ == "__main__":

    fft_encoder = FourierEncoder(
        FourierEncoderParameters(
            resolutions_in_ranges=[1.0],
            # search for frequencies peaks between 0 and 200 Hz
            frequency_ranges=[(0, 100)],
            # every contributing frequency gets 40 active bits, this divides up from total active bits
            size=4096,
            # active bits in range times number of ranges
            sparsity_in_ranges=[0.01],
            total_sparsity=0.01,
        )
    )

    a, b, c, d = 1, 2, 3, 4

    y1 = np.sin(2 * np.pi * a * np.linspace(0, 1, 2048, endpoint=False))
    y1 += np.sin(2 * np.pi * b * np.linspace(0, 1, 2048, endpoint=False))
    y1 += np.sin(2 * np.pi * c * np.linspace(0, 1, 2048, endpoint=False))
    y2 = np.sin(2 * np.pi * c * np.linspace(0, 1, 2048, endpoint=False))
    y2 += np.sin(2 * np.pi * d * np.linspace(0, 1, 2048, endpoint=False))
    y2 += np.sin(2 * np.pi * a * np.linspace(0, 1, 2048, endpoint=False))

    fft_one = fft_encoder.encode(y1)
    fft_two = fft_encoder.encode(y2)

    print(f"SDR One: {len(fft_one)}")
    print(f"SDR active bits One: {sum(fft_one)}")
    print(f"SDR Two: {len(fft_two)}")
    print(f"SDR active bits Two: {sum(fft_two)}")

    plot_sdr(fft_one)
    plot_sdr(fft_two)

    fft_one = np.array(fft_one)
    fft_two = np.array(fft_two)

    hamming = hamming_distance(fft_one, fft_two)
    print(f"Hamming distance between SDRs: {hamming} bits")
    overlap = overlap(fft_one, fft_two)
    print(f"Overlap between SDRs: {overlap} bits")

    ih = InputHandler()
    hot_gym_records = ih.input_data(os.path.join(PROJECT_ROOT, "data", "rec-center-hourly.csv"))
    signal_values = []
    for record in hot_gym_records:
        for key, value in record.items():
            if key == "timestamp":
                continue
            signal_values.append(float(value))
    signal = np.asarray(signal_values, dtype=float)

    fft_encoder = FourierEncoder(
        FourierEncoderParameters(
            resolutions_in_ranges=[0.10],
            total_resolution=0.1,
            # search for frequencies peaks between 0 and 200 Hz
            frequency_ranges=[(0, 128)],
            # every contributing frequency gets 40 active bits, this divides up from total active bits
            size=2048,
            # active bits in range times number of ranges
            sparsity_in_ranges=[0.02],
            total_sparsity=0.02,
        )
    )

    print(signal)

    sdr_hot_gym = fft_encoder.encode(signal)

    plot_sdr(sdr_hot_gym)
