import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from scipy.fft import fft, ifft

from psu_capstone.encoder_layer.fourier_encoder import FourierEncoder
from psu_capstone.input_layer.input_handler import InputHandler
from utils import DATA_PATH, PROJECT_ROOT

plt.style.use("seaborn-v0_8-poster")

if __name__ == "__main__":

    ih = InputHandler()

    # df_sine = cast(pd.DataFrame, ih.input_data(os.path.join(PROJECT_ROOT, "data", "sine_wave.csv")))
    # sin_wave = df_sine.to_numpy(dtype=float, copy=False).flatten()

    sample_rate = 190
    time_step = 1 / sample_rate
    t = np.arange(0, 1, time_step, dtype=float)
    f = 100

    # sin_wave = np.sin(2 * np.pi * f * t)
    # sin_wave += 2 * np.sin(2 * np.pi * 200 * t)
    # sin_wave += np.sin(2 * np.pi * 300 * t)

    hot_gym = ih.input_data(os.path.join(PROJECT_ROOT, "data", "hot_gym_short.csv"))

    hot_gym = cast(pd.DataFrame, hot_gym).drop(columns="timestamp")

    hot_gym = hot_gym.to_numpy(dtype=float, copy=False).flatten()

    print(hot_gym.shape)

    fourier_encoder = FourierEncoder()
    # freq_data = fourier_encoder.transform(hot_gym)
    gym_fft = cast(np.ndarray, fft(hot_gym))

    freq_data = gym_fft
    y = hot_gym

    plt.figure(figsize=(16, 8))
    plt.plot(t, y, "r")  # Plot the sine wave, plot(x, y, 'r') means red line
    plt.title("Sine Wave in Time Domain")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    samples = len(freq_data)
    samples_n = np.arange(samples)
    period = samples / sample_rate
    freq = samples_n / period

    # divide spectrum in half due to N/2 symmetry
    freq_data = freq_data[: len(freq_data) // 2]
    freq_data = fourier_encoder._normalize(freq_data)
    freq = freq[: samples // 2]

    plt.figure(figsize=(16, 8))
    plt.stem(freq, np.abs(freq_data), linefmt="b-", markerfmt="bo", basefmt="r-")
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(len(freq_data) // 10))
    plt.title("FFT Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(which="both", axis="both", linestyle="--", linewidth=0.8)
    plt.show()
