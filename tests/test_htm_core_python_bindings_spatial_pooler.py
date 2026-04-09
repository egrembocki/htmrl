"""
# -*- coding: utf-8 -*-
'''
Created on Thu Apr  9 07:17:55 2026

@author: Josh
This was made in spyder ide with a different local environment. Including it here as this tests were made by me
but I did not want to make htm core python bindings a dependency for this repo. This was so Dr. Agrawal can compare
the HTM results.
'''
import numpy as np
import matplotlib.pyplot as plt
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.encoders import RDSE, RDSE_Parameters
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters
import random


def get_random_indices(size: int, active_bits: int):
    # range of length of cells and getting the number of active bits desired.
    # this is likely 2048 and 40.
    indices = random.sample(range(size), active_bits)
    return indices


def test_activation_with_random_cells_excluding_encoder():
    '''Active column frequency over 49 epochs on random inputs, bypassing the encoder'''

    input_size  = 2048
    column_count = 2048
    active_bits = 40

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    sp = SpatialPooler(
        inputDimensions           = (input_size,),
        columnDimensions          = (column_count,),
        potentialPct              = 1,
        globalInhibition          = True,
        localAreaDensity          = 0.02,
        synPermInactiveDec        = 0.02,
        synPermActiveInc          = 0.10,
        synPermConnected          = 0.50,
        boostStrength             = 0.0,
        wrapAround                = False,
    )

    all_indices = []
    for _ in range(100):
        random_indices = get_random_indices(input_size, active_bits)
        random_indices = sorted(random_indices)
        all_indices.append(random_indices)

    for _ in range(49):
        for idx in all_indices:
            input_sdr.sparse = idx
            sp.compute(input_sdr, True, active_sdr)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for idx in all_indices:
        input_sdr.sparse = idx
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with random cells excluding encoder\n(epoch 49)",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()


def test_activation_with_random_cells_excluding_encoder_with_boost():
    '''Active column frequency over 49 epochs on random inputs, bypassing the encoder'''

    input_size  = 2048
    column_count = 2048
    active_bits = 40

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    sp = SpatialPooler(
        inputDimensions           = (input_size,),
        columnDimensions          = (column_count,),
        potentialPct              = 1,
        globalInhibition          = True,
        localAreaDensity          = 0.02,
        synPermInactiveDec        = 0.02,
        synPermActiveInc          = 0.10,
        synPermConnected          = 0.50,
        boostStrength             = 100.0,
        wrapAround                = False,
    )

    all_indices = []
    for _ in range(100):
        random_indices = get_random_indices(input_size, active_bits)
        random_indices = sorted(random_indices)
        all_indices.append(random_indices)

    for _ in range(49):
        for idx in all_indices:
            input_sdr.sparse = idx
            sp.compute(input_sdr, True, active_sdr)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for idx in all_indices:
        input_sdr.sparse = idx
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with random cells excluding encoder with 100 boost\n(epoch 49)",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()


def test_activation_converge_on_desired_sparsity_random_once_with_rdse():
    '''Random inputs through encoder and SP to see activations converge on desired sparsity'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # encoder
    rdse_params = RDSE_Parameters()
    rdse_params.size       = input_size
    rdse_params.activeBits = active_bits
    rdse_params.resolution = 1
    encoder = RDSE(rdse_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    # spatial pooler
    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 0.0,
        wrapAround         = False,
    )

    pattern_a = random.sample(range(-10000, 10000), 100)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            encoder.encode(value, input_sdr)
            sp.compute(input_sdr, True, active_sdr)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        encoder.encode(value, input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with Random data once with rdse\n(epoch 49)",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()


def test_activation_converge_on_desired_sparsity_random_once_with_rdse_and_boost():
    '''Random inputs through encoder and SP to see activations converge on desired sparsity'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # encoder
    rdse_params = RDSE_Parameters()
    rdse_params.size       = input_size
    rdse_params.activeBits = active_bits
    rdse_params.resolution = 1
    encoder = RDSE(rdse_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    # spatial pooler
    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 100.0,
        wrapAround         = False,
    )

    pattern_a = random.sample(range(-10000, 10000), 100)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            encoder.encode(value, input_sdr)
            sp.compute(input_sdr, True, active_sdr)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        encoder.encode(value, input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with Random data once with rdse and boost\n(epoch 49)",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()


def test_activation_converge_on_desired_sparsity_random_once_with_scalar_encoder():
    '''Random inputs through encoder and SP to see activations converge on desired sparsity'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # encoder
    scalar_params = ScalarEncoderParameters()
    scalar_params.size       = input_size
    scalar_params.activeBits = active_bits
    scalar_params.minimum = -10000
    scalar_params.maximum = 10000
    encoder = ScalarEncoder(scalar_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    # spatial pooler
    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 0.0,
        wrapAround         = False,
    )

    pattern_a = random.sample(range(-10000, 10000), 100)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            encoder.encode(value, input_sdr)
            sp.compute(input_sdr, True, active_sdr)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        encoder.encode(value, input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with Random data once with scalar encoder\n(epoch 49)",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()


def test_activation_converge_on_desired_sparsity_random_once_with_scalar_encoder_and_boost():
    '''Random inputs through encoder and SP to see activations converge on desired sparsity'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # encoder
    scalar_params = ScalarEncoderParameters()
    scalar_params.size       = input_size
    scalar_params.activeBits = active_bits
    scalar_params.minimum = -10000
    scalar_params.maximum = 10000
    encoder = ScalarEncoder(scalar_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    # spatial pooler
    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 100.0,
        wrapAround         = False,
    )

    pattern_a = random.sample(range(-10000, 10000), 100)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            encoder.encode(value, input_sdr)
            sp.compute(input_sdr, True, active_sdr)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        encoder.encode(value, input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with Random data once with scalar encoder and boost\n(epoch 49)",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()


def test_activation_zero_epoch_exclude_encoder():
    '''Zero epoch untrained SP on random inputs, bypassing the encoder'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 0.0,
        wrapAround         = False,
    )

    # build 100 random indices
    all_indices = []
    for _ in range(100):
        random_indices = get_random_indices(input_size, active_bits)
        random_indices = sorted(random_indices)
        all_indices.append(random_indices)

    # no training — measure directly
    activation_counts = {}
    for idx in all_indices:
        input_sdr.sparse = idx
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with Random data zero epoch excluding encoder",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()


def test_activation_zero_epoch_scalar_encoder():
    '''Zero epoch untrained SP on random inputs, bypassing the encoder'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # scalar encoder
    enc_params = ScalarEncoderParameters()
    enc_params.size       = input_size
    enc_params.activeBits = active_bits
    enc_params.minimum    = -10000
    enc_params.maximum    =  10000
    encoder = ScalarEncoder(enc_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 0.0,
        wrapAround         = False,
    )

    # 100 random scalar values
    pattern = random.sample(range(-10000, 10000), 100)

    # no training measure directly
    activation_counts = {}
    for value in pattern:
        encoder.encode(value, input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with Random data zero epoch scalar encoder",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()


def test_activation_zero_epoch_rdse():
    '''Zero epoch untrained SP on random inputs, bypassing the encoder'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # encoder
    rdse_params = RDSE_Parameters()
    rdse_params.size       = input_size
    rdse_params.activeBits = active_bits
    rdse_params.resolution = 1
    encoder = RDSE(rdse_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 0.0,
        wrapAround         = False,
    )

    # 100 random scalar values
    pattern = random.sample(range(-10000, 10000), 100)

    # no training measure directly
    activation_counts = {}
    for value in pattern:
        encoder.encode(value, input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with Random data zero epoch rdse",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.00)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()


def test_activation_with_sin_wave_scalar_encoder_periodic_false():
    '''Tests a spatial pooler with the scalar encoder and periodic false'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # scalar encoder
    scalar_params = ScalarEncoderParameters()
    scalar_params.size       = input_size
    scalar_params.activeBits = active_bits
    scalar_params.minimum    = -1
    scalar_params.maximum    =  1
    scalar_params.periodic   = False
    encoder = ScalarEncoder(scalar_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 0.0,
        wrapAround         = False,
    )

    x = np.linspace(0, 1, 100, endpoint=False)
    pattern_a = np.sin(2 * np.pi * 1 * x)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            encoder.encode(float(value), input_sdr)
            sp.compute(input_sdr, True, active_sdr)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        encoder.encode(float(value), input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with sin wave with scalar encoder periodic false\n(epoch 49)",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.0)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()

def test_activation_with_sin_wave_scalar_encoder_periodic_false_and_boosting():
    '''Tests a spatial pooler with the scalar encoder and periodic false'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # scalar encoder
    scalar_params = ScalarEncoderParameters()
    scalar_params.size       = input_size
    scalar_params.activeBits = active_bits
    scalar_params.minimum    = -1
    scalar_params.maximum    =  1
    scalar_params.periodic   = False
    encoder = ScalarEncoder(scalar_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 100.0,
        wrapAround         = False,
    )

    x = np.linspace(0, 1, 100, endpoint=False)
    pattern_a = np.sin(2 * np.pi * 1 * x)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            encoder.encode(float(value), input_sdr)
            sp.compute(input_sdr, True, active_sdr)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        encoder.encode(float(value), input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with sin wave with scalar encoder periodic false and boosting\n(epoch 49)",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.0)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()

def test_activation_with_sin_wave_with_rdse():
    '''Tests a sin wave and rdse encoder spatial pooler to see dominate columns form on the pattern'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # rdse encoder
    rdse_params = RDSE_Parameters()
    rdse_params.size       = input_size
    rdse_params.activeBits = active_bits
    rdse_params.resolution = 0.001
    encoder = RDSE(rdse_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 0.0,
        wrapAround         = False,
    )

    x = np.linspace(0, 1, 100, endpoint=False)
    pattern_a = np.sin(2 * np.pi * 1 * x)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            encoder.encode(float(value), input_sdr)
            sp.compute(input_sdr, True, active_sdr)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        encoder.encode(float(value), input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with sin wave with rdse encoder and boosting\n(epoch 49)",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.0)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()
def test_activation_with_sin_wave_with_rdse_and_boosting():
    '''Tests a sin wave and rdse encoder spatial pooler to see dominate columns form on the pattern'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # rdse encoder
    rdse_params = RDSE_Parameters()
    rdse_params.size       = input_size
    rdse_params.activeBits = active_bits
    rdse_params.resolution = 0.001
    encoder = RDSE(rdse_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 100.0,
        wrapAround         = False,
    )

    x = np.linspace(0, 1, 100, endpoint=False)
    pattern_a = np.sin(2 * np.pi * 1 * x)

    # train for 49 epochs
    for _ in range(49):
        for value in pattern_a:
            encoder.encode(float(value), input_sdr)
            sp.compute(input_sdr, True, active_sdr)

    # measure activation frequency across the full pattern
    activation_counts = {}
    for value in pattern_a:
        encoder.encode(float(value), input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        for col in active_sdr.sparse:
            activation_counts[col] = activation_counts.get(col, 0) + 1

    freqs = []
    for col in range(column_count):
        if col in activation_counts:
            freqs.append(activation_counts[col] / 100)
        else:
            freqs.append(0.0)
    print(freqs)
    n_active = sum(1 for f in freqs if f > 0)
    pct_active = (n_active / column_count) * 100
    print(f"Percent of columns ever active: {pct_active:.2f}")

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))
    max_freq = max(freqs) if max(freqs) > 0 else 0.10
    bin_edges = np.linspace(0, max_freq, int(max_freq / 0.01) + 1)

    freq_array = np.array(freqs)
    counts, _ = np.histogram(freq_array, bins=bin_edges)
    fractions = counts / len(freq_array)
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1]
    ax.bar(bin_centers, fractions, width=bin_widths, align="center",
           color="#3366CC", edgecolor="white", linewidth=0.5)
    ax.set_title(
        "Activation Frequency Distribution with sin wave with rdse encoder and boosting\n(epoch 49)",
        fontsize=8, fontweight="bold",
    )
    ax.set_xlabel("Activation Frequency", fontsize=11)
    ax.set_ylabel("Fraction of SP Columns", fontsize=11)
    ax.set_xlim(-0.005, max_freq)
    ax.set_ylim(0.0, 1.0)
    ax.text(0.95, 0.95,
            f"{n_active} / {column_count} cols active ({pct_active:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.show()

def _overlap_count_list(first: list[int], second: list[int]) -> int:
    return sum(1 for a, b in zip(first, second) if a == 1 and b == 1)

def test_scalar_gradient():
    '''Tests the overlaps of scalar encoded values to the base value 1.
    This reveals how the similarity acts between values and sdrs'''
    input_size  = 2048
    active_bits = 40

    scalar_params = ScalarEncoderParameters()
    scalar_params.size       = input_size
    scalar_params.activeBits = active_bits
    scalar_params.minimum    = 0
    scalar_params.maximum    = 1000
    scalar_params.periodic   = True
    encoder = ScalarEncoder(scalar_params)

    def encode_to_dense_list(value) -> list[int]:
        sdr = SDR(input_size)
        encoder.encode(value, sdr)
        dense = [0] * input_size
        for i in sdr.sparse:
            dense[i] = 1
        return dense

    base_value = 1
    base_encoding = encode_to_dense_list(base_value)

    overlaps = []
    for offset in range(1, 1000):
        value = base_value + offset
        encoding = encode_to_dense_list(value)
        overlap = _overlap_count_list(base_encoding, encoding)
        overlaps.append(overlap)

    plt.plot(range(1, 1000), overlaps)
    plt.xlabel("Offset")
    plt.ylabel("Overlap with base encoding")
    plt.title("Scalar Encoder Overlap vs Distance")
    plt.show()

def test_rdse_gradient():
    '''Tests the overlaps of rdse encoded values to the base value 1.
    This reveals how the similarity acts between values and sdrs.'''
    input_size  = 2048
    active_bits = 40

    rdse_params = RDSE_Parameters()
    rdse_params.size       = input_size
    rdse_params.activeBits = active_bits
    rdse_params.resolution = 1
    encoder = RDSE(rdse_params)

    def encode_to_dense_list(value) -> list[int]:
        sdr = SDR(input_size)
        encoder.encode(value, sdr)
        dense = [0] * input_size
        for i in sdr.sparse:
            dense[i] = 1
        return dense

    base_value = 1
    base_encoding = encode_to_dense_list(base_value)

    overlaps = []
    for offset in range(1, 1000):
        value = base_value + offset
        encoding = encode_to_dense_list(value)
        overlap = _overlap_count_list(base_encoding, encoding)
        overlaps.append(overlap)

    plt.plot(range(1, 1000), overlaps)
    plt.xlabel("Offset")
    plt.ylabel("Overlap with base encoding")
    plt.title("RDSE Overlap vs Distance")
    plt.show()

def test_sp_overlap_gradient():
    ''''''Tests the spatial poolers overlap gradient when compared to a base value.
    This paired with the rdse and scalar
    gradient tests can help reveal how the SP is acting directly.'''
    input_size   = 2048
    column_count = 2048
    active_bits  = 40

    # encoder
    rdse_params = RDSE_Parameters()
    rdse_params.size       = input_size
    rdse_params.activeBits = active_bits
    rdse_params.resolution = 1
    encoder = RDSE(rdse_params)

    input_sdr  = SDR(input_size)
    active_sdr = SDR(column_count)

    sp = SpatialPooler(
        inputDimensions    = (input_size,),
        columnDimensions   = (column_count,),
        potentialPct       = 1,
        globalInhibition   = True,
        localAreaDensity   = 0.02,
        stimulusThreshold  = 1,
        synPermInactiveDec = 0.02,
        synPermActiveInc   = 0.10,
        synPermConnected   = 0.50,
        boostStrength      = 0.0,
        dutyCyclePeriod    = 1000,
        wrapAround         = False,
    )

    def sdr_to_dense_list(sdr: SDR, size: int) -> list[int]:
        dense = [0] * size
        for i in sdr.sparse:
            dense[i] = 1
        return dense

    base_value = 1

    for _ in range(100):
        encoder.encode(base_value, input_sdr)
        sp.compute(input_sdr, True, active_sdr)

    encoder.encode(base_value, input_sdr)
    sp.compute(input_sdr, False, active_sdr)
    cols_base = sdr_to_dense_list(active_sdr, column_count)

    overlaps = []
    for offset in range(1, 1000):
        encoder.encode(base_value + offset, input_sdr)
        sp.compute(input_sdr, False, active_sdr)
        cols_test = sdr_to_dense_list(active_sdr, column_count)
        overlaps.append(_overlap_count_list(cols_base, cols_test))

    plt.plot(range(1, 1000), overlaps)
    plt.xlabel("Offset")
    plt.ylabel("Overlap with base encoding")
    plt.title("SP Overlap vs Distance with RDSE input field")
    plt.show()


if __name__ == "__main__":
    test_activation_with_random_cells_excluding_encoder()
    test_activation_with_random_cells_excluding_encoder_with_boost()
    test_activation_converge_on_desired_sparsity_random_once_with_rdse()
    test_activation_converge_on_desired_sparsity_random_once_with_rdse_and_boost()
    test_activation_converge_on_desired_sparsity_random_once_with_scalar_encoder()
    test_activation_converge_on_desired_sparsity_random_once_with_scalar_encoder_and_boost()
    test_activation_zero_epoch_exclude_encoder()
    test_activation_zero_epoch_scalar_encoder()
    test_activation_zero_epoch_rdse()
    test_activation_with_sin_wave_scalar_encoder_periodic_false()
    test_activation_with_sin_wave_scalar_encoder_periodic_false_and_boosting()
    test_activation_with_sin_wave_with_rdse()
    test_activation_with_sin_wave_with_rdse_and_boosting()
    test_scalar_gradient()
    test_rdse_gradient()
    test_sp_overlap_gradient()

"""
