import random
from collections import Counter
from typing import cast

import numpy as np
import pytest

from psu_capstone.agent_layer.HTM import (
    CONNECTED_PERM,
    DESIRED_LOCAL_SPARSITY,
    RECEPTIVE_FIELD_PCT,
    Cell,
    Column,
    ColumnField,
    Field,
    InputField,
    ProximalSynapse,
)
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


def activate_cells(cf: ColumnField, input_value):
    """Activate input cells."""
    in_fi = cast(InputField, cf.input_fields[0])
    in_fi.encode(input_value)


def make_input_field(n_cells: int = 2048) -> InputField:
    """Create a simple input Field with n cells."""
    return InputField(
        RDSEParameters(resolution=1.0), n_cells
    )  # this defaults to an rdse of n_cell as size


def make_standard_cf(
    input_field: Field, num_columns: int = 2048, cells_per_column: int = 10
) -> ColumnField:
    """Construct a standard spatial temporal ColumnField for testing."""
    return ColumnField(
        input_fields=[input_field],
        num_columns=num_columns,
        cells_per_column=cells_per_column,
        non_spatial=False,
        non_temporal=True,
    )


def make_spatial_only_cf(
    input_field: Field, num_columns: int = 2048, duty_cycle_period: int = 1000
) -> ColumnField:
    """Construct a spatial only non temporal ColumnField for SP tests."""
    return ColumnField(
        input_fields=[input_field],
        num_columns=num_columns,
        cells_per_column=1,
        non_spatial=False,
        non_temporal=True,
        duty_cycle_period=duty_cycle_period,
    )


def make_temporal_only_cf(
    input_field: Field, num_columns: int = 2048, duty_cycle_period: int = 1000
) -> ColumnField:
    """Construct a temporal only non spatial ColumnField for TM tests."""
    return ColumnField(
        input_fields=[input_field],
        num_columns=num_columns,
        cells_per_column=1,
        non_spatial=True,
        non_temporal=False,
        duty_cycle_period=duty_cycle_period,
    )


def _overlap_count(first, second) -> int:
    """Count overlapping active column indices between two sets."""
    return len(first & second)


def activate_noisy(cf: ColumnField, base_value: float, noise_prob: float) -> None:
    """Encode a value then flip each bit independently with noise_prob."""
    in_fi = cast(InputField, cf.input_fields[0])
    in_fi.encode(base_value)
    # create a numpy representation of the cells
    bits = np.array([1 if cell.active else 0 for cell in in_fi.cells])
    # first generate a random array the length of cells in input field
    # this has floats between 0 and 1 at each position, if it is below the noise_prob
    # it will be marked active
    flip_mask = (np.random.rand(len(bits)) < noise_prob).astype(int)
    # bits XOR flip_mask
    noisy_bits = bits ^ flip_mask

    # adjust the cells that are active to match the noisy sdr
    for i, cell in enumerate(in_fi.cells):
        if noisy_bits[i]:
            cell.set_active()
        else:
            cell.active = False


"""Spatial Pooling inside the column field correctness tests based on Numenta Paper: spatial-pooling-algorithm/HTM-Spatial-Pooler-Overview.pdf"""

"""FIXED SPARSENESS"""


def test_active_ratio_matches_desired_sparsity():
    """After compute, the number of active columns should match desired sparsity."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)
    activate_cells(cf, 42.0)

    expected_active = int(len(cf.columns) * DESIRED_LOCAL_SPARSITY)

    cf.compute(learn=True)
    active_cols = {i for i, col in enumerate(cf.columns) if col.active}

    assert (
        len(active_cols) == expected_active
    ), f"Expected exactly {expected_active} active columns, got {len(active_cols)}"


def test_sparsity_invariant_to_input_density():
    """Output sparsity stays fixed regardless of how many input bits are on."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)
    expected_active = int(len(cf.columns) * DESIRED_LOCAL_SPARSITY)

    # use encoders with different sparsity settings to vary input density
    for input_sparsity in (0.02, 0.05, 0.10, 0.20, 0.25, 0.30, 0.80):
        cast(InputField, cf.input_fields[0])._encoder._sparsity = input_sparsity
        print(cast(InputField, cf.input_fields[0])._encoder._sparsity)
        activate_cells(cf, 100.0)
        cf.compute(learn=True)
        active_cols = {i for i, col in enumerate(cf.columns) if col.active}

        assert (
            len(active_cols) == expected_active
        ), f"Input sparsity {input_sparsity}: expected {expected_active} active cols, got {len(active_cols)}"


def test_activate_top_k_columns_respects_k():
    """activate_top_k_columns should activate exactly k columns."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    for i, col in enumerate(cf.columns):
        col.overlap = float(i)

    for k in (1, 10, 50, 128):
        for col in cf.columns:
            col.active = False
        cf.activate_top_k_columns(k)
        active_cols = {i for i, col in enumerate(cf.columns) if col.active}
        assert (
            len(active_cols) >= k
        ), f"activate_top_k_columns({k}) activated only {len(active_cols)}"


"""DISTRIBUTED CODING"""


def test_many_columns_participate_across_patterns():
    """Over many RDSE encoded inputs, a large share of columns should activate at least once."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    rng = random.Random(42)
    ever_active: set[Column] = set()
    print(len(in_fi.cells) * RECEPTIVE_FIELD_PCT)
    for _ in range(100):
        value = rng.uniform(-10000, 10000)
        activate_cells(cf, value)
        cf.compute(learn=True)
        active_cols = {col for col in cf.columns if col.active}
        ever_active.update(active_cols)
        print("Total columns ever active: ", len(ever_active))
        # for col in active_cols:
        #    print(col)

    participation = len(ever_active) / len(cf.columns)
    assert (
        participation > 0.7
    ), f"Only {participation:.1%} of columns were ever active not distributed enough"


def test_no_single_column_dominates():
    """No single column should be active in more than a small fraction of all patterns."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    rng = random.Random(42)
    activation_counts: dict = {}
    for _ in range(200):
        value = rng.uniform(-10000, 10000)
        activate_cells(cf, value)
        cf.compute(learn=True)
        for col in cf.columns:
            if col.active:
                activation_counts[col] = activation_counts.get(col, 0) + 1

    if activation_counts:
        max_freq = max(activation_counts.values()) / 200
        print(max_freq)
        threshold = min(1.0, 5 * DESIRED_LOCAL_SPARSITY)
        assert (
            max_freq < threshold
        ), f"Most used column was active {max_freq:.1%} of the time threshold {threshold:.1%}"


def test_duty_cycle_boosting_engages_inactive_columns():
    """Duty cycle tracking should engage after repeated presentations."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size, duty_cycle_period=50)

    for value in range(60):
        activate_cells(cf, value)
        cf.compute(learn=True)

    assert cf._duty_cycle_window > 0, "Duty cycle window was never advanced"

    duty_cycles = [col.active_duty_cycle for col in cf.columns]
    non_zero = sum(1 for d in duty_cycles if d > 0)
    assert non_zero > 0, "No columns have positive duty cycles"


"""PRESERVING SEMANTIC SIMILARITY"""


def test_similar_inputs_produce_similar_sdrs():
    """Nearby scalar values should produce overlapping column activations."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    base_value = 50.0
    similar_value = 51.0  # one resolution step away since default rdse is 1.0

    # train on base
    for _ in range(100):
        activate_cells(cf, base_value)
        cf.compute(learn=True)

    # test base
    activate_cells(cf, base_value)
    cf.compute(learn=False)
    cols_base = {i for i, col in enumerate(cf.columns) if col.active}

    # test similar
    activate_cells(cf, similar_value)
    cf.compute(learn=False)
    cols_similar = {i for i, col in enumerate(cf.columns) if col.active}

    sim = _overlap_count(cols_base, cols_similar)
    print(sim)
    assert (
        sim > 0
    ), f"Similar inputs ({base_value} vs {similar_value}) produced SDRs with only {sim} overlapping columns"


def test_dissimilar_inputs_produce_dissimilar_sdrs():
    """Values far apart should produce low column overlap."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    value_a = 1.0
    value_b = 1000.0  # far apart no rdse bit overlap expected

    activate_cells(cf, value_a)
    activate_cells(cf, value_b)

    # train on both
    for _ in range(100):
        cf.compute(learn=True)

    # test A
    activate_cells(cf, value_a)
    cf.compute(learn=False)
    cols_a = {i for i, col in enumerate(cf.columns) if col.active}

    # test B
    activate_cells(cf, value_b)
    cf.compute(learn=False)
    cols_b = {i for i, col in enumerate(cf.columns) if col.active}

    sim = _overlap_count(cols_a, cols_b)
    print(sim)
    expected_active = int(len(cf.columns) * DESIRED_LOCAL_SPARSITY)
    # overlap should be well below the number of active columns
    assert (
        sim < expected_active
    ), f"Distant inputs ({value_a} vs {value_b}) still had {sim} overlapping columns out of {expected_active} active"


def test_overlap_gradient():
    """As encoded values move further from the base, column overlap should decrease."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    base_value = 1

    # train on base
    for _ in range(100):
        activate_cells(cf, base_value)
        cf.compute(learn=True)

    # get base columns
    activate_cells(cf, base_value)
    cf.compute(learn=False)
    cols_base = {i for i, col in enumerate(cf.columns) if col.active}

    # test increasing distances
    overlaps = []
    for i in range(1000):
        activate_cells(cf, i)
        cf.compute(learn=False)
        cols_test = {i for i, col in enumerate(cf.columns) if col.active}
        print(_overlap_count(cols_base, cols_test))
        overlaps.append(_overlap_count(cols_base, cols_test))

    for i in range(len(overlaps)):
        if i is not len(overlaps) - 50:
            for j in range(50):
                assert (
                    overlaps[i] >= overlaps[i + j]
                ), f"Overlap increased at index {i}: {overlaps[i]} < {overlaps[i + j]}"


"""NOISE ROBUSTNESS"""


def test_small_noise_preserves_most_of_sdr():
    """Flipping a small fraction of encoded bits should keep most output columns the same."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    base_value = 50.0

    # train on clean input
    for _ in range(100):
        activate_cells(cf, base_value)
        cf.compute(learn=True)

    # test clean
    activate_cells(cf, base_value)
    cf.compute(learn=False)
    cols_clean = {i for i, col in enumerate(cf.columns) if col.active}

    # test with increasing noise
    for noise_pct in (0.05, 0.10, 0.15):
        activate_noisy(cf, base_value, noise_pct)
        cf.compute(learn=False)
        cols_noisy = {i for i, col in enumerate(cf.columns) if col.active}
        sim = _overlap_count(cols_clean, cols_noisy)
        print(sim)
        assert sim > 0, f"{noise_pct:.0%} input noise left zero column overlap"


def test_repeated_noise_does_not_drift_representation():
    """Repeatedly presenting noisy versions should not cause the clean representation to drift."""
    input_size = 2048
    in_fi = make_input_field(input_size)
    cf = make_spatial_only_cf(in_fi, num_columns=input_size)

    base_value = 50.0

    # initial training
    for _ in range(20):
        activate_cells(cf, base_value)
        cf.compute(learn=True)

    activate_cells(cf, base_value)
    cf.compute(learn=False)
    cols_before = {i for i, col in enumerate(cf.columns) if col.active}

    # present many noisy variants with learning on
    for _ in range(50):
        activate_noisy(cf, base_value, 0.15)
        cf.compute(learn=True)

    # retest clean
    activate_cells(cf, base_value)
    cf.compute(learn=False)
    cols_after = {i for i, col in enumerate(cf.columns) if col.active}

    sim = _overlap_count(cols_before, cols_after)
    print(sim)
    assert sim > 0, f"Representation {sim} drifted completely after noisy training: zero overlap"


"""FAULT TOLERANCE"""

"""CONTINUOUS LEARNING"""

"""STABILITY"""
