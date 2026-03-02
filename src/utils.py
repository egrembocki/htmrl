"""Utility functions for PSU Capstone project.

This module provides global utilities used across the project, including
SDR comparison functions and encoder parameter management.
"""

import os
from ctypes import Structure as Struct
from ctypes import c_bool, c_float, c_int
from math import isclose

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")


def hamming_distance(sdr1: np.ndarray | list[int], sdr2: np.ndarray | list[int]) -> int:
    """Calculate the Hamming distance between two SDRs.

    The Hamming distance is the number of positions at which the corresponding
    bits differ between two binary arrays.

    Args:
        sdr1: First sparse distributed representation array.
        sdr2: Second sparse distributed representation array.

    Returns:
        The number of differing bits between the two SDRs.

    Raises:
        ValueError: If SDRs have different shapes.
    """
    sdr1 = np.asarray(sdr1, dtype=bool)
    sdr2 = np.asarray(sdr2, dtype=bool)
    if sdr1.shape != sdr2.shape:
        raise ValueError("SDRs must have the same shape for Hamming distance calculation.")

    return int(np.count_nonzero(sdr1 != sdr2))


def overlap(sdr1: np.ndarray | list[int], sdr2: np.ndarray | list[int]) -> int:
    """Calculate matching active bits between two SDRs.

    The overlap measures how many bits are active (set to 1) in both SDRs,
    which indicates similarity between the representations.

    Args:
        sdr1: First sparse distributed representation array.
        sdr2: Second sparse distributed representation array.

    Returns:
        The count of bits that are active in both SDRs.

    Raises:
        ValueError: If SDRs have different shapes.
    """
    sdr1 = np.asarray(sdr1, dtype=bool)
    sdr2 = np.asarray(sdr2, dtype=bool)
    if sdr1.shape != sdr2.shape:
        raise ValueError("SDRs must have the same shape for overlap calculation.")

    return int(np.sum(np.logical_and(sdr1, sdr2)))


class Parameters(Struct):
    """Structure to hold parameters for all encoder types.

    This ctypes Structure provides a unified interface for configuring
    scalar encoders, RDSE encoders, date encoders, and category encoders
    with type-safe default values.

    Initializes with sensible default values suitable for general-purpose
    encoding tasks. Defaults can be overridden after instantiation.

    Attributes:
        scalar_minimum: Minimum value for scalar encoding.
        scalar_maximum: Maximum value for scalar encoding.
        scalar_clip_input: Whether to clip input values to min/max range.
        scalar_periodic: Whether the scalar encoding wraps periodically.
        scalar_category: Whether scalar encoder treats input as categories.
        scalar_active_bits: Number of active bits in scalar encoding.
        scalar_sparsity: Sparsity level for scalar encoding.
        scalar_size: Total size of scalar encoding output.
        scalar_radius: Radius parameter for scalar encoding.
        scalar_resolution: Resolution parameter for scalar encoding.
        rdse_active_bits: Number of active bits in RDSE encoding.
        rdse_sparsity: Sparsity level for RDSE encoding.
        rdse_size: Total size of RDSE encoding output.
        rdse_radius: Radius parameter for RDSE encoding.
        rdse_category: Whether RDSE treats input as categories.
        rdse_resolution: Resolution parameter for RDSE encoding.
        rdse_seed: Random seed for RDSE initialization.
        season_width: Width of seasonal encoding in date encoder.
        season_radius: Radius for seasonal encoding in date encoder.
        day_of_week_width: Width of day-of-week encoding.
        day_of_week_radius: Radius for day-of-week encoding.
        weekend_width: Width of weekend indicator encoding.
        holiday_width: Width of holiday indicator encoding.
        time_of_day_width: Width of time-of-day encoding.
        time_of_day_radius: Radius for time-of-day encoding.
        custom_width: Width for custom date encoding features.
        verbose: Whether to enable verbose logging.
        cat_w: Width parameter for category encoding.
    """

    _fields_ = [
        # ScalarEncoder
        ("scalar_minimum", c_float),
        ("scalar_maximum", c_float),
        ("scalar_clip_input", c_bool),
        ("scalar_periodic", c_bool),
        ("scalar_category", c_bool),
        ("scalar_active_bits", c_int),
        ("scalar_sparsity", c_float),
        ("scalar_size", c_int),
        ("scalar_radius", c_float),
        ("scalar_resolution", c_float),
        # RDSE
        ("rdse_active_bits", c_int),
        ("rdse_sparsity", c_float),
        ("rdse_size", c_int),
        ("rdse_radius", c_float),
        ("rdse_category", c_bool),
        ("rdse_resolution", c_float),
        ("rdse_seed", c_int),
        # DateEncoder
        ("season_width", c_int),
        ("season_radius", c_float),
        ("day_of_week_width", c_int),
        ("day_of_week_radius", c_float),
        ("weekend_width", c_int),
        ("holiday_width", c_int),
        ("time_of_day_width", c_int),
        ("time_of_day_radius", c_float),
        ("custom_width", c_int),
        ("verbose", c_bool),
        # CategoryEncoder
        ("cat_w", c_int),
    ]

    def __init__(self):
        super().__init__()
        # ScalarEncoder defaults
        self.scalar_minimum = 0.0
        self.scalar_maximum = 100.0
        self.scalar_clip_input = True
        self.scalar_periodic = False
        self.scalar_active_bits = 5
        self.scalar_sparsity = 0.0
        self.scalar_size = 10
        self.scalar_radius = 0.0
        self.scalar_resolution = 0.0
        self.scalar_category = False
        self.size_or_radius_or_category_or_resolution = 0.0
        self.active_bits_or_sparsity = 0.0
        # RDSE defaults
        self.rdse_active_bits = 5
        self.rdse_sparsity = 0.0
        self.rdse_size = 10
        self.rdse_radius = 5.0
        self.rdse_category = False
        self.rdse_resolution = 0.0
        self.rdse_seed = 42
        # DateEncoder defaults
        self.season_width = 0
        self.season_radius = 91.5
        self.day_of_week_width = 3
        self.day_of_week_radius = 1.0
        self.weekend_width = 3
        self.holiday_width = 0
        self.time_of_day_width = 3
        self.time_of_day_radius = 4.0
        self.custom_width = 0
        self.verbose = False
        # CategoryEncoder defaults
        self.cat_w = 3


def smoke_check() -> None:
    """Run basic smoke tests to verify utils module functionality.

    Performs simple assertions on Parameters initialization to ensure
    default values are set correctly. Outputs path information and
    success message.

    Raises:
        AssertionError: If any parameter defaults are not set correctly.
    """
    params = Parameters()
    assert isclose(params.scalar_minimum, 0.0)
    assert params.rdse_seed == 42
    assert params.day_of_week_width == 3

    print(PROJECT_ROOT + DATA_PATH)
    print("Smoke check passed.")


if __name__ == "__main__":
    print("Utility module for PSU Capstone project.")

    smoke_check()
