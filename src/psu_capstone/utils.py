"""Utility functions for PSU Capstone project. Global utilities used across the project."""

from ctypes import Structure as Struct
from ctypes import c_bool, c_float, c_int
from math import isclose


class Parameters(Struct):
    """Structure to hold parameters for all encoders, with default values."""

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


def smoke_check():
    """Basic smoke check for utils module."""

    params = Parameters()
    assert isclose(params.scalar_minimum, 0.0)
    assert params.rdse_seed == 42
    assert params.day_of_week_width == 3
    print("Smoke check passed.")


if __name__ == "__main__":
    print("Utility module for PSU Capstone project.")

    smoke_check()
