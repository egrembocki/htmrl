"""Utility functions for PSU Capstone project. Global utilities used across the project."""

import os
from ctypes import Structure as Struct
from ctypes import c_bool, c_float, c_int

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test.csv")


class Parameters(Struct):
    """Structure to hold parameters for all encoders, with default values."""

    _fields_ = [
        # ScalarEncoder
        ("minimum", c_float),
        ("maximum", c_float),
        ("clip_input", c_bool),
        ("periodic", c_bool),
        ("category", c_bool),
        ("active_bits", c_int),
        ("sparsity", c_float),
        ("size", c_int),
        ("radius", c_float),
        ("resolution", c_float),
        ("size_or_radius_or_category_or_resolution", c_float),
        ("active_bits_or_sparsity", c_float),
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
        self.minimum = 0.0
        self.maximum = 100.0
        self.clip_input = True
        self.periodic = False
        self.active_bits = 5
        self.sparsity = 0.0
        self.size = 10
        self.radius = 0.0
        self.resolution = 0.0
        self.category = False
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
    try:
        from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder
        from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder

        p = Parameters()
        # Only pass 'size' as the second argument to avoid the error
        s_enc = ScalarEncoder(p)
        print("Smoke check passed: Parameters and ScalarEncoder initialized.", s_enc)
        rdse_enc = RandomDistributedScalarEncoder(p)
        print(
            "Smoke check passed: Parameters and RandomDistributedScalarEncoder initialized.",
            rdse_enc,
        )
        return True
    except Exception as e:
        print(f"Smoke check failed: {e}")
        return False


if __name__ == "__main__":
    print("Utility module for PSU Capstone project.")

    smoke_check()
