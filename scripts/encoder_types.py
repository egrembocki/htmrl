#!/usr/bin/env python3
"""Demo all encoder types with valid sample inputs and SDR visualization.

Usage:
        python scripts/encoder_types.py
        python scripts/encoder_types.py --encoder rdse
        python scripts/encoder_types.py --no-plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np

import grapher
import psu_capstone.encoder_layer as el


@dataclass
class EncoderDemo:
    name: str
    build_and_encode: Callable[[], list[int]]
    input_summary: str


def _rdse_demo() -> list[int]:
    params = el.RDSEParameters(size=256, sparsity=0.1, resolution=0.05, seed=42)
    encoder = el.RandomDistributedScalarEncoder(params)
    return encoder.encode(12.34)


def _scalar_demo() -> list[int]:
    params = el.ScalarEncoderParameters(
        minimum=0,
        maximum=100,
        active_bits=16,
        size=256,
        radius=0.0,
        resolution=1.0,
    )
    encoder = el.ScalarEncoder(params)
    return encoder.encode(42)


def _category_demo() -> list[int]:
    params = el.CategoryParameters(w=8, category_list=["cat", "dog", "bird"], rdse_used=True)
    encoder = el.CategoryEncoder(params)
    return encoder.encode("dog")


def _category_new_demo() -> list[int]:
    params = el.CategoryParametersNew(
        active_bits_per_category=12,
        size=256,
        category_list=["red", "green", "blue"],
        rdse_used=True,
    )
    encoder = el.CategoryEncoderNew(params)
    return encoder.encode("green")


def _date_demo() -> list[int]:
    params = el.DateEncoderParameters(
        size=512,
        year_size=64,
        season_size=64,
        day_of_week_size=64,
        weekend_size=64,
        holiday_size=64,
        time_of_day_size=64,
        custom_size=64,
    )
    encoder = el.DateEncoder(params)
    return encoder.encode(datetime(2026, 4, 19, 15, 30, 0))


def _fourier_demo() -> list[int]:
    params = el.FourierEncoderParameters(
        size=512,
        frequency_ranges=[(0, 32), (32, 64), (64, 128)],
        sparsity_in_ranges=[0.05, 0.05, 0.05],
        resolutions_in_ranges=[1.0, 1.0, 1.0],
        total_samples=256,
        start_time=0.0,
        stop_time=1.0,
    )
    encoder = el.FourierEncoder(params)
    t = np.linspace(0.0, 1.0, 256, endpoint=False)
    signal = np.sin(2 * np.pi * 8 * t) + 0.5 * np.sin(2 * np.pi * 24 * t)
    return encoder.encode(signal)


def _coordinate_demo() -> list[int]:
    params = el.CoordinateParameters(n=128, w=8, max_radius=4, dims=2, use_all_neighbors=False)
    encoder = el.CoordinateEncoder(params)
    return encoder.encode(([10, 20], 2))


def _geospatial_demo() -> list[int]:
    geo_params = el.GeospatialParameters(
        xy_scale=5.0,
        z_scale=1.0,
        timestep=1.0,
        max_radius=4,
        use_altitude=False,
    )
    coord_params = el.CoordinateParameters(n=128, w=8, max_radius=4, dims=2)
    encoder = el.GeospatialEncoder(geo_params=geo_params, coord_params=coord_params)
    return encoder.encode((7.5, -77.0365, 38.8977))


def _delta_demo() -> list[int]:
    params = el.DeltaEncoderParameters(size=256, sparsity=0.1)
    encoder = el.DeltaEncoder(params)
    return encoder.encode((12.5, 9.75))


def _build_demos() -> list[EncoderDemo]:
    return [
        EncoderDemo("rdse", _rdse_demo, "12.34 (float)"),
        EncoderDemo("scalar", _scalar_demo, "42 (int)"),
        EncoderDemo("category", _category_demo, '"dog" (str category)'),
        EncoderDemo("category_new", _category_new_demo, '"green" (str category)'),
        EncoderDemo("date", _date_demo, "datetime(2026-04-19 15:30:00)"),
        EncoderDemo("fourier", _fourier_demo, "1D sine mixture numpy array (256 samples)"),
        EncoderDemo("coordinate", _coordinate_demo, "([10, 20], radius=2)"),
        EncoderDemo("geospatial", _geospatial_demo, "(speed, lon, lat) = (7.5, -77.0365, 38.8977)"),
        EncoderDemo("delta", _delta_demo, "(12.5, 9.75) numeric pair"),
    ]


def _active_count(sdr: list[int]) -> int:
    return int(np.count_nonzero(np.asarray(sdr)))


def main() -> None:
    demos = _build_demos()
    demo_names = [demo.name for demo in demos]

    parser = argparse.ArgumentParser(
        description="Show each encoder type with valid input and resulting SDR plots."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="all",
        choices=["all", *demo_names],
        help="Run a single encoder demo or all demos.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip grapher.plot_sdr windows (useful for headless runs).",
    )
    args = parser.parse_args()

    selected = (
        demos if args.encoder == "all" else [next(d for d in demos if d.name == args.encoder)]
    )

    print("\\nEncoder Demo: valid input -> SDR")
    print("=" * 64)

    for demo in selected:
        print(f"\\n[{demo.name}] input: {demo.input_summary}")
        sdr = demo.build_and_encode()
        active = _active_count(sdr)
        print(f"[{demo.name}] SDR length={len(sdr)} active_bits={active}")

        if not args.no_plot:
            grapher.plot_sdr(
                sdr,
                title=(
                    f"{demo.name} encoder\\n"
                    f"input={demo.input_summary}\\n"
                    f"len={len(sdr)} active={active}"
                ),
            )

    print("\\nDone.")


if __name__ == "__main__":
    main()
