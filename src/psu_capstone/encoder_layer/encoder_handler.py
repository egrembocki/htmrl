"""Encoder Handler to build composite SDRs

This module provides the EncoderHandler class, which manages multiple encoder types
and dynamically selects the appropriate encoder for each column in a pandas DataFrame
based on its dtype. It builds composite Sparse Distributed Representations (SDRs)
from the encoded columns.
"""

import copy
from datetime import datetime
from typing import Any, List, Self

import numpy as np  # Add this import
import pandas as pd

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR


class EncoderHandler:
    """Handles multiple encoders to create composite SDRs.

    This class uses a singleton pattern to ensure only one instance exists.
    It dynamically selects the appropriate encoder for each DataFrame column
    based on its dtype and builds a composite SDR from the encoded columns.
    """

    __instance: Self | None = None

    def __new__(cls, input_data: pd.DataFrame) -> "EncoderHandler":
        """Implements the singleton pattern for EncoderHandler.

        Args:
            input_data (pd.DataFrame): Input data for encoder initialization.

        Returns:
            EncoderHandler: The singleton instance.
        """

        if cls.__instance is None:
            cls.__instance = super(EncoderHandler, cls).__new__(cls)

        return cls.__instance

    def __init__(self, input_data: pd.DataFrame):
        """Initializes the EncoderHandler with a DataFrame of input data.

        Args:
            input_data (pd.DataFrame): DataFrame containing input data.
        """
        self._data_frame = copy.deepcopy(input_data)
        self._encoders: List[BaseEncoder] = []

    def build_composite_sdr(self, input_data: pd.DataFrame) -> SDR:
        """Builds a composite SDR from multiple encoders based on the input data.

        For each column in the input DataFrame, selects an encoder based on the column's dtype,
        encodes the value, and concatenates the resulting SDRs into a single composite SDR.

        Args:
            input_data (pd.DataFrame): DataFrame containing input values for each encoder.

        Returns:
            SDR: Composite SDR built from all encoded columns.

        Raises:
            TypeError: If a column's value type is unsupported.
            ValueError: If no SDRs are created or an unexpected error occurs.
        """
        row = input_data.iloc[0]
        sdrs = []
        self._encoders = []  # Reset encoders for each call

        for col_name, value in row.items():
            # Select encoder based on dtype
            if isinstance(value, float) or isinstance(value, np.floating):
                encoder = RandomDistributedScalarEncoder(
                    RDSEParameters(
                        active_bits=5,
                        sparsity=0.0,
                        size=10,
                        radius=10.0,
                        resolution=0.0,
                        category=False,
                        seed=42,
                    ),
                    [2, 5],
                )
                sdr = SDR(encoder.dimensions)
                encoder.encode(float(value), sdr)

            elif isinstance(value, int) or isinstance(value, np.integer):
                encoder = ScalarEncoder(
                    ScalarEncoderParameters(
                        minimum=0.0,
                        maximum=100.0,
                        clip_input=True,
                        periodic=False,
                        active_bits=5,
                        sparsity=0.0,
                        size=10,
                        radius=0.0,
                        category=False,
                        resolution=0.0,
                        size_or_radius_or_category_or_resolution=0,
                        active_bits_or_sparsity=0,
                    ),
                    [2, 5],
                )
                sdr = SDR(encoder.dimensions)
                encoder.encode(float(value), sdr)

            elif isinstance(value, str):
                # Build category_list from all unique values in the column
                category_list = input_data[col_name].unique().tolist()
                encoder = CategoryEncoder(CategoryParameters(w=3, category_list=category_list))
                print(
                    f"Encoding string value '{value}' with category list: {encoder.parameters.category_list}"
                )
                sdr = SDR(encoder.dimensions)
                encoder.encode(value, sdr)

            elif isinstance(value, pd.Timestamp) or isinstance(value, datetime):
                encoder = DateEncoder(
                    DateEncoderParameters(
                        season_width=0,
                        season_radius=91.5,
                        day_of_week_width=3,
                        day_of_week_radius=1.0,
                        weekend_width=3,
                        holiday_width=0,
                        holiday_dates=[[12, 25]],
                        time_of_day_width=3,
                        time_of_day_radius=4.0,
                        custom_width=0,
                        custom_days=[],
                        verbose=False,
                    ),
                    [1, 45],
                )
                # Use encoder.size for SDR size
                sdr = SDR(encoder.dimensions)
                encoder.encode(value, sdr)

            else:
                raise TypeError(f"Unsupported value type for encoder: {type(value)}")

            print(f"Column '{col_name}' encoded sparse SDR:", sdr.get_sparse())
            if sdr.get_sparse() == []:
                print(
                    f"Warning: Encoding failed for column '{col_name}' with value '{value}' and encoder '{type(encoder).__name__}'"
                )
                continue  # Skip this column
            self._encoders.append(copy.deepcopy(encoder))
            sdrs.append(copy.deepcopy(sdr))

        if not sdrs:
            raise ValueError("No SDRs were created from the input data.")

        if len(sdrs) >= 2:
            # Flatten all SDRs to 1D before concatenation
            flat_sdrs = []
            for sdr in sdrs:
                # If SDR is not 1D, flatten it
                if len(sdr.dimensions) != 1:
                    flat_sdr = SDR([sdr.size])
                    flat_sdr.set_sparse(sdr.get_sparse())
                    flat_sdrs.append(flat_sdr)
                else:
                    flat_sdrs.append(sdr)
            total_size = sum(sdr.size for sdr in flat_sdrs)
            union_sdr = SDR([total_size])
            union_sdr.concatenate(flat_sdrs, axis=0)
            return union_sdr
        elif len(sdrs) == 1:
            return copy.deepcopy(sdrs[0])
        else:
            raise ValueError("Unexpected error in building composite SDR.")


if __name__ == "__main__":
    """Smoke test for EncoderHandler.

    Creates a sample DataFrame with various column types, initializes the EncoderHandler,
    builds a composite SDR, and prints its sparse representation and size.
    """

    df = pd.DataFrame(
        [
            {
                "float_col": float(3.14),
                "int_col": int(42),
                "str_col": str("B"),
                "date_col": datetime(2023, 12, 25),
            }
        ]
    )

    handler = EncoderHandler(df)
    composite_sdr = handler.build_composite_sdr(df)
    print("Composite SDR sparse representation:", composite_sdr.get_sparse())
    print("Composite SDR size:", composite_sdr.size)
