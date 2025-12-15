from __future__ import annotations

import copy
from datetime import datetime
from multiprocessing import Manager, Process
from threading import Thread
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR


class RdseThread(Thread):
    def __init__(
        self, column_data: pd.Series, params: RDSEParameters, output: list[SDR], row_offset: int
    ):
        super().__init__()
        self._column_data = column_data
        self._encoder = RandomDistributedScalarEncoder(params)
        self._output = output
        self._row_offset = row_offset

    def run(self):
        for i in range(len(self._column_data)):
            value = self._column_data.iloc[i]

            # fresh sdr instance per row
            sdr_instance = SDR([self._encoder.size])
            self._encoder.encode(value, sdr_instance)
            self._output[self._row_offset + i] = sdr_instance

            """
            #debug prints if you want
            active_bits = sdr_instance.get_sum()
            sparsity_fraction = sdr_instance.get_sparsity()
            print(f"Row {self._row_offset + i}: active_bits={active_bits}, sparsity={sparsity_fraction:.4f}")
            """


class ScalarThread(Thread):
    def __init__(
        self,
        column_data: pd.Series,
        params: ScalarEncoderParameters,
        output: list[SDR],
        row_offset: int,
    ):
        super().__init__()
        self._column_data = column_data
        self._encoder = ScalarEncoder(params)
        self._output = output
        self._row_offset = row_offset

    def run(self):
        for i in range(len(self._column_data)):
            value = self._column_data.iloc[i]

            # fresh sdr instance per row
            sdr_instance = SDR([self._encoder.size])
            self._encoder.encode(value, sdr_instance)
            self._output[self._row_offset + i] = sdr_instance

            """
            #debug prints if you want
            active_bits = sdr_instance.get_sum()
            sparsity_fraction = sdr_instance.get_sparsity()
            print(f"Row {self._row_offset + i}: active_bits={active_bits}, sparsity={sparsity_fraction:.4f}")
            """


class DateThread(Thread):
    def __init__(
        self,
        column_data: pd.Series,
        params: DateEncoderParameters,
        output: list[SDR],
        row_offset: int,
    ):
        super().__init__()
        self._column_data = column_data
        self._encoder = DateEncoder(params)
        self._output = output
        self._row_offset = row_offset

    def run(self):
        for i in range(len(self._column_data)):
            value = self._column_data.iloc[i]

            # fresh sdr instance per row
            sdr_instance = SDR([self._encoder.size])
            self._encoder.encode(value, sdr_instance)
            self._output[self._row_offset + i] = sdr_instance

            """
            #debug prints if you want
            active_bits = sdr_instance.get_sum()
            sparsity_fraction = sdr_instance.get_sparsity()
            print(f"Row {self._row_offset + i}: active_bits={active_bits}, sparsity={sparsity_fraction:.4f}")
            """


class CategoryThread(Thread):
    def __init__(
        self, column_data: pd.Series, params: CategoryEncoder, output: list[SDR], row_offset: int
    ):
        super().__init__()
        self._column_data = column_data
        self._encoder = CategoryEncoder(params)
        self._output = output
        self._row_offset = row_offset

    def run(self):
        for i in range(len(self._column_data)):
            value = self._column_data.iloc[i]

            # fresh sdr instance per row
            sdr_instance = SDR([self._encoder.size])
            self._encoder.encode(value, sdr_instance)
            self._output[self._row_offset + i] = sdr_instance

            """
            #debug prints if you want
            active_bits = sdr_instance.get_sum()
            sparsity_fraction = sdr_instance.get_sparsity()
            print(f"Row {self._row_offset + i}: active_bits={active_bits}, sparsity={sparsity_fraction:.4f}")
            """


class BatchEncoderHandler:
    __instance: BatchEncoderHandler | None = None

    def __new__(cls, input_data: pd.DataFrame | None = None) -> "BatchEncoderHandler":
        if cls.__instance is None:
            cls.__instance = super(BatchEncoderHandler, cls).__new__(cls)
        return cls.__instance

    def __init__(self, input_data: pd.DataFrame | None = None):
        self._data_frame = copy.deepcopy(input_data) if input_data is not None else pd.DataFrame()
        self._rdse_params = RDSEParameters(
            size=2048,
            active_bits=40,
            sparsity=0,
            radius=1,
            resolution=0,
            category=False,
            seed=1,
        )
        self._scalar_params = ScalarEncoderParameters(
            minimum=0,
            maximum=100,
            clip_input=True,
            periodic=False,
            active_bits=40,
            sparsity=0.0,
            size=2048,
            radius=0.0,
            category=False,
            resolution=0.0,
        )
        self._date_params = DateEncoderParameters(
            season_width=0,
            season_radius=91.5,
            day_of_week_width=7,
            day_of_week_radius=1.0,
            weekend_width=2,
            holiday_width=4,
            holiday_dates=[[12, 25], [1, 1], [7, 4], [11, 11]],
            time_of_day_width=24,
            time_of_day_radius=1.0,
            custom_width=0,
            custom_days=[],
        )
        category_list = input_data.columns.unique().tolist()
        self._category_params = CategoryParameters(w=3, category_list=category_list)

    @classmethod
    def get_instance(cls) -> "BatchEncoderHandler":
        if cls.__instance is None:
            cls.__instance = BatchEncoderHandler()
        return cls.__instance

    def _build_dict_list_sdr(
        self, input_data: pd.DataFrame, threads_per_column: int
    ) -> dict[list[SDR]]:
        num_rows = len(input_data)
        column_sdrs = {}
        # The sizes here may need to be dynamic based on parameter settings
        for col in input_data.columns:
            size = 2048
            column_sdrs[col] = [SDR([size]) for _ in range(num_rows)]

        threads: list[Thread] = []

        for col in input_data.columns:
            series = input_data[col].reset_index(drop=True)
            batches = np.array_split(series, threads_per_column)
            scalartrue = False
            offset = 0
            for batch in batches:
                if len(batch) == 0:
                    continue
                # will add proper data checks eventually
                # if not pd.api.types.is_numeric_dtype(series):
                #    print(f"Skipping column '{col}' (not numeric)")
                #    continue
                if pd.api.types.is_numeric_dtype(series) or (
                    isinstance(series.dropna().iloc[0], (int, float))
                ):
                    params = self._rdse_params
                    thread = RdseThread(batch, params, column_sdrs[col], offset)
                elif pd.api.types.is_string_dtype(series):
                    # category_list = input_data[col].unique().tolist()
                    params = self._category_params
                    thread = CategoryThread(batch, params, column_sdrs[col], offset)
                elif pd.api.types.is_datetime64_any_dtype(series):
                    params = self._date_params
                    thread = DateThread(batch, params, column_sdrs[col], offset)
                elif scalartrue and pd.api.types.is_numeric_dtype(series):
                    params = self._scalar_params
                    thread = ScalarThread(batch, params, column_sdrs[col], offset)
                else:
                    print(f"Not any type. Column '{col}' dtype: {series.dtypes}")
                    thread = None

                if thread is not None:
                    threads.append(thread)
                    thread.start()
                    offset += len(batch)

        for t in threads:
            t.join()

        return column_sdrs

    def build_composite_sdr(self, input_data, threads_per_column: int) -> list[SDR]:
        if isinstance(input_data, pd.DataFrame):
            column_sdrs = self._build_dict_list_sdr(input_data, threads_per_column)
        else:
            column_sdrs = input_data

        num_rows = len(next(iter(column_sdrs.values())))
        composite_sdrs = []

        for i in range(num_rows):
            row_sdrs = [col_sdrs[i] for col_sdrs in column_sdrs.values()]

            flat_sdrs = []
            for s in row_sdrs:
                if len(s.dimensions) != 1:
                    flat = SDR([s.size])
                    flat.set_sparse(s.get_sparse())
                    flat_sdrs.append(flat)
                else:
                    flat_sdrs.append(s)

            total_size = sum(s.size for s in flat_sdrs)
            if len(flat_sdrs) >= 2:
                composite = SDR([total_size])
                composite.concatenate(flat_sdrs, axis=0)
            else:
                composite = copy.deepcopy(flat_sdrs[0])

            composite_sdrs.append(composite)

        return composite_sdrs

    def create_knn_composite_sdr(
        self, input_data: pd.DataFrame, column_sdrs: dict[list[SDR]] | None, threads_per_column: int
    ) -> Tuple[list[SDR], NearestNeighbors]:
        """Takes in the dataframe returns a composite list of our sdrs as well as a kNN model to predict SDR values."""
        """Build our dict of lists of sdrs to use for kNN"""
        column_sdrs = self._build_dict_list_sdr(input_data, threads_per_column)

        """Make the kNN model."""
        knn = NearestNeighbors(5)

        """Build the composite not that the model is finished."""
        row_sdrs = self.build_composite_sdr(column_sdrs, threads_per_column)

        """Return our composite SDRs and the model for them."""
        return row_sdrs, knn

    def set_category_encoder_parameters(self, params: CategoryParameters):
        self._category_params = params

    def set_rdse_encoder_parameters(self, params: RDSEParameters):
        self._rdse_params = params

    def set_scalar_encoder_parameters(self, params: ScalarEncoderParameters):
        self._scalar_params = params

    def set_date_encoder_parameters(self, params: DateEncoderParameters):
        self._date_params = params

    def choose_custom_column_encoding(self, custom_encoding: dict[str, str]):
        pass
