from __future__ import annotations

import copy
from datetime import datetime
from threading import Thread

import numpy as np
import pandas as pd

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

    def run(self):
        pass


class DateThread(Thread):

    def run(self):
        pass


class CategoryThread(Thread):

    def run(self):
        pass


class BatchEncoderHandler:
    __instance: BatchEncoderHandler | None = None

    def __new__(cls, input_data: pd.DataFrame | None = None) -> "BatchEncoderHandler":
        if cls.__instance is None:
            cls.__instance = super(BatchEncoderHandler, cls).__new__(cls)
        return cls.__instance

    def __init__(self, input_data: pd.DataFrame | None = None):
        self._data_frame = copy.deepcopy(input_data) if input_data is not None else pd.DataFrame()

    @classmethod
    def get_instance(cls) -> "BatchEncoderHandler":
        if cls.__instance is None:
            cls.__instance = BatchEncoderHandler()
        return cls.__instance

    def build_composite_sdr(self, input_data: pd.DataFrame, threads_per_column: int) -> list[SDR]:
        num_rows = len(input_data)
        column_sdrs = {}
        for col in input_data.columns:
            size = 2048
            column_sdrs[col] = [SDR([size]) for _ in range(num_rows)]

        threads: list[Thread] = []

        for col in input_data.columns:
            series = input_data[col].reset_index(drop=True)
            batches = np.array_split(series, threads_per_column)

            offset = 0
            for batch in batches:
                if len(batch) == 0:
                    continue
                # will add proper data checks eventually
                if not pd.api.types.is_numeric_dtype(series):
                    print(f"Skipping column '{col}' (not numeric)")
                    continue
                params = RDSEParameters(
                    size=2048,
                    active_bits=40,
                    sparsity=0,
                    radius=1,
                    resolution=0,
                    category=False,
                    seed=1,
                )
                thread = RdseThread(batch, params, column_sdrs[col], offset)

                threads.append(thread)
                thread.start()
                offset += len(batch)

        for t in threads:
            t.join()

        return column_sdrs["RSI"]
