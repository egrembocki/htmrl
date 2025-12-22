import copy
import os
import warnings
from datetime import datetime

import pandas as pd
import pytest

from psu_capstone.encoder_layer.batch_encoder_handler import BatchEncoderHandler
from psu_capstone.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easyData.xlsx")

df = pd.DataFrame(
    [
        {
            "float_col": float(3.14),  # rdse
            "int_col": int(42),  # scalar
            "str_col": str("B"),  # category
            "date_col": datetime(2023, 12, 25),  # date
        }
    ]
)


@pytest.fixture
def handler() -> BatchEncoderHandler:
    """Fixture to create an EncoderHandler with multiple encoders"""

    df = pd.DataFrame(
        [
            {
                "float_col": float(3.14),  # rdse
                "int_col": int(42),  # scalar
                "str_col": str("B"),  # category
                "date_col": datetime(2023, 12, 25),  # date
            }
        ]
    )

    handler = BatchEncoderHandler(df)

    return handler


def test_handler_singleton(handler: BatchEncoderHandler):
    """
    Tests to make sure the handler is a singleton.
    """
    test_input = handler._data_frame

    h1 = handler
    h2 = BatchEncoderHandler(test_input)

    assert h1 is h2


def test_dataframe_composite():
    """
    This test makes sure our built composite sdr is identical to if
    we built it in separate sdrs. Concatenation was employed from the
    sdr to enable a proper composite sdr from the different encodings.
    """

    # this suppresses a pandas future warning of a method we do not use here.
    warnings.simplefilter(action="ignore", category=FutureWarning)
    handler = BatchEncoderHandler(df)
    test_input = handler._data_frame

    rdseparams = RDSEParameters(100, 2, 0, 0, 1, False, 1)
    handler.set_rdse_encoder_parameters(params=rdseparams)
    categoryparams = CategoryParameters(3, ["B"], rdse_used=False)
    handler.set_category_encoder_parameters(params=categoryparams)
    dateparams = DateEncoderParameters(
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
        rdse_used=False,
    )
    handler.set_date_encoder_parameters(params=dateparams)
    composite_sdrs = handler.build_composite_sdr(test_input, 4)

    composite_test = []

    e1 = RandomDistributedScalarEncoder(rdseparams)
    sdrfloat = SDR([rdseparams.size])
    e1.encode(3.14, sdrfloat)
    composite_test.append(sdrfloat)

    sdrint = SDR([rdseparams.size])
    e1.encode(42, sdrint)
    composite_test.append(sdrint)

    e2 = CategoryEncoder(categoryparams)
    sdrstring = SDR([6])
    e2.encode("B", sdrstring)
    composite_test.append(sdrstring)

    e3 = DateEncoder(dateparams)
    sdrdate = SDR([e3.size])
    date_value = df["date_col"].iloc[0]
    e3.encode(date_value, sdrdate)
    composite_test.append(sdrdate)

    composite = SDR([e1.size + e1.size + e2.size + e3.size])
    composite.concatenate(composite_test, 0)

    expected = composite
    actual = composite_sdrs[0]

    assert expected.size == actual.size
    assert expected.get_dimensions() == actual.get_dimensions()
    assert expected.get_sparse() == actual.get_sparse()


def test_individual_column_sdrs():
    """
    This tests to make sure our dicstionary encoding allows us to access every
    column with a simple line. This retrieves all of the sdrs for that respective
    column. Each should have 4 sdrs as we have 4 rows in the example data.
    """

    # this suppresses a pandas future warning of a method we do not use here.
    warnings.simplefilter(action="ignore", category=FutureWarning)
    df1 = pd.DataFrame(
        [
            {"float_col": 3.14, "int_col": 42, "str_col": "B", "date_col": datetime(2023, 12, 25)},
            {"float_col": 5.4, "int_col": 21, "str_col": "C", "date_col": datetime(2023, 12, 26)},
            {"float_col": 6.7, "int_col": 10, "str_col": "D", "date_col": datetime(2023, 12, 27)},
            {"float_col": 12.4, "int_col": 5, "str_col": "E", "date_col": datetime(2023, 12, 28)},
        ]
    )

    handler = BatchEncoderHandler(df1)

    rdseparams = RDSEParameters(100, 2, 0, 0, 1, False, 1)
    handler.set_rdse_encoder_parameters(params=rdseparams)
    categoryparams = CategoryParameters(3, ["B"], rdse_used=False)
    handler.set_category_encoder_parameters(params=categoryparams)
    dateparams = DateEncoderParameters(
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
        rdse_used=False,
    )
    handler.set_date_encoder_parameters(params=dateparams)

    dictionary_sdrs = handler._build_dict_list_sdr(df1, 4)
    assert len(dictionary_sdrs["float_col"]) == 4
    assert len(dictionary_sdrs["int_col"]) == 4
    assert len(dictionary_sdrs["str_col"]) == 4
    assert len(dictionary_sdrs["date_col"]) == 4
    """
    #Optional Printing:
    print(dictionary_sdrs["float_col"])
    print(dictionary_sdrs["int_col"])
    print(dictionary_sdrs["str_col"])
    print(dictionary_sdrs["date_col"])
    """


def test_custom_encoding():
    """
    This tests that we can tell the handler that we want the float_col to be encoded
    as a category instead of an rdse.
    """

    # this suppresses a pandas future warning of a method we do not use here.
    warnings.simplefilter(action="ignore", category=FutureWarning)
    df1 = pd.DataFrame(
        [
            {"float_col": 3.14, "int_col": 42, "str_col": "B", "date_col": datetime(2023, 12, 25)},
            {"float_col": 5.4, "int_col": 21, "str_col": "C", "date_col": datetime(2023, 12, 26)},
            {"float_col": 6.7, "int_col": 10, "str_col": "D", "date_col": datetime(2023, 12, 27)},
            {"float_col": 12.4, "int_col": 5, "str_col": "E", "date_col": datetime(2023, 12, 28)},
        ]
    )

    handler = BatchEncoderHandler(df1)

    handler.choose_custom_column_encoding({"float_col": "category"})
    rdseparams = RDSEParameters(100, 2, 0, 0, 1, False, 1)
    handler.set_rdse_encoder_parameters(params=rdseparams)
    unique_column_values = df1["float_col"].astype(str).unique().tolist()
    categoryparams = CategoryParameters(3, unique_column_values, rdse_used=False)
    handler.set_category_encoder_parameters(params=categoryparams)
    dateparams = DateEncoderParameters(
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
        rdse_used=False,
    )
    handler.set_date_encoder_parameters(params=dateparams)
    dictionary_sdrs = handler._build_dict_list_sdr(df1, 4)
    float_sdrs = dictionary_sdrs["float_col"]

    # Since we have 4 categories plus the 1 designated unknown, we are width 3, this is 3*5=15 size.
    # 3 active bits per value since every value is unique and width is 3.
    for sdr in float_sdrs:
        assert sdr.size == 15
        assert sdr.dimensions == [15]
        assert len(sdr.get_sparse()) == 3
    # print(float_sdrs)
