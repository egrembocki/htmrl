import os
from importlib.metadata import PackageNotFoundError, version

from .encoder_layer.base_encoder import BaseEncoder, ParentDataClass
from .encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from .encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from .encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters

try:
    __version__ = version("psu_capstone")
except PackageNotFoundError:
    __version__ = "0.0.1"

PYTHONPATH = "/src"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easyData.xlsx")

__all__ = [
    "BaseEncoder",
    "ParentDataClass",
    "CategoryEncoder",
    "CategoryParameters",
    "RDSEParameters",
    "RandomDistributedScalarEncoder",
    "ScalarEncoder",
    "ScalarEncoderParameters",
]
