import os
from importlib.metadata import version

from src.psu_capstone import agent_layer, encoder_layer, utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")


__version__ = version("psu_capstone")


__all__ = [
    "encoder_layer",
    "agent_layer",
    "utils",
]
