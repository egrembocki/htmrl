import os
from importlib.metadata import version

from . import agent_layer, encoder_layer, input_layer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")


__version__ = version("psu_capstone")


__all__ = [
    "input_layer",
    "encoder_layer",
    "agent_layer",
]
