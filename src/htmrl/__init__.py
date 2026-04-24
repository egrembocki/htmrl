"""PSU Capstone project for Hierarchical Temporal Memory.

Attributes:
    __version__: Version string of the package.
    PROJECT_ROOT: Root directory of the project.
    DATA_PATH: Default path to the sample data file.
    PYTHONPATH: Python path for the project.
"""

import os
from importlib.metadata import PackageNotFoundError, version

import htmrl.agent_layer as agent_tools
import htmrl.encoder_layer as encoder_tools
import htmrl.environment as environment_tools
import htmrl.input_layer as input_tools

try:
    __version__ = version("psu_capstone")
except PackageNotFoundError:
    __version__ = "0.0.1"

PYTHONPATH = "/src"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easyData.xlsx")


__all__ = ["agent_tools", "encoder_tools", "environment_tools", "input_tools"]
