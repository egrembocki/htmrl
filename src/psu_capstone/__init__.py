"""PSU Capstone project for Hierarchical Temporal Memory.

Attributes:
    __version__: Version string of the package.
    PROJECT_ROOT: Root directory of the project.
    DATA_PATH: Default path to the sample data file.
    PYTHONPATH: Python path for the project.
"""

import os
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("psu_capstone")
except PackageNotFoundError:
    __version__ = "0.0.1"

PYTHONPATH = "/src"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easyData.xlsx")
