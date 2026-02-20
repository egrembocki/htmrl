import os
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("psu_capstone")
except PackageNotFoundError:
    __version__ = "0.0.1"


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easyData.xlsx")
