import logging
import os
import sys

logger = logging.getLogger("htmrl")
_handler = logging.StreamHandler(stream=sys.stdout)
_formatter = logging.Formatter(fmt="%(levelname)s: %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

if os.environ.get("DEBUG"):
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
