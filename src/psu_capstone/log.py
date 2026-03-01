import logging
import os
import sys
from typing import Any

logger = logging.getLogger("htmrl")
_handler = logging.StreamHandler(stream=sys.stdout)
_formatter = logging.Formatter(fmt="%(levelname)s [%(name)s]: %(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

if os.environ.get("DEBUG"):
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)


def get_logger(source: Any | None = None) -> logging.Logger:
    """Return the shared logger or a child logger named after the given source.

    Passing a class instance or class type generates a child logger whose
    name includes that class, so log lines show the originating class.
    """

    if source is None:
        return logger

    if isinstance(source, str):
        suffix = source
    elif isinstance(source, type):
        suffix = source.__name__
    else:
        suffix = source.__class__.__name__

    return logger.getChild(suffix)
