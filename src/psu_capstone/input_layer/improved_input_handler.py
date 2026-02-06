"""Compatibility wrapper for the improved InputHandler.

Historically this module duplicated `input_handler.InputHandler` with additional
experiments. The core implementation now lives in `input_handler`, and this
module simply re-exports it to preserve existing imports.
"""

from __future__ import annotations

from psu_capstone.input_layer.input_handler import InputHandler

__all__ = ["InputHandler"]
