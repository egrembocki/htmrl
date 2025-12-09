# htm_core/synapse.py
from __future__ import annotations


class Synapse:
    """Proximal synapse (input space) used by Spatial Pooler only."""

    def __init__(self, source_input: int, permanence: float) -> None:
        self.source_input: int = source_input
        self.permanence: float = permanence
