"""HTM proximal synapse definitions: link input-space bits to mini-columns and track permanence/duty-cycle statistics."""

# htm_core/synapse.py
from __future__ import annotations


class Synapse:
    """Represents an HTM proximal synapse whose permanence models Hebbian-like strength and is compared to the connection threshold."""

    def __init__(self, source_input: int, permanence: float) -> None:
        """Create a synapse referencing an input bit index with an initial permanence (usually 0.0–1.0)."""
        self.source_input: int = source_input
        self.permanence: float = permanence
