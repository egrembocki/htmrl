"""HTM proximal synapse: links input-space bits to columns and models strength via permanence.

Context:
- Spatial Pooler (SP) uses proximal synapses to connect columns to the input space.
- Permanence (typically in [0.0, 1.0]) represents synaptic strength; if it exceeds
  the connection threshold (CONNECTED_PERM), the synapse is considered connected.
- Column overlap is computed as the count of active input bits arriving through
  connected synapses, often scaled by a boosting factor. Learning adjusts permanence
  up or down depending on whether the input bit was active, stabilizing sparse
  distributed representations.
"""

# htm_core/synapse.py
from __future__ import annotations


class Synapse:
    """Proximal synapse (input space) used by Spatial Pooler only.

    Args:
        source_input: The integer index of the input bit in the combined SDR.
        permanence: Initial strength (0.0-1.0 typical). Values above CONNECTED_PERM
            mark the synapse as connected for overlap computations.
    """

    def __init__(self, source_input: int, permanence: float) -> None:
        self.source_input: int = source_input
        self.permanence: float = permanence


if __name__ == "__main__":

    syn = Synapse(0, 0.5)
    assert isinstance(syn, Synapse)
