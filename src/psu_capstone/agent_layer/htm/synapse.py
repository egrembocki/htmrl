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
    """HTM proximal synapse with an input index and a permanence value.

    Usage:
    - Owned by Columns to form the proximal receptive field.
    - During SP compute, only synapses whose permanence > CONNECTED_PERM contribute
      to overlap if their source input bit is active.
    - During SP learning, permanence is incremented/decremented to reinforce correct
      matches and discourage incorrect ones.

    Attributes:
    - source_input: Index into the input SDR this synapse monitors.
    - permanence: Synaptic strength; compared to CONNECTED_PERM to determine connectivity.
    """

    def __init__(self, source_input: int, permanence: float) -> None:
        """Create a synapse referencing an input bit index with an initial permanence.

        Parameters:
        - source_input: The integer index of the input bit in the combined SDR.
        - permanence: Initial strength (0.0–1.0 typical). Values above CONNECTED_PERM
          mark the synapse as connected for overlap computations.
        """
        self.source_input: int = source_input
        self.permanence: float = permanence
