# htm_core/synapse.py
from __future__ import annotations


class Synapse:
    """Proximal synapse (input space) used by Spatial Pooler only.

    Arguments:
        source_input: int -- index of input bit this synapse connects to in the input space
        permanence: float -- initial permanence value

    """

    def __init__(self, source_input: int, permanence: float) -> None:
        self.source_input: int = source_input
        self.permanence: float = permanence


if __name__ == "__main__":
    syn = Synapse(0, 0.5)
    assert isinstance(syn, Synapse)
