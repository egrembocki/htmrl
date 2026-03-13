"""HTM constants for Spatial Pooler (SP) and Temporal Memory (TM).

Spatial Pooler (SP):
- CONNECTED_PERM: Permanence threshold above which a proximal synapse is considered connected.
- MIN_OVERLAP: Minimum raw overlap required before boost is applied; below this, overlap is 0.
- PERMANENCE_INC/DEC: Learning rates for increasing/decreasing proximal synapse permanence.
- DESIRED_LOCAL_ACTIVITY: Target number of active columns within an inhibition neighborhood (k for k-th score).

Temporal Memory (TM):
- SEGMENT_ACTIVATION_THRESHOLD: Number of active, connected distal synapses required for a segment to activate (cell becomes predictive).
- SEGMENT_LEARNING_THRESHOLD: Reserved threshold used in selecting best-matching segments during learning.
- INITIAL_DISTAL_PERM: Initial permanence assigned to newly grown distal synapses.
- NEW_SYNAPSE_MAX: Maximum number of new distal synapses to add to a segment during reinforcement.
"""

# htm_core/constants.py
from __future__ import annotations

# Spatial Pooler constants
CONNECTED_PERM = 0.5  # Synapse permanence > threshold => connected
MIN_OVERLAP = 3  # Raw overlap must meet/exceed this to count
PERMANENCE_INC = 0.01  # Increment for proximal permanence during learning
PERMANENCE_DEC = 0.01  # Decrement for proximal permanence during learning
DESIRED_LOCAL_ACTIVITY = 10  # k for local competition (k-th neighbor overlap)

# Temporal Memory constants
SEGMENT_ACTIVATION_THRESHOLD = 3  # Active connected distal synapses required for prediction
SEGMENT_LEARNING_THRESHOLD = 3  # Reserved: threshold for best-matching segment selection
INITIAL_DISTAL_PERM = 0.21  # Initial permanence for new distal synapses
NEW_SYNAPSE_MAX = 6  # Number of new distal synapses to add on reinforcement
