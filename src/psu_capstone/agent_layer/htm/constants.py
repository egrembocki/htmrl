# htm_core/constants.py
from __future__ import annotations

# Spatial Pooler constants
CONNECTED_PERM = 0.5  # Permanence threshold for connected proximal synapse
MIN_OVERLAP = 3  # Minimum overlap to be considered during inhibition
PERMANENCE_INC = 0.01
PERMANENCE_DEC = 0.01
DESIRED_LOCAL_ACTIVITY = 10

# Temporal Memory constants
SEGMENT_ACTIVATION_THRESHOLD = 3  # Active connected distal synapses required for prediction
SEGMENT_LEARNING_THRESHOLD = 3  # For best matching segment selection (reserved)
INITIAL_DISTAL_PERM = 0.21  # Initial permanence for new distal synapses
NEW_SYNAPSE_MAX = 6  # New distal synapses to add on reinforcement
