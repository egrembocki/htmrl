"""
Test suite for Segment class.

Segment is a collection of synapses on a Cell that learns coincident patterns
of activity in other cells. Each segment can represent a different learned pattern.

Key Properties:
  - Synapses: connections to other cells (distal synapses)
  - Permanence: strength of each synapse (0.0 to 1.0)
  - Connected threshold: permanence >= ~0.5 counts as connected
  - Active: whether segment is currently receiving active input
  - Matching: whether segment receives matching (but perhaps inactive) input

Temporal Learning Disabled:
  - Tests use non_temporal=True to avoid HTM temporal memory bug
  - Bug affects segment activation during temporal learning phase

Tests validate:
  1. Segment initialization and basic properties
  2. Synapse addition and removal
  3. Permanence updates (learning)
  4. Active/matching state computation
  5. Prediction contribution calculation
"""

import pytest

from psu_capstone.agent_layer.HTM import (
    ApicalSegment,
    ApicalSynapse,
    Cell,
    DistalSynapse,
    Field,
    Segment,
    Synapse,
)

"""Segment"""

"""Apical Segment"""
