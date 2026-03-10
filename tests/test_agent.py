"""
tests.test_agent

Test suite for Agent layer components and integration.

The Agent class orchestrates the entire HTM learning pipeline, coordinating:
- Input data loading and preprocessing
- Encoder layer (converting data to SDRs)
- HTM Brain (temporal memory, sequence learning)
- Output field generation and motor actions

Tests validate:
- Agent initialization with configuration
- Data pipeline integration (input → encoder → HTM)
- Temporal learning and sequence prediction
- Active columns and confidence tracking
- Integration between all major components

These tests ensure the Agent correctly orchestrates all lower-level components
into a cohesive learning and prediction system.
"""

import pytest

from psu_capstone.agent_layer.agent import Agent
