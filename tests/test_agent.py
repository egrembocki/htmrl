"""Tests suite for the Agent layer components."""

import pytest

from psu_capstone.agent_layer.agent import Agent


@pytest.fixture
def agent():
    """Fixture to create an Agent instance."""
    return Agent()
