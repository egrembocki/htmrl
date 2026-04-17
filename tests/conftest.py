"""Centralized pytest taxonomy assignment for test levels.

This module assigns exactly one of: unit, integration, system, acceptance
for every collected test item.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _relative_test_path(item: pytest.Item) -> str:
    """Return a stable, workspace-relative test path for classification."""
    return Path(str(item.fspath)).as_posix()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Assign one taxonomy marker to each collected test item.

    Rules are ordered from most specific to most general:
    1) Acceptance tests: user-facing/demo acceptance scenarios.
    2) System tests: cross-subsystem end-to-end tests.
    3) Integration tests: multi-component collaboration tests.
    4) Unit tests: default for isolated behavior tests.
    """

    del config  # unused by design; hook signature requires it

    for item in items:
        path = _relative_test_path(item)
        nodeid = item.nodeid

        # Acceptance: full demo artifact generation checked from a user workflow perspective.
        if path.endswith("tests/test_demo_driver_full_demo.py"):
            item.add_marker(pytest.mark.acceptance)
            continue

        # System: end-to-end behavior spanning major runtime subsystems.
        if path.endswith("tests/test_cartpole_brain_training.py"):
            item.add_marker(pytest.mark.system)
            continue
        if nodeid.endswith(
            "tests/integration/test_input_to_encoder.py::test_sine_wave_through_input_handler"
        ):
            item.add_marker(pytest.mark.system)
            continue

        # Integration: explicit integration folder and adapter/environment bridge tests.
        if "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
            continue
        if path.endswith("tests/test_agent_server.py"):
            item.add_marker(pytest.mark.integration)
            continue
        if path.endswith("tests/test_env_adapter.py"):
            item.add_marker(pytest.mark.integration)
            continue
        if path.endswith("tests/test_fin_gym.py"):
            item.add_marker(pytest.mark.integration)
            continue

        # Default classification for isolated component behavior tests.
        item.add_marker(pytest.mark.unit)
