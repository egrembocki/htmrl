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
    1) Integration tests: multi-component collaboration tests.
    2) Unit tests: default for isolated behavior tests.
    """

    del config  # unused by design; hook signature requires it

    for item in items:
        path = _relative_test_path(item)
        nodeid = item.nodeid

        # Integration: explicit integration folder and selected cross-component tests.
        if "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
            continue
        if path.endswith("tests/test_agent_server.py"):
            item.add_marker(pytest.mark.integration)
            continue

        # Default classification for isolated component behavior tests.
        item.add_marker(pytest.mark.unit)
