#!/usr/bin/env python3
"""Validate that repository tests include standardized ``# commit: ...`` annotations.

This checker scans ``tests/`` for test functions/methods (``def test_*``) and
verifies each has an allowed commit annotation comment immediately above its
decorator block/function definition.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

TEST_ROOT = Path("tests")
ALLOWED_TAGS = {
    "unit test",
    "integration test",
    "system/integration test",
    "system/acceptance test",
}


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    function_name: str
    reason: str


def _extract_commit_tag(line: str) -> str | None:
    marker = "# commit:"
    if marker not in line:
        return None
    return line.split(marker, 1)[1].strip().lower()


def _find_commit_annotation(
    lines: list[str], definition_start_line: int, definition_line: int
) -> str | None:
    """Return commit tag if present near a test definition.

    Supports both common layouts:
    1) annotation above decorators
    2) annotation between decorators and ``def``
    """

    # Look in the block from first decorator to line before ``def``.
    for idx in range(definition_start_line - 1, definition_line - 1):
        tag = _extract_commit_tag(lines[idx].strip())
        if tag is not None:
            return tag

    # Fallback: nearest non-empty line before the decorator block.
    idx = definition_start_line - 2
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx >= 0:
        return _extract_commit_tag(lines[idx].strip())

    return None


def _test_nodes(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            nodes.append(node)
    return nodes


def collect_violations() -> list[Violation]:
    violations: list[Violation] = []

    for path in sorted(TEST_ROOT.rglob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source)

        for node in _test_nodes(tree):
            start_line = min((d.lineno for d in node.decorator_list), default=node.lineno)
            annotation = _find_commit_annotation(lines, start_line, node.lineno)

            if annotation is None:
                violations.append(
                    Violation(path, node.lineno, node.name, "missing '# commit:' annotation")
                )
                continue

            if annotation not in ALLOWED_TAGS:
                violations.append(
                    Violation(
                        path,
                        node.lineno,
                        node.name,
                        f"unsupported tag '{annotation}'",
                    )
                )

    return violations


def main() -> int:
    violations = collect_violations()

    if not violations:
        print("OK: all discovered tests have valid # commit annotations.")
        return 0

    print("Found invalid/missing commit annotations:")
    for violation in violations:
        print(
            f"- {violation.path}:{violation.lineno} ({violation.function_name}) -> {violation.reason}"
        )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
