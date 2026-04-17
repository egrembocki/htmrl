#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path

TEST_ROOT = Path("tests")
OUT_CSV = Path("docs/test_audit_ts_tc_report.csv")
OUT_MD = Path("docs/test_audit_ts_tc_report.md")

TS_RE = re.compile(r"\bTS\s*[- ]?\s*(\d+)\b", re.IGNORECASE)
TC_RE = re.compile(r"\bTC\s*[- ]?\s*(\d+)\b", re.IGNORECASE)
TI_RE = re.compile(r"\bTI\s*[- ]?\s*(\d+)\b", re.IGNORECASE)

TYPE_FROM_COMMIT = {
    "unit test": "unit",
    "integration test": "integration",
    "system test": "system",
    "system/integration test": "system",
    "system/acceptance test": "acceptance",
    "acceptance test": "acceptance",
}


@dataclass
class Row:
    file: str
    line: int
    test_name: str
    ts_id: str
    tc_id: str
    ti_id: str
    commit_tag: str
    test_type: str
    type_source: str


def extract_commit_tag(line: str) -> str | None:
    m = re.search(r"#\s*commit\s*:\s*(.+)$", line, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip().lower()


def infer_type(path: Path, test_name: str) -> tuple[str, str]:
    p = str(path).lower()
    n = test_name.lower()
    if "/integration/" in p:
        return "integration", "inferred:path(/integration/)"
    if "accept" in n or "full_demo" in n or "end_to_end" in n:
        return "acceptance", "inferred:test_name"
    if any(k in p for k in ["env_adapter", "fin_gym", "trainer", "agent_server", "demo_driver"]):
        return "system", "inferred:file_name"
    return "unit", "inferred:default"


def extract_ids(text: str) -> tuple[str, str, str]:
    ts = TS_RE.findall(text)
    tc = TC_RE.findall(text)
    ti = TI_RE.findall(text)
    return (f"TS-{ts[0]}" if ts else "", f"TC-{tc[0]}" if tc else "", f"TI-{ti[0]}" if ti else "")


def test_nodes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            yield node


def main() -> None:
    rows: list[Row] = []

    for path in sorted(TEST_ROOT.rglob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source)

        suite_window = "\n".join(lines[:80])
        suite_ts, _, _ = extract_ids(suite_window)

        for node in sorted(test_nodes(tree), key=lambda n: n.lineno):
            start_line = min((d.lineno for d in node.decorator_list), default=node.lineno)

            pre_start = max(0, start_line - 6)
            post_end = min(len(lines), node.lineno + 2)
            nearby = "\n".join(lines[pre_start:post_end])
            ts_matches = TS_RE.findall(nearby)
            tc_matches = TC_RE.findall(nearby)
            ti_matches = TI_RE.findall(nearby)
            ts_id = f"TS-{ts_matches[-1]}" if ts_matches else ""
            tc_id = f"TC-{tc_matches[-1]}" if tc_matches else ""
            ti_id = f"TI-{ti_matches[-1]}" if ti_matches else ""

            if not ts_id:
                ts_id = suite_ts

            commit_tag = ""
            for idx in range(start_line - 1, node.lineno):
                tag = extract_commit_tag(lines[idx].strip())
                if tag:
                    commit_tag = tag
                    break
            if not commit_tag:
                i = start_line - 2
                while i >= 0 and not lines[i].strip():
                    i -= 1
                if i >= 0:
                    commit_tag = extract_commit_tag(lines[i].strip()) or ""

            if commit_tag in TYPE_FROM_COMMIT:
                test_type = TYPE_FROM_COMMIT[commit_tag]
                type_source = f"commit:{commit_tag}"
            else:
                test_type, type_source = infer_type(path, node.name)

            rows.append(
                Row(
                    file=str(path),
                    line=node.lineno,
                    test_name=node.name,
                    ts_id=ts_id or "—",
                    tc_id=tc_id or "—",
                    ti_id=ti_id or "—",
                    commit_tag=commit_tag or "—",
                    test_type=test_type,
                    type_source=type_source,
                )
            )

    rows.sort(key=lambda r: (r.test_type, r.file, r.line, r.test_name))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "file",
                "line",
                "test_name",
                "ts_id",
                "tc_id",
                "ti_id",
                "commit_tag",
                "test_type",
                "type_source",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.file,
                    r.line,
                    r.test_name,
                    r.ts_id,
                    r.tc_id,
                    r.ti_id,
                    r.commit_tag,
                    r.test_type,
                    r.type_source,
                ]
            )

    by_type: dict[str, int] = {}
    missing_ts = 0
    missing_tc = 0
    for r in rows:
        by_type[r.test_type] = by_type.get(r.test_type, 0) + 1
        if r.ts_id == "—":
            missing_ts += 1
        if r.tc_id == "—" and r.ti_id == "—":
            missing_tc += 1

    md = []
    md.append("# Comprehensive Test Audit Report (TS/TC + Type)")
    md.append("")
    md.append(
        "Generated from all `tests/test_*.py` and `tests/integration/test_*.py` functions named `test_*`."
    )
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append(f"- Total discovered tests: **{len(rows)}**")
    md.append(f"- Unit tests: **{by_type.get('unit', 0)}**")
    md.append(f"- Integration tests: **{by_type.get('integration', 0)}**")
    md.append(f"- System tests: **{by_type.get('system', 0)}**")
    md.append(f"- Acceptance tests: **{by_type.get('acceptance', 0)}**")
    md.append(f"- Missing TS ID: **{missing_ts}** tests")
    md.append(f"- Missing TC/TI ID: **{missing_tc}** tests")
    md.append("")
    md.append("## Deliverables")
    md.append("")
    md.append("- Full machine-readable table: `docs/test_audit_ts_tc_report.csv`")
    md.append("- This Markdown summary for report sections 8.2.3/8.2.4")
    md.append("")
    md.append("## Full Table (all tests)")
    md.append("")
    md.append("| Test Type | TS | TC | TI | Test Name | File | Line | Type Source |")
    md.append("|---|---|---|---|---|---|---:|---|")
    for r in rows:
        md.append(
            f"| {r.test_type} | {r.ts_id} | {r.tc_id} | {r.ti_id} | `{r.test_name}` | `{r.file}` | {r.line} | `{r.type_source}` |"
        )

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {OUT_CSV} and {OUT_MD} with {len(rows)} tests.")


if __name__ == "__main__":
    main()
