# Consolidated Test Reports (Latest)

_Last updated: 2026-04-17 (UTC)_

This file consolidates the previously separate test-report documents into a single source of truth:

- `docs/test_audit_ts_tc_report.md`
- `docs/final_report_4_0_test_inventory.md`
- `docs/test_stage_reconciliation.md`

## 1) Global TS/TC/TI Audit Snapshot

Generated from discovered `test_*` functions under `tests/`.

- Total discovered tests: **358**
- Unit tests: **314**
- Integration tests: **34**
- System tests: **9**
- Acceptance tests: **1**
- Missing TS ID: **151** tests
- Missing TC/TI ID: **201** tests

### Canonical machine-readable dataset

- Full table (all tests): `docs/test_audit_ts_tc_report.csv`

## 2) Integration, System, and Acceptance Inventory

### Integration

- TS-11 suite uses **TI-001..TI-014** identifiers in `tests/integration/test_encoder_htm_integration.py`.
- Additional integration tests exist without explicit TS/TC/TI tags and should be assigned IDs for complete traceability.

### System

System/integration-tagged tests currently include:

- `tests/test_cartpole_brain_training.py` (TS-21 / TC-167)
- `tests/test_env_adapter.py` (TS-20 / TC-170, TC-171, TC-172)
- `tests/test_fin_gym.py` (currently unnumbered)
- `tests/integration/test_input_to_encoder.py::test_sine_wave_through_input_handler` (currently unnumbered)

### Acceptance

- `tests/test_demo_driver_full_demo.py::test_run_full_demo_generates_visual_artifacts` is acceptance-tagged and mapped to TS-10 / TC-097.

## 3) Stage Reconciliation Against Provided TS/TC Table

The supplied table is mostly aligned. The following **Test Stage** updates are needed to match the current codebase annotations and test-ID usage:

| Suite | Test Case ID | Provided Stage | Updated Stage | Evidence |
|---|---|---|---|---|
| TS-10 or TS-13 (ID collision) | TC-097 | Unit | Acceptance (for the existing annotated TC-097 test) | `tests/test_demo_driver_full_demo.py` is tagged `# commit: system/acceptance test` and mapped as TS-10 / TC-097 in the audit export. |
| TS-19 and TS-21 (ID collision) | TC-167 | Unit | Mixed: Unit **and** System | TC-167 appears as unit in `tests/test_agent.py` and as system/integration in `tests/test_cartpole_brain_training.py`. |
| TS-20 (EnvAdapter / Wrapper) | TC-170 | Unit | System (integration-style) | `tests/test_env_adapter.py` uses `# commit: system/integration test` for TC-170. |
| TS-20 (EnvAdapter / Wrapper) | TC-171 | Unit | System (integration-style) | `tests/test_env_adapter.py` uses `# commit: system/integration test` for TC-171. |
| TS-20 (EnvAdapter / Wrapper) | TC-172 | Unit | System (integration-style) | `tests/test_env_adapter.py` uses `# commit: system/integration test` for TC-172. |
| TS-24 (FinGym) | TC-205 | Unit | System (integration-style intent) | FinGym tests in `tests/test_fin_gym.py` are tagged `# commit: system/integration test`; explicit TC-205 annotation is currently missing in code. |

## 4) Known ID Consistency Issues (Latest)

- `TC-097` is reused across different intents (date-decode table vs acceptance demo mapping in the code audit).
- `TC-167` is reused with conflicting classification in current tests:
  - Unit usage in TS-19 (`tests/test_agent.py`)
  - System usage in TS-21 (`tests/test_cartpole_brain_training.py`)

Recommendation: assign unique TC IDs per test intent/type to maintain one-to-one traceability.

## 5) Current Actionable Recommendations

1. Keep TS-10 / TC-097 as acceptance.
2. Keep TS-20 and TS-21 as system/integration-style mappings.
3. Decide whether TI-001..TI-014 should be mirrored with TC IDs for table consistency.
4. Assign TS/TC IDs to currently unnumbered FinGym/pipeline tests.
5. Resolve the `TC-167` duplicate-ID collision.
