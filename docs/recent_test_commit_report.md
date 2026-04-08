# Recent Added Test Commits Report

Generated from git history and `# commit:` annotations in test files.

## 1) Recent commits that added/updated tests

- `fd89fa9` (2026-04-08) — Annotate test cases with commit type comments
- `3918ce7` (2026-04-07) — testing
- `a9de0d9` (2026-04-07) — test
- `acb9569` (2026-04-07) — Merge remote-tracking branch 'upstream/our_htm' into master-fork
- `50b3384` (2026-04-07) — Fix import and add stub classes for tests
- `3d465ed` (2026-04-07) — Resolve merge conflicts and finish merge
- `cab6199` (2026-04-07) — Added a lot of tests for the column field spatial pooler. Made a rough spatial pooler based on the research paper and adjusted some tests to it.
- `cb0c534` (2026-04-02) — Latest.
- `9013d15` (2026-04-02) — Added more graph tests where I excluded the encoder so we can see how the SP acts alone.
- `4205781` (2026-03-30) — More graphs to check activation frequency on SP.
- `4963df8` (2026-03-26) — Added test no single column dominates.
- `037c09e` (2026-03-25) — SP converge on sparsity test/print, not working fully yet.
- `9dfcdf0` (2026-03-24) — Added a robustness test that plots it over multiple epochs with many noise tests.
- `69c5244` (2026-03-24) — working on noise generation for the spatial pooler inside of the column field.
- `903bad1` (2026-03-24) — Added printing of overlaps for rdse, scalar, and sp. Added a rough skeleton of what the spatial pooler could be from the bami paper.
- `ae814a3` (2026-03-23) — flake
- `562a74b` (2026-03-23) — linting
- `44ecf17` (2026-03-22) — Correction on index out of bounds.
- `0d917cb` (2026-03-22) — modified an assert for test overlap gradient.
- `9bea86f` (2026-03-20) — added TS and TC

## 2) Test annotations currently present

- **unit test**: 282 test case annotations across 29 files
- **system/integration test**: 8 test case annotations across 3 files
- **integration test**: 3 test case annotations across 2 files
- **system/acceptance test**: 1 test case annotations across 1 files

## 3) What to add where

### Unit tests
- Primary bucket. Most test cases are already marked `unit test`.
- Continue adding single-module logic checks (encoders, decoders, HTM primitives, handlers).

### Integration tests
- Add/extend data-flow checks between adjacent layers (Input → Encoder, Encoder → HTM).
- Existing integration files:
  - `tests/integration/test_encoder_to_htm.py`
  - `tests/integration/test_input_to_encoder.py`

### System tests
- Add end-to-end scenario coverage (environment + adapter + brain + agent runtime).
- Existing system-tagged files:
  - `tests/test_cartpole_brain_training.py` (system/integration test)
  - `tests/test_env_adapter.py` (system/integration test)
  - `tests/test_fin_gym.py` (system/integration test)
  - `tests/test_demo_driver_full_demo.py` (system/acceptance test)

### Acceptance tests
- Add user-facing outcome checks (full demo behavior, success criteria, regressions on expected business behavior).
- Current acceptance coverage appears in `tests/test_demo_driver_full_demo.py` (`system/acceptance test`).

## 4) Gap summary

- Unit tests are heavily represented.
- Integration coverage exists but is relatively small and concentrated in `tests/integration/`.
- Acceptance coverage is minimal (single annotated file).
- Recommendation: prioritize new **acceptance** and **cross-layer integration** tests next.
