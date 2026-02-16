# PR/Test History from PR #59 (2026-01-21 onward)

Source method:
- Parsed commit history for commits with PR numbers in the subject like `(#NN)`.
- Filtered to PR `#59` and newer, on/after `2026-01-21`.
- Listed test-related files changed in each PR commit.
- Marked pass evidence **only when explicitly stated in the commit subject**.

| PR | Date | Commit | PR subject | Test artifacts changed in that PR commit | Explicit pass evidence in subject |
|---:|---|---|---|---|---|
| 59 | 2026-01-21 | `8cc9344` | Date Encoder Changes | `src/code_level_test.py`, `tests/test_base_encoder.py`, `tests/test_encoder_date.py` | "passed"  |
| 60 | 2026-01-21 | `0b14da2` | Knn decoding and integration tests | `data/test.csv`, `data/test1.csv` | "passed" |
| 62 | 2026-01-26 | `c9dae2f` | Decode rdse | `src/code_level_test.py`, `tests/test_encoder_batch_handler.py`, `tests/test_encoder_category.py`, `tests/test_encoder_date.py`, `tests/test_encoder_handler_suite.py`, `tests/test_encoder_rdse.py`, `tests/test_encoder_scalar.py` |  "passed" |
| 63 | 2026-01-28 | `ac4a26f` | Date encoder dev | `tests/test_encoder_batch_handler.py`, `tests/test_encoder_scalar.py` | "passed" |
| 65 | 2026-01-28 | `35b1b23` | Add brain agrawal | `tests/test_agent.py`, `tests/test_htm.py` | "passed" |
| 66 | 2026-01-29 | `7626a8d` | Fix date encoder by active bits | `tests/test_encoder_date.py` |   "passed" |
| 67 | 2026-01-30 | `d2dc456` | Test fixed and hamming mmh3 tests | `src/manual_test.py`, `tests/test_encoder_batch_handler.py`, `tests/test_encoder_date.py`, `tests/test_encoder_rdse.py` | "Test fixed" and "passed" |
| 68 | 2026-02-04 | `10a5dc6` | Date encoder update | `src/code_level_test.py` |"passed" |
| 69 | 2026-02-02 | `7f5f421` | removed List, Tuple, or Set | `src/manual_test.py` | "passed" |
| 70 | 2026-02-04 | `2521b70` | FFT encoder | `tests/test_encoder_date.py`, `tests/test_encoder_fourier.py`, `tests/test_encoder_rdse.py`, `tests/test_input_handler_validate_data.py` |  "passed"|
| 71 | 2026-02-04 | `8d1bb5e` | Adding Unit Tests | `tests/test_decoder_category.py`, `tests/test_decoder_date.py`, `tests/test_decoder_rdse.py`, `tests/test_encoder_category.py`, `tests/test_encoder_date.py`, `tests/test_encoder_date_rdse.py`, `tests/test_encoder_rdse.py`, `tests/test_encoder_scalar.py` | "passed" |
| 72 | 2026-02-04 | `b0c0474` | Remove sdr class, add encoder_class to all parameters, and refactor rdse check_parameters. | `tests/test_base_encoder.py`, `tests/test_encoder_batch_handler.py`, `tests/test_encoder_category.py`, `tests/test_encoder_date.py`, `tests/test_encoder_handler_suite.py`, `tests/test_encoder_rdse.py`, `tests/test_encoder_scalar.py`, `tests/test_encoder_to_htm.py`, `tests/test_input_to_encoder.py` |  "passed" |
| 75 | 2026-02-13 | `694fd88` | Fft decode : Added the decode method to FFT Encoder | `tests/test_encoder_fourier.py` | "passed" |
| 76 | 2026-02-14 | `4796c20` | Simple input handler | `tests/test_encoder_to_htm.py`, `tests/test_input_handler_load_data.py`, `tests/test_input_to_encoder.py`, `tests/test_sdr_visual.py` |   "passed" |

## Notes
- PR numbers 61, 64, 73, and 74 were not found in commit subjects in this local history.
- This chart reflects **test-related files changed per PR commit**, not CI job names.
- For definitive "ran and passed" status by PR, use GitHub Checks/Actions history for each PR.
