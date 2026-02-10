# Appendix T Addendum: TE/TC and Test Suite Table Updates

This addendum starts the missing Appendix **TE** and **TC** build-out for the FourierEncoder encoding/decoding suites and Date Encoder tests, following the same field structure used in the current Appendix T tables.

## Test Suite Tables (Additions)

### TS-12: Fourier Transform Encoder (Encoding + Decoding)
- **Suite ID:** TS-12
- **Suite Name:** Fourier Transform Encoder
- **Purpose:** Validate Fourier-based SDR encoding locality, determinism, orthogonality, and decode behavior for candidate frequency recovery.
- **Mapped Test Cases:** TC-080, TC-081, TC-082, TC-083, TC-084, TC-085, TC-086, TC-087
- **Pytest Module:** `tests/test_encoder_fourier.py`

### TS-05: SDR Date Encoder (Encoding + Decoding)
- **Suite ID:** TS-05
- **Suite Name:** SDR Date Encoder
- **Purpose:** Validate DateEncoder feature encodings and decoder outputs for season/day/weekend/holiday/time/custom/all-feature scenarios.
- **Mapped Test Cases:** TC-088, TC-089, TC-090, TC-091, TC-092
- **Pytest Modules:** `tests/test_encoder_date.py`, `tests/test_decoder_date.py`

---

## TC Tables (New)

### Table TC-080
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-12: Fourier Transform Encoder
- **Test Case ID:** TC-080 (Unit Test)
- **What to Test:** Verify identical pure-tone frequency inputs generate near-identical SDRs.
- **Test Data Input:** 75 Hz sinusoid encoded twice.
- **Expected Result:** Overlap ratio `>= 0.99`.

### Table TC-081
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-12: Fourier Transform Encoder
- **Test Case ID:** TC-081 (Unit Test)
- **What to Test:** Verify nearby frequencies overlap more than mid/far frequencies.
- **Test Data Input:** 60 Hz baseline compared to 61 Hz, 90 Hz, and 5 Hz tones.
- **Expected Result:** `close > mid > far`; close `>= 0.9`; far `<= 0.38`.

### Table TC-082
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-12: Fourier Transform Encoder
- **Test Case ID:** TC-082 (Unit Test)
- **What to Test:** Verify amplitude scaling does not materially alter encoding of same frequency.
- **Test Data Input:** 75 Hz tone at amplitudes 2.5 and 0.2.
- **Expected Result:** Overlap ratio `>= 0.99`.

### Table TC-083
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-12: Fourier Transform Encoder
- **Test Case ID:** TC-083 (Unit Test)
- **What to Test:** Verify far-separated tones remain mostly orthogonal.
- **Test Data Input:** 10 Hz and 180 Hz tones.
- **Expected Result:** Overlap ratio `<= 0.35`.

### Table TC-084
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-12: Fourier Transform Encoder
- **Test Case ID:** TC-084 (Unit Test)
- **What to Test:** Verify composite signal retains component frequency information.
- **Test Data Input:** Composite of 30 Hz and 90 Hz compared against components and unrelated 5 Hz signal.
- **Expected Result:** Composite overlap with each component `>= 0.55` and both exceed unrelated overlap.

### Table TC-085
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-12: Fourier Transform Encoder
- **Test Case ID:** TC-085 (Unit Test)
- **What to Test:** Verify amplitude modulation preserves carrier representation more than modulator.
- **Test Data Input:** 120 Hz carrier, 5 Hz modulator, depth 0.6.
- **Expected Result:** Modulated/carrier overlap `>= 0.55` and `carrier overlap > modulator overlap`.

### Table TC-086
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-12: Fourier Transform Encoder
- **Test Case ID:** TC-086 (Unit Test)
- **What to Test:** Verify decode identifies strongest frequency from candidate set.
- **Test Data Input:** Encoded 60 Hz signal; candidates `[20.0, 60.0, 120.0]`.
- **Expected Result:** Decoded frequency identifies 60 Hz candidate with valid confidence tuple.

### Table TC-087
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-12: Fourier Transform Encoder
- **Test Case ID:** TC-087 (Unit Test)
- **What to Test:** Verify decode rejects SDR vectors with invalid size.
- **Test Data Input:** SDR list with size mismatch to encoder expectations.
- **Expected Result:** Exception raised for incorrect SDR size.

### Table TC-088
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-05: SDR Date Encoder
- **Test Case ID:** TC-088 (Unit Test)
- **What to Test:** Verify DateEncoder season encoding and decoding range behavior.
- **Test Data Input:** Multi-date set across year and edge dates.
- **Expected Result:** Decode contains `season`; values remain within day-of-year bounds and deterministic for same input.

### Table TC-089
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-05: SDR Date Encoder
- **Test Case ID:** TC-089 (Unit Test)
- **What to Test:** Verify DateEncoder day-of-week decoding range and determinism.
- **Test Data Input:** Multi-date set including weekday/weekend values.
- **Expected Result:** Decode contains `dayofweek`; values in `[0, 7)` and stable for repeated input.

### Table TC-090
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-05: SDR Date Encoder
- **Test Case ID:** TC-090 (Unit Test)
- **What to Test:** Verify DateEncoder weekend decode validity.
- **Test Data Input:** Multi-date set with weekdays and weekends.
- **Expected Result:** Decode contains `weekend`; values in `{0,1}`.

### Table TC-091
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-05: SDR Date Encoder
- **Test Case ID:** TC-091 (Unit Test)
- **What to Test:** Verify encoded DateEncoder output is binary.
- **Test Data Input:** Encoded outputs from representative datetime values.
- **Expected Result:** All bits are 0 or 1.

### Table TC-092
- **Project Name:** HTM-RL Integration Pipeline
- **Test Suite:** TS-05: SDR Date Encoder
- **Test Case ID:** TC-092 (Unit Test)
- **What to Test:** Verify encoded DateEncoder output length equals configured encoder size.
- **Test Data Input:** Encoded outputs generated with known parameterized size.
- **Expected Result:** Output vector length exactly equals encoder `size`.

---

## TE Tables (Starter Drafts)

> Notes: Keeping the same Appendix T structure and starting execution records. Additional tester/date rows can be appended as test history grows.

### Table TE-080
- **Project Name:** HTM-RL Integration Pipeline
- **Test Case ID:** TC-080 – Verify identical frequency overlap determinism.
- **Testing Tools Used:** Pytest
- **Testing Type:** Function coverage
- **Execution Steps:** `pytest tests/test_encoder_fourier.py::test_identical_frequencies_overlap_completely`
- **Test Execution Records:**
  1. Tester: TBD | Date: TBD | Actual Result: Pending initial run | Status: Pending | Defect Correction: N/A
- **Execution Summary:** Starter TE entry created; ready for formal execution logging.

### Table TE-086
- **Project Name:** HTM-RL Integration Pipeline
- **Test Case ID:** TC-086 – Verify decode identifies strongest candidate frequency.
- **Testing Tools Used:** Pytest
- **Testing Type:** Function coverage
- **Execution Steps:** `pytest tests/test_encoder_fourier.py::test_decode_single_tone_returns_expected_frequency`
- **Test Execution Records:**
  1. Tester: TBD | Date: TBD | Actual Result: Pending initial run | Status: Pending | Defect Correction: N/A
- **Execution Summary:** Decode path TE starter entry created.

### Table TE-087
- **Project Name:** HTM-RL Integration Pipeline
- **Test Case ID:** TC-087 – Verify decode rejects incorrect SDR size.
- **Testing Tools Used:** Pytest
- **Testing Type:** Function coverage
- **Execution Steps:** `pytest tests/test_encoder_fourier.py::test_decode_rejects_incorrect_sdr_size`
- **Test Execution Records:**
  1. Tester: TBD | Date: TBD | Actual Result: Pending initial run | Status: Pending | Defect Correction: N/A
- **Execution Summary:** Error-path decode TE starter entry created.

### Table TE-088
- **Project Name:** HTM-RL Integration Pipeline
- **Test Case ID:** TC-088 – Verify DateEncoder season behavior.
- **Testing Tools Used:** Pytest
- **Testing Type:** Function coverage
- **Execution Steps:** `pytest tests/test_decoder_date.py::test_season`
- **Test Execution Records:**
  1. Tester: TBD | Date: TBD | Actual Result: Pending initial run | Status: Pending | Defect Correction: N/A
- **Execution Summary:** Date seasonal decode TE starter entry created.

### Table TE-091
- **Project Name:** HTM-RL Integration Pipeline
- **Test Case ID:** TC-091 – Verify DateEncoder output is binary.
- **Testing Tools Used:** Pytest
- **Testing Type:** Function coverage
- **Execution Steps:** `pytest tests/test_encoder_date.py::test_date_encode_output_only_zeros_and_ones`
- **Test Execution Records:**
  1. Tester: TBD | Date: TBD | Actual Result: Pending initial run | Status: Pending | Defect Correction: N/A
- **Execution Summary:** Date output validity TE starter entry created.

### Table TE-092
- **Project Name:** HTM-RL Integration Pipeline
- **Test Case ID:** TC-092 – Verify DateEncoder output length matches size.
- **Testing Tools Used:** Pytest
- **Testing Type:** Function coverage
- **Execution Steps:** `pytest tests/test_encoder_date.py::test_date_encode_output_length_equals_size`
- **Test Execution Records:**
  1. Tester: TBD | Date: TBD | Actual Result: Pending initial run | Status: Pending | Defect Correction: N/A
- **Execution Summary:** Date size-conformance TE starter entry created.
