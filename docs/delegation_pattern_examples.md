# Delegation Pattern Examples

This document catalogs concrete places where the **delegation pattern** is used in the project.

## What counts as delegation here?

A class or method receives a request, then forwards the real work to a more specialized collaborator instead of implementing the behavior inline.

## 1) `Brain` delegates orchestration to field objects

### Where
- `src/psu_capstone/agent_layer/brain.py`

### Delegation flow
- `Brain.step(...)` delegates encoding to `encode_only(...)`, computation to `compute_only(...)`, and output decoding to each `OutputField.decode(...)`.
- `Brain.encode_only(...)` delegates each input value to `InputField.encode(...)`.
- `Brain.compute_only(...)` delegates learning/inference to each `ColumnField.compute(...)`.

### Why this is delegation
`Brain` is an orchestration façade. It does not perform encoding/column computation itself; it forwards these responsibilities to the field-level collaborators.

## 2) `InputField` delegates encoding/decoding to its configured encoder

### Where
- `src/psu_capstone/agent_layer/HTM.py` (`InputField`)

### Delegation flow
- `InputField.__init__(...)` constructs an encoder from parameters (`params.encoder_class(params)`).
- `InputField.encode(...)` delegates transformation to `self.encoder.encode(...)`.
- `InputField.decode(...)` delegates reverse mapping to `self.encoder.decode(...)`.

### Why this is delegation
`InputField` manages cell state and coordination, while the concrete encoder object owns bit-level encoding logic.

## 3) `EncoderHandler` delegates per-value encoding to concrete encoder instances

### Where
- `src/psu_capstone/encoder_layer/encoder_handler.py` (`build_composite_sdr`)

### Delegation flow
For each column value, `build_composite_sdr(...)` selects and delegates to a specialized encoder:
- `RandomDistributedScalarEncoder.encode(...)` for floating-point values.
- `ScalarEncoder.encode(...)` for integer values.
- `CategoryEncoder.encode(...)` for string/category values.
- `DateEncoder.encode(...)` for datetime values.

### Why this is delegation
`EncoderHandler` coordinates type-based routing and SDR assembly, while each concrete encoder handles the encoding algorithm.

## 4) `DateEncoder` delegates feature-specific work to sub-encoders

### Where
- `src/psu_capstone/encoder_layer/date_encoder.py`

### Delegation flow
- `_setup_feature_encoder(...)` creates a feature encoder (`ScalarEncoder` or `RandomDistributedScalarEncoder`) for each enabled temporal feature.
- During encode-time, `DateEncoder` delegates each feature slice (season/day-of-week/weekend/custom/holiday/time-of-day) to the corresponding sub-encoder and combines results.

### Why this is delegation
`DateEncoder` acts as a composite coordinator: per-feature conversion is performed by dedicated sub-encoders.


## 5) `InputHandler` delegates normalization work to specialized helper methods

### Where
- `src/psu_capstone/input_layer/input_handler.py`

### Delegation flow
- `InputHandler.input_data(...)` routes file paths to `_load_from_file(...)`, then delegates shaping to `_raw_to_sequence(...)`, schema alignment to `_apply_required_columns(...)`, and final checks to `_validate_data(...)`.
- `InputHandler.to_encoder_sequence(...)` delegates record construction to `input_data(...)`, then delegates payload validation to `_validate_sequence(...)`.
- `_raw_to_sequence(...)` further delegates to `_coerce_records_for_sequence(...)`, `_fill_missing_values(...)`, `_normalize_record_entries(...)`, and `_detect_repeating_values(...)` to keep each transformation step focused.

### Why this is delegation
`InputHandler` acts as a coordinator for ingestion. Instead of one monolithic method, it forwards each phase (loading, coercion, normalization, validation) to narrower helpers that encapsulate the real processing logic.

## Summary
The delegation pattern is primarily used to keep orchestration components (`Brain`, handlers, composite encoders) separate from algorithmic components (concrete encoders and field processors). This improves modularity and allows implementations to be swapped without changing top-level flow.
