# Strategy Pattern Examples

This document catalogs concrete places where the **strategy pattern** is used in the project.

## What counts as strategy here?

Multiple interchangeable algorithm implementations expose a common operation, and a context selects which implementation to use at runtime.

## 1) Encoder families implement interchangeable `encode(...)` algorithms

### Where
- `src/psu_capstone/encoder_layer/base_encoder.py`
- `src/psu_capstone/encoder_layer/scalar_encoder.py`
- `src/psu_capstone/encoder_layer/rdse.py`
- `src/psu_capstone/encoder_layer/category_encoder.py`
- `src/psu_capstone/encoder_layer/date_encoder.py`

### Strategy flow
- `BaseEncoder` defines the encoding contract.
- Concrete encoders (`ScalarEncoder`, `RandomDistributedScalarEncoder`, `CategoryEncoder`, `DateEncoder`) implement different `encode(...)` algorithms.

### Why this is strategy
Each concrete encoder is an interchangeable algorithm object behind the same interface shape.

## 2) `EncoderHandler` selects concrete encoder strategy by runtime value type

### Where
- `src/psu_capstone/encoder_layer/encoder_handler.py`

### Strategy flow
- For each input value, handler logic chooses a concrete encoder type based on runtime type checks.
- The selected encoder is invoked uniformly through `encode(...)`.

### Why this is strategy
The handler acts as context and selects an algorithm implementation dynamically at runtime.

## 3) `DateEncoder` switches feature sub-encoder strategy using configuration

### Where
- `src/psu_capstone/encoder_layer/date_encoder.py`

### Strategy flow
- `_setup_feature_encoder(...)` chooses either `RandomDistributedScalarEncoder` or `ScalarEncoder` based on `rdse_used`.
- Each enabled date feature (season/day-of-week/weekend/custom/holiday/time-of-day) delegates encoding to the selected sub-encoder strategy.

### Why this is strategy
Per-feature algorithm choice is configurable and interchangeable while keeping a stable call surface.

## Summary
Strategy appears as interchangeable encoding algorithms plus runtime selection contexts (`EncoderHandler` and `DateEncoder`).
