# Singleton Pattern Examples

This document catalogs concrete places where the **singleton pattern** is used in the project.

## What counts as singleton here?

A class maintains exactly one shared instance, usually via a private class-level instance field plus controlled construction/access (`__new__`, classmethod accessors).

## 1) `EncoderHandler` singleton

### Where
- `src/psu_capstone/encoder_layer/encoder_handler.py`

### Singleton flow
- Class variable `__instance` stores the shared instance.
- `__new__(...)` creates the object only once and reuses it on subsequent calls.
- `get_instance()` returns the same shared instance.

### Why this is singleton
Construction is constrained to a single class-level object reused across calls.

## 2) `BatchEncoderHandler` singleton

### Where
- `src/psu_capstone/encoder_layer/batch_encoder_handler.py`

### Singleton flow
- Class variable `__instance` tracks the sole instance.
- `__new__(...)` returns the existing instance when available.
- `get_instance()` exposes stable singleton access.

### Why this is singleton
The class enforces a single, reusable handler object rather than letting callers create multiple independent instances.

## 3) `InputHandler` singleton

### Where
- `src/psu_capstone/input_layer/input_handler.py`

### Singleton flow
- Class variable `__instance` retains shared state.
- `__new__(...)` creates the object only on first construction.
- `get_instance()` provides class-level retrieval of that same instance.

### Why this is singleton
Input ingestion state is coordinated through one shared handler object.

## Summary
Singleton usage is concentrated in stateful coordinator/handler classes where one shared lifecycle is desired across the application.
