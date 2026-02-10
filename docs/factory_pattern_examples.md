# Factory Pattern Examples

This document catalogs concrete places where **factory-style creation** is used in the project.

## What counts as factory here?

A class or method centralizes object construction and returns/configures concrete objects so callers do not need to instantiate those objects directly.

## 1) `Agent` factory methods for HTM components

### Where
- `src/psu_capstone/agent_layer/agent.py`

### Factory flow
- `Agent.create_pooler(...)` builds a `SpatialPooler(...)` from caller-supplied configuration and appends it to `self._poolers`.
- `Agent.create_memory(...)` builds a `TemporalMemory(...)` from configuration and appends it to `self._memory`.

### Why this is factory
`Agent` provides creator methods that encapsulate concrete class construction and initialization details.

## 2) `InputField` creates pluggable encoders from parameter metadata

### Where
- `src/psu_capstone/agent_layer/HTM.py` (`InputField`)

### Factory flow
- `InputField.__init__(...)` receives encoder parameter objects.
- It instantiates the concrete encoder through the parameter-owned class reference: `params.encoder_class(params)`.

### Why this is factory
Construction is parameter-driven and deferred to runtime metadata, allowing callers to switch concrete encoder types without changing `InputField` logic.

## 3) `HTM.make_state_class(...)` dynamically creates state classes

### Where
- `src/psu_capstone/agent_layer/HTM.py`

### Factory flow
- `make_state_class(label)` generates and returns a new class type with shared state behavior.
- The module then creates concrete state classes (`Active`, `Winner`, `Predictive`, etc.) by calling the factory repeatedly.

### Why this is factory
A single creator function produces multiple concrete class types with common behavior based on input configuration (`label`).

## 4) Encoder handlers create concrete encoders at runtime

### Where
- `src/psu_capstone/encoder_layer/encoder_handler.py`
- `src/psu_capstone/encoder_layer/batch_encoder_handler.py`

### Factory flow
- Handler code inspects value/column metadata and creates specific encoder objects (`RandomDistributedScalarEncoder`, `ScalarEncoder`, `CategoryEncoder`, `DateEncoder`).
- Each created encoder is then used to encode the record values.

### Why this is factory
Handlers centralize runtime selection + construction of concrete encoders, so callers can pass data without knowing concrete encoder classes.

## Summary
Factory-style creation appears in both explicit creator methods (`Agent`), dynamic class factories (`make_state_class`), and runtime type-driven object construction (encoder handlers).
