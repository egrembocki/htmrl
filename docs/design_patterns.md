# Design Pattern Survey (Quick)

This repository contains clear examples (and a few "in-progress" uses) of the requested design patterns.

## 1) Singleton

- `EncoderHandler` implements singleton behavior via `__instance`, a custom `__new__`, and `get_instance()`.  
- `BatchEncoderHandler` and `InputHandler` follow the same approach.

## 2) Factory

- `Agent.create_pooler()` creates/configures `SpatialPooler` objects.
- `Agent.create_memory()` creates/configures `TemporalMemory` objects.
- `InputField.__init__()` creates the concrete encoder using `params.encoder_class(params)`, so encoder construction is selected by configuration and instantiated in one place (factory-style creation used by Brain-managed fields).
- The module notes some parts are practical factory methods and could be evolved into fuller factory patterns.

## 3) Strategy

- `DateEncoder` switches between `RandomDistributedScalarEncoder` and `ScalarEncoder` in `_setup_feature_encoder()` based on configuration (`rdse_used`).
- Both classes provide interchangeable encoding behavior behind a common `encode` shape.

## 4) Marker Interface (Python equivalent)

- `AgentInterface`, `EncoderInterface`, `InputInterface`, and `SDRInterface` are `Protocol` types with `@runtime_checkable`.
- In Python, these protocols act as lightweight type markers/contracts to indicate capability and compatibility.

## 5) Delegation

- `EncoderHandler.build_composite_sdr()` delegates actual encoding work to concrete encoder objects (`RandomDistributedScalarEncoder`, `ScalarEncoder`, `CategoryEncoder`, `DateEncoder`) selected per value type.
- `Brain.step()` delegates to `encode_only()` and `compute_only()`, which then delegate to field objects (`InputField.encode`, `ColumnField.compute`).
- `InputHandler.input_data()` delegates ingestion stages to helper methods (`_load_from_file`, `_raw_to_sequence`, `_apply_required_columns`, `_validate_data`) rather than implementing all parsing/normalization inline.

## 6) HTM + Brain specifics

- `HTM.make_state_class()` is a small **factory** for behavior mixins. It dynamically creates state-tracking classes (`Active`, `Winner`, `Predictive`, etc.) used throughout HTM entities.
- `HTM.Segment` uses a configurable `synapse_cls` and grows new synapses via that class, which acts like a lightweight **strategy/factory blend** for synapse behavior (`DistalSynapse` vs `ApicalSynapse`).
- `Brain` is a strong **delegation/facade** entry point: one `step()` call orchestrates input encoding and column computation across many field objects.
- `InputField` supports a **factory-like encoder plug-in point** via encoder parameter classes (`encoder_class`), which Brain composes through its managed fields.

---

## Quick Diagram (PlantUML)

`docs/puml/design_patterns.puml`

```plantuml
@startuml
!include docs/puml/design_patterns.puml
@enduml
```

## Delegation Reference

- See `docs/delegation_pattern_examples.md` for concrete code locations and call flows where delegation is used.
