# Marker Pattern Examples

This document catalogs concrete places where **marker-style typing/contracts** appear in the project.

## What counts as marker here?

In Python, marker behavior is often approximated using lightweight interfaces/protocols that signal object compatibility and capabilities. These are not Java-style empty marker interfaces, but they do provide tagging/contract semantics.

## 1) Runtime-checkable protocol interfaces as compatibility markers

### Where
- `src/psu_capstone/agent_layer/agent_interface.py`
- `src/psu_capstone/encoder_layer/encoder_interface.py`
- `src/psu_capstone/input_layer/input_interface.py`
- `src/psu_capstone/sdr_layer/sdr_interface.py`

### Marker flow
- Each interface is declared as `@runtime_checkable` `Protocol`.
- Objects can be validated for protocol conformance at runtime and statically typed as compatible collaborators.

### Why this is marker-style
These protocol types mark expected capabilities for each layer (agent, encoder, input, SDR), enabling compatibility checks without inheritance coupling.

## 2) Capability tagging across module boundaries

### Where
- Interface modules above and their concrete implementers in `src/psu_capstone/...`.

### Marker flow
- Consumer code can depend on protocol contracts instead of concrete classes.
- Concrete implementations can be swapped as long as they satisfy the protocol shape.

### Why this is marker-style
The protocols act as cross-layer type markers and contracts for pluggability.

## Summary
The codebase does not use empty marker interfaces in the strict GoF/Java sense, but it does use runtime-checkable protocols as Pythonic marker-style capability tags.
