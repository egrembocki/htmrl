# Marker Pattern Evidence

This document provides code-level evidence of marker-style contracts in the codebase.

## Why this is marker-style in Python

Python typically uses `Protocol` + `@runtime_checkable` instead of empty Java-style marker interfaces. These protocol declarations still mark capability and compatibility boundaries.

## 1) Agent-layer protocol marker

### Source files
- `src/psu_capstone/agent_layer/agent_interface.py`

### Code example
```python
@runtime_checkable
class AgentInterface(Protocol):
    def create_memory(...):
        ...

    def create_pooler(...):
        ...
```

### Why this is marker evidence
The protocol marks the capability set any agent implementation must provide.

## 2) Encoder-layer protocol marker

### Source files
- `src/psu_capstone/encoder_layer/encoder_interface.py`

### Code example
```python
@runtime_checkable
class EncoderInterface(Protocol):
    def encode(self, data: list[int]) -> list[int]:
        ...

    def decode(self, data: list[int]) -> list[int]:
        ...
```

### Why this is marker evidence
The protocol acts as a compatibility marker for encoder-like components across modules.

## 3) Input/SDR protocol markers

### Source files
- `src/psu_capstone/input_layer/input_interface.py`
- `src/psu_capstone/sdr_layer/sdr_interface.py`

### Code example
```python
@runtime_checkable
class InputInterface(Protocol):
    def input_data(...):
        ...

@runtime_checkable
class SDRInterface(Protocol):
    def sparsity(self) -> float:
        ...
```

### Why this is marker evidence
These interfaces mark required cross-layer behaviors without inheritance coupling to one base class.

## Evidence summary
The codebase uses runtime-checkable protocols as Pythonic marker contracts to define layer capabilities and swappable collaborator boundaries.
