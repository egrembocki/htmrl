# Delegation Pattern Evidence

This document provides code-level evidence of delegation behavior across orchestration and handler components.

## Why this is Delegation (GoF mapping)

- A coordinator receives a request.
- It forwards the real work to specialized collaborators.
- It aggregates results instead of implementing all algorithms itself.

## 1) `Brain` delegates encoding and compute phases

### Source files
- `src/psu_capstone/agent_layer/brain.py`

### Code example
```python
def step(self, x: list[Any], learn: bool = False) -> list[int]:
    in_encode = self.encode_only(x)
    self.compute_only(in_encode, learn=learn)
    out = [field.decode() for field in self.output_fields]
    return [item for sublist in out for item in sublist]
```

### Why this is delegation evidence
`Brain` orchestrates and forwards responsibilities to helper methods and field objects.

## 2) `InputField` delegates to configured encoder object

### Source files
- `src/psu_capstone/agent_layer/HTM.py`

### Code example
```python
class InputField(NonAllocInput):
    ...
    def encode(self, value: Any) -> None:
        enc = self.encoder.encode(value)
        self.set_state(enc)

    def decode(self) -> list[Any]:
        return self.encoder.decode(self.state)
```

### Why this is delegation evidence
`InputField` owns coordination/state updates, while the encoder object performs encoding/decoding algorithms.

## 3) `EncoderHandler` delegates to concrete encoders

### Source files
- `src/psu_capstone/encoder_layer/encoder_handler.py`

### Code example
```python
if isinstance(value, float) or isinstance(value, np.floating):
    encoder = RandomDistributedScalarEncoder(...)
    dense = encoder.encode(float(value))
elif isinstance(value, str):
    encoder = CategoryEncoder(...)
    dense = encoder.encode(value)
```

### Why this is delegation evidence
The handler routes each value to a specialized encoder instead of encoding inline.

## 4) `InputHandler` delegates ingestion stages to helpers

### Source files
- `src/psu_capstone/input_layer/input_handler.py`

### Code example
```python
def input_data(self, raw_data, required_columns=None):
    if isinstance(raw_data, (str, Path)):
        records = self._load_from_file(raw_data)
    else:
        records = raw_data
    sequence = self._raw_to_sequence(records)
    sequence = self._apply_required_columns(sequence, required_columns)
    self._validate_data(sequence)
    return sequence
```

### Why this is delegation evidence
A high-level method forwards each stage to focused helpers for loading, shaping, alignment, and validation.

## Evidence summary
Delegation is consistently used so orchestrators coordinate flows while specialized collaborators implement the concrete behavior.
