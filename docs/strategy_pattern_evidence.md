# Strategy Pattern Evidence

This document provides code-level evidence that the encoder layer uses the Strategy pattern.

## Why this is Strategy (GoF mapping)

- **Strategy interface**: common `encode(...)` contract.
- **Concrete strategies**: multiple encoder classes implement different algorithms.
- **Context**: handler/context selects a strategy at runtime and invokes a uniform method.

## 1) `BaseEncoder` defines the strategy contract

### Source files
- `src/psu_capstone/encoder_layer/base_encoder.py`

### Code example
```python
class BaseEncoder(ABC, Generic[T]):
    @abstractmethod
    def encode(self, input_value: T) -> list[int]:
        ...
```

### Why this is strategy evidence
All concrete encoder strategies share this common operation contract.

## 2) Concrete encoder strategies implement `encode(...)`

### Source files
- `src/psu_capstone/encoder_layer/scalar_encoder.py`
- `src/psu_capstone/encoder_layer/rdse.py`
- `src/psu_capstone/encoder_layer/category_encoder.py`
- `src/psu_capstone/encoder_layer/date_encoder.py`

### Code example
```python
class ScalarEncoder(BaseEncoder[int]):
    def encode(self, input_value: int | float) -> list[int]:
        self.register_encoding(input_value)
        return self._compute_encoding(input_value)

class RandomDistributedScalarEncoder(BaseEncoder[float]):
    def encode(self, input_value: float) -> list[int]:
        self.register_encoding(input_value)
        return self._compute_encoding(input_value)
```

### Why this is strategy evidence
Each class provides a distinct algorithm while honoring the same interface.

## 3) `EncoderHandler` selects strategy at runtime

### Source files
- `src/psu_capstone/encoder_layer/encoder_handler.py`

### Code example
```python
if isinstance(value, float) or isinstance(value, np.floating):
    encoder = RandomDistributedScalarEncoder(...)
    dense = encoder.encode(float(value))
elif scalartrue or isinstance(value, int) or isinstance(value, np.integer):
    encoder = ScalarEncoder(...)
    dense = encoder.encode(int(value))
elif isinstance(value, str):
    encoder = CategoryEncoder(...)
    dense = encoder.encode(value)
elif isinstance(value, datetime):
    encoder = DateEncoder(...)
    dense = encoder.encode(value)
```

### Why this is strategy evidence
A context object chooses among interchangeable strategies and calls them through one stable operation.

## 4) `DateEncoder` selects feature sub-strategies from configuration

### Source files
- `src/psu_capstone/encoder_layer/date_encoder.py`

### Code example
```python
def _setup_feature_encoder(self, options):
    if options.rdse_used:
        return RandomDistributedScalarEncoder(...)
    return ScalarEncoder(...)
```

### Why this is strategy evidence
Date feature encoding strategy is dynamically selected from configuration while usage remains uniform.

## Evidence summary
The encoder layer contains all Strategy roles: interface contract, multiple concrete algorithms, runtime selection, and uniform invocation.
