# Strategy Pattern Evidence: Encoder Layer

This document provides concrete evidence that the encoder layer is using the
**Strategy pattern**.

## Why this is Strategy (GoF mapping)

- **Strategy interface**: `BaseEncoder` defines a common `encode(...)` algorithm contract.
- **Concrete strategies**: `ScalarEncoder`, `RandomDistributedScalarEncoder`, `CategoryEncoder`, and `DateEncoder` each implement `encode(...)` differently.
- **Context**: `EncoderHandler` decides which concrete encoder to instantiate and delegates encoding to the selected object at runtime.

## 1) Strategy contract (`BaseEncoder`)

`BaseEncoder` is an abstract base class with an abstract `encode` method:

```python
class BaseEncoder(ABC, Generic[T]):
    ...
    @abstractmethod
    def encode(self, input_value: T) -> list[int]:
        ...
```

This is the shared strategy interface used by all encoder implementations.

## 2) Concrete strategy examples

### `ScalarEncoder` strategy

`ScalarEncoder` inherits from `BaseEncoder[int]` and provides its own `encode` implementation:

```python
class ScalarEncoder(BaseEncoder[int]):
    ...
    @override
    def encode(self, input_value: int | float) -> list[int]:
        self.register_encoding(input_value)
        return self._compute_encoding(input_value)
```

### `RandomDistributedScalarEncoder` strategy

`RandomDistributedScalarEncoder` inherits from `BaseEncoder[float]` and implements `encode` with an RDSE-specific algorithm:

```python
class RandomDistributedScalarEncoder(BaseEncoder[float]):
    ...
    @override
    def encode(self, input_value: float) -> list[int]:
        self.register_encoding(input_value)
        return self._compute_encoding(input_value)
```

### `CategoryEncoder` strategy

`CategoryEncoder` inherits from `BaseEncoder[str]` and uses category index mapping:

```python
class CategoryEncoder(BaseEncoder[str]):
    ...
    @override
    def encode(self, input_value: str) -> list[int]:
        if input_value not in self._category_list:
            index = 0
        else:
            index = self._category_list.index(input_value) + 1
        a = self.encoder.encode(int(index))
        return a
```

### `DateEncoder` strategy

`DateEncoder` inherits from `BaseEncoder[datetime | time.struct_time | None]` and composes several temporal sub-encodings:

```python
class DateEncoder(BaseEncoder[datetime | time.struct_time | None]):
    ...
    @override
    def encode(self, input_value: datetime | time.struct_time | None) -> list[int]:
        ...
        output_sdr: list[int] = []
        ...
        return output_sdr
```

## 3) Runtime strategy selection in context (`EncoderHandler`)

`EncoderHandler.build_composite_sdr(...)` selects different concrete strategies based on runtime value type, then calls the same method (`encode`) on each selected strategy:

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
else:
    raise TypeError(...)
```

This is exactly the Strategy pattern behavior:
- interchangeable algorithms with a shared interface,
- selected at runtime,
- used via uniform invocation (`encode(...)`).

## 4) Evidence summary

The encoder layer is not just "similar" to Strategy; it has the required structural parts:

1. Abstract strategy contract: `BaseEncoder.encode(...)`.
2. Multiple concrete strategy classes implementing that contract.
3. A context (`EncoderHandler`) that chooses and delegates to a strategy at runtime.

That combination is strong, direct evidence of the Strategy pattern in production code.
