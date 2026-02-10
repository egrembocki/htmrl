# Design Pattern Audit (InputField/HTM/Brain Focus)

This note summarizes where core design patterns appear in the current codebase,
with emphasis on `InputField`, `HTM.py`, and `Brain`.

## Factory

### Strong example: `InputField` in `HTM.py`

`InputField.__init__` takes parameter objects (`encoder_params`) that carry an
`encoder_class` reference and then constructs the encoder via:

```python
self.encoder = params.encoder_class(params)
```

This is a parameter-driven factory style (sometimes called a simple factory via
configuration): creation is deferred to runtime metadata instead of hard-coding
a concrete encoder type in each caller.

### Additional factory-like sites

- `Agent.create_pooler` and `Agent.create_memory` are explicit creator methods.
- `EncoderHandler.build_composite_sdr` and `BatchEncoderHandler._build_dict_list_sdr`
  choose concrete encoder implementations based on input/column type.

## Singleton

- `InputHandler`, `EncoderHandler`, and `BatchEncoderHandler` each implement a
  singleton via `__instance` + `__new__`/`get_instance`.

## Delegation

### `Brain` as orchestrator/delegator

`Brain.step()` delegates work to:

- `encode_only()` (delegates to `InputField.encode`), and
- `compute_only()` (delegates to `ColumnField.compute`),

then delegates output conversion via `field.decode()` on output fields.

## Marker

A strict GoF/Java-style marker interface is not clearly present.

- The project uses `Protocol`-based interfaces (`AgentInterface`,
  `EncoderInterface`, etc.), but these are behavioral contracts with required
  methods, not empty tagging types.

## Prototype

- `SDR.get_sdr()` and `set_sdr()` provide explicit copy/clone behavior.
- `copy.deepcopy(...)` is used heavily for parameter/state duplication in
  encoder and handler constructors.

## Strategy

- `BaseEncoder` defines the shared `encode(...)` contract.
- Concrete strategies are implemented by encoders such as `ScalarEncoder`,
  `RandomDistributedScalarEncoder`, `CategoryEncoder`, and `DateEncoder`.
- Handlers select strategies at runtime based on data types or custom mapping.


## Immutable pattern (assessment)

A strict immutable-object pattern is **not** a core architectural pattern in this
codebase.

- Parameter objects are mostly mutable dataclasses (no `frozen=True` usage in the
  encoder parameter definitions).
- The code frequently uses `copy.deepcopy(...)` before storing config/state,
  which is a defensive-copy approach that reduces side effects but does not make
  objects immutable.

Practical takeaway: the project favors *mutable objects + defensive copies* over
full immutability.

## Decorator pattern (assessment)

There is no strong GoF structural **Decorator** pattern (wrapping objects that
share the same interface while adding behavior dynamically) in the HTM/Brain
path.

However, Python language decorators are used heavily:

- `@runtime_checkable` for Protocol interfaces,
- `@property` for computed/read-only accessors,
- `@dataclass` for parameter containers.

Those are Python metaprogramming decorators, not the GoF object-structure
Decorator pattern.

## Applicability Summary

- **Factory**: use when runtime metadata should control which concrete object is created.
- **Singleton**: use when one shared coordinator/state holder is required.
- **Delegation**: use when a façade/orchestrator coordinates specialized collaborators.
- **Marker**: useful for metadata-only tagging, but not clearly used here.
- **Prototype**: use when copying configured objects is cheaper/clearer than rebuilding.
- **Strategy**: use when interchangeable algorithms share one contract.
