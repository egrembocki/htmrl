# Factory Pattern Evidence

This document provides code-level evidence that factory-style creation is used in the project.

## Why this is Factory (GoF mapping)

- **Factory/creator** centralizes construction.
- **Concrete products** are created based on runtime configuration or creator methods.
- **Callers** use created objects without building them inline.

## 1) `Agent` creator methods construct HTM components

### Source files
- `src/psu_capstone/agent_layer/agent.py`

### Code example
```python
def create_pooler(self, input_dimensions: list[int], column_dimensions: list[int],
                  potential_radius: int, potential_pct: float,
                  global_inhibition: bool, local_area_density: float,
                  num_active_columns_per_inh_area: float, stimulus_threshold: int,
                  syn_perm_inactive_dec: float, syn_perm_active_inc: float,
                  syn_perm_connected: float, min_pct_overlap_duty_cycle: float,
                  duty_cycle_period: int, boost_strength: float,
                  seed: int, wrap_around: bool) -> SpatialPooler:
    pooler = SpatialPooler(
        inputDimensions=input_dimensions,
        columnDimensions=column_dimensions,
        potentialRadius=potential_radius,
        potentialPct=potential_pct,
        globalInhibition=global_inhibition,
        localAreaDensity=local_area_density,
        numActiveColumnsPerInhArea=num_active_columns_per_inh_area,
        stimulusThreshold=stimulus_threshold,
        synPermInactiveDec=syn_perm_inactive_dec,
        synPermActiveInc=syn_perm_active_inc,
        synPermConnected=syn_perm_connected,
        minPctOverlapDutyCycle=min_pct_overlap_duty_cycle,
        dutyCyclePeriod=duty_cycle_period,
        boostStrength=boost_strength,
        seed=seed,
        wrapAround=wrap_around,
    )
    self._poolers.append(pooler)
    return pooler
```

### Why this is factory evidence
`Agent` exposes a creator method that encapsulates concrete `SpatialPooler` construction and returns the built object.

## 2) `InputField` builds encoder products from parameter metadata

### Source files
- `src/psu_capstone/agent_layer/HTM.py`

### Code example
```python
class InputField(NonAllocInput):
    ...
    def __init__(self, n: int, sparsity: float, count: int = 1,
                 encoder_params: EncoderParameter | list[EncoderParameter] | None = None):
        ...
        if encoder_params is not None:
            params = encoder_params if isinstance(encoder_params, EncoderParameter) else encoder_params[0]
            self.encoder = params.encoder_class(params)
```

### Why this is factory evidence
Construction is delegated to runtime-supplied metadata (`encoder_class`), so callers can choose products without changing `InputField` logic.

## 3) `HTM.make_state_class(...)` is a class factory

### Source files
- `src/psu_capstone/agent_layer/HTM.py`

### Code example
```python
def make_state_class(label: str):
    class StateClass:
        __slots__ = ("_state",)
        ...
    StateClass.__name__ = label
    return StateClass

Active = make_state_class("Active")
Winner = make_state_class("Winner")
Predictive = make_state_class("Predictive")
```

### Why this is factory evidence
A single creator function dynamically returns different concrete class types.

## 4) Encoder handlers build concrete encoders at runtime

### Source files
- `src/psu_capstone/encoder_layer/encoder_handler.py`

### Code example
```python
if isinstance(value, float) or isinstance(value, np.floating):
    encoder = RandomDistributedScalarEncoder(minimum=minimum, maximum=maximum)
    dense = encoder.encode(float(value))
elif scalartrue or isinstance(value, int) or isinstance(value, np.integer):
    encoder = ScalarEncoder(minimum=minimum, maximum=maximum)
    dense = encoder.encode(int(value))
elif isinstance(value, str):
    encoder = CategoryEncoder(category_list=column_values)
    dense = encoder.encode(value)
```

### Why this is factory evidence
The handler performs type-driven product selection and object construction before use.

## Evidence summary
Factory behavior is demonstrated through explicit creator methods, parameter-driven construction, and runtime selection + instantiation of concrete products.
