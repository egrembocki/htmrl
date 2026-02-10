# Singleton Pattern Evidence

This document provides code-level evidence that singleton behavior is implemented in multiple handlers.

## Why this is Singleton (GoF mapping)

- **Singleton instance storage**: class-level `__instance` fields.
- **Controlled construction**: `__new__(...)` returns existing object once initialized.
- **Global access point**: `get_instance()` exposes shared instance.

## 1) `EncoderHandler` singleton enforcement

### Source files
- `src/psu_capstone/encoder_layer/encoder_handler.py`

### Code example
```python
class EncoderHandler(EncoderInterface):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @staticmethod
    def get_instance() -> "EncoderHandler":
        if EncoderHandler.__instance is None:
            EncoderHandler()
        return EncoderHandler.__instance
```

### Why this is singleton evidence
Only one `EncoderHandler` object is created and reused on all access paths.

## 2) `BatchEncoderHandler` singleton enforcement

### Source files
- `src/psu_capstone/encoder_layer/batch_encoder_handler.py`

### Code example
```python
class BatchEncoderHandler(EncoderInterface):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @staticmethod
    def get_instance() -> "BatchEncoderHandler":
        if BatchEncoderHandler.__instance is None:
            BatchEncoderHandler()
        return BatchEncoderHandler.__instance
```

### Why this is singleton evidence
Creation and retrieval are routed through class-level controls to maintain one shared instance.

## 3) `InputHandler` singleton enforcement

### Source files
- `src/psu_capstone/input_layer/input_handler.py`

### Code example
```python
class InputHandler(InputInterface):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @staticmethod
    def get_instance() -> "InputHandler":
        if InputHandler.__instance is None:
            InputHandler()
        return InputHandler.__instance
```

### Why this is singleton evidence
The class constrains callers to a single reusable handler object.

## Evidence summary
Each handler includes the canonical singleton triad: class-level instance storage, guarded construction, and a static instance accessor.
