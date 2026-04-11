from abc import ABC, abstractmethod
from typing import Any


class AbstractBrain(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, inputs: dict[str, Any]):
        raise NotImplementedError("Subclasses must implement the step method")

    @abstractmethod
    def prediction(self) -> tuple[Any, ...]:
        raise NotImplementedError("Subclasses must implement the prediction method")

    @abstractmethod
    def encode_only(self, inputs: dict[str, Any]):
        raise NotImplementedError("Subclasses must implement the encode only method")

    @abstractmethod
    def compute_only(self, learn: bool = True):
        raise NotImplementedError("Subclasses must implement the compute only method")

    @abstractmethod
    def print_stats(self):
        raise NotImplementedError("Subclasses must implement the print states method")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("Subclasses must implement the reset method")
