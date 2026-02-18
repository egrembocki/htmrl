"""Train Model facade to pre-train the Brain on a dataset."""

from __future__ import annotations

from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.HTM import ColumnField, Field, InputField, OutputField


class TrainModel:
    """Facade for training the Brain on a dataset."""

    def __init__(self, brain: Brain) -> None:

        self._brain = brain

    @property
    def brain(self) -> Brain:
        """Access the Brain being trained."""
        return self._brain

    def build_brain(self) -> None:
        """Build the Brain for training."""
        print("Building the Brain for training.")

    def train(self, dataset: str) -> None:
        """Train the Brain on the specified dataset."""
        print(f"Training on dataset: {dataset}")

    def save_brain_state(self, path: str) -> None:
        """Save the Brain's state to the specified path."""
        print(f"Saving Brain state to: {path}")
