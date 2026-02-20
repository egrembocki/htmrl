"""Train Model facade to pre-train the Brain on a dataset.

Manager of Brains to build and train them on datasets. Provides a high-level API for training
without needing to interact with the Brain's internal fields directly.

"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from joblib import dump, load

from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.HTM import ColumnField, Field, InputField, OutputField
from psu_capstone.encoder_layer.base_encoder import ParentDataClass
from psu_capstone.encoder_layer.category_encoder import CategoryParameters
from psu_capstone.encoder_layer.date_encoder import DateEncoderParameters
from psu_capstone.encoder_layer.fourier_encoder import FourierEncoderParameters
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.log import get_logger, logger


class Trainer:
    """Build a Trainer for training Brains on a dataset."""

    def __init__(self, brain: Brain) -> None:
        """Initializes the Trainer with a Brain instance."""
        self.logger = get_logger(self)
        self._main_brain: Brain = brain
        self._brains: list[Brain] = []
        self._trainer_input_fields: list[Field] = []
        self._trainer_output_fields: list[Field] = []
        self._trainer_column_fields: list[Field] = []
        self._values: list[Any] = []

    @property
    def brains(self) -> list[Brain]:
        """Access the Brains being trained."""

        return self._brains

    @property
    def main_brain(self) -> Brain:
        """Access the main Brain being trained."""
        if self._main_brain is None:
            raise ValueError("Main Brain is not initialized. Please build the Brain first.")
        return self._main_brain

    @main_brain.setter
    def main_brain(self, brain: Brain) -> None:
        """Set the main Brain being trained."""
        if brain is None:
            raise ValueError("Cannot set main Brain to None.")
        elif not isinstance(brain, Brain):
            raise ValueError("main_brain must be an instance of Brain.")

        self._main_brain = brain
        if brain not in self._brains:
            self._brains.append(brain)

    def _setup_io_fields(self, fields: list[tuple[str, int, ParentDataClass]]) -> None:
        """Setup the fields for the Brain through the passed in tuple.

        Args:
            fields: A list of tuples containing field name, field size, and encoder parameters.

            Example:
            fields = [
                ("consumption_input", 12288, RDSEParameters(size=12288, sparsity=0.02, resolution=0.001, category=False, seed=5)),
                ("date_input", 12288, DateEncoderParameters(size=12288, resolution=0.001, seed=5)),

            ]

            Note: Field names ending with '_input' will automatically be treated as InputFields.
        """

        for name, size, param in fields:
            self.logger.info(
                "Setting up field %-24s size=%5d encoder=%-28s",
                name,
                size,
                type(param).__name__,
            )
            if isinstance(param, RDSEParameters):
                encoder_params = RDSEParameters(
                    size=param.size,
                    sparsity=param.sparsity,
                    resolution=param.resolution,
                    category=param.category,
                    seed=param.seed,
                )
            elif isinstance(param, DateEncoderParameters):
                encoder_params = DateEncoderParameters(size=param.size)

            elif isinstance(param, CategoryParameters):
                encoder_params = CategoryParameters(size=param.size)
            elif isinstance(param, FourierEncoderParameters):
                encoder_params = FourierEncoderParameters(size=param.size)
            else:
                raise ValueError("Unsupported encoder parameters type: {}".format(type(param)))

            if name.endswith("_input"):
                field = InputField(size=size, encoder_params=encoder_params)
                field.name = name
            elif name.endswith("_output"):
                field = OutputField(size=size, motor_action=(None,))
                field.name = name
            else:
                raise ValueError("Unsupported field type: {}".format(name))

            if isinstance(field, InputField):
                self._trainer_input_fields.append(field)
            elif isinstance(field, OutputField):
                self._trainer_output_fields.append(field)

    def _setup_column_fields(self, num_columns: int, cells_per_column: int) -> None:
        """Setup the ColumnField for the Brain."""
        column_field = ColumnField(
            input_fields=self._trainer_input_fields,
            non_spatial=True,
            num_columns=num_columns,
            cells_per_column=cells_per_column,
        )
        column_field.name = "column"
        self._trainer_column_fields.append(column_field)

    def _create_brain(self) -> Brain:
        """Create a Brain instance with the configured fields."""

        if not self._trainer_input_fields:
            raise ValueError("No input fields defined for the Brain.")
        if not self._trainer_column_fields:
            raise ValueError("No column fields defined for the Brain.")

        brain = Brain(
            {
                **{field.name: field for field in self._trainer_input_fields if field is not None},
                **{field.name: field for field in self._trainer_output_fields if field is not None},
                **{field.name: field for field in self._trainer_column_fields if field is not None},
            }
        )

        self.logger.info("Brain created with fields: %s", list(brain.fields.keys()))

        return brain

    def _build_values_list(self, dataset: dict[str, list[Any]]) -> None:
        """Build a list of values from the dataset for training."""

        for column_name, column_data in dataset.items():
            self.logger.debug(
                f"Processing column '{column_name}' with {len(column_data)} data points."
            )
            for value in column_data:
                self._values.append(value)
        self.logger.info(
            f"Built values list with {len(self._values)} total data points from dataset."
        )

    def print_train_stats(self) -> None:
        """Print statistics about the training dataset."""
        self.logger.info("Training dataset statistics:")
        self._main_brain.print_stats()

    def build_brain(self, fields: list[tuple[str, int, ParentDataClass]]) -> Brain:
        """Build the Brain for training. Building the Brain this way allows for more direct control over the fields and their parameters, which can be crucial for effective training.

        Args:
            fields: A list of tuples containing field name, field size, and encoder parameters.

                Example:
                fields = [
                    ("consumption_input", 2048, RDSEParameters(size=2048, sparsity=0.02, resolution=1.0, category=False, seed=5)),
                    ("date_input", 2048, DateEncoderParameters(size=2048, resolution=1.0, seed=5)),
                ]
        Returns:
            The built Brain instance ready for training.

        """
        # build a brain based on the provided fields, using the maximum size from the fields to determine the number of columns in the column field and the size of the output SDR. This ensures that the column field can accommodate the largest input field

        size = max(field[1] for field in fields)  # Get the maximum size from the fields
        # make the input/output fields
        self._setup_io_fields(fields)
        # make column fields
        self._setup_column_fields(num_columns=size, cells_per_column=32)
        # create the Brain with fields
        brain = self._create_brain()
        self._main_brain = brain
        self._brains.append(brain)

        self.save_brain_state(brain, "./model/initial_brain_state.joblib")

        return brain

    def add_input_field(self, name: str, size: int, encoder_params: ParentDataClass) -> None:
        """Add an input field to the Brain."""

        if self._main_brain is None:
            raise ValueError("Main Brain is not initialized. Please build the Brain first.")

        if name.endswith("_input"):
            field = InputField(size=size, encoder_params=encoder_params)
            field.name = name
            self._trainer_input_fields.append(field)
            self._main_brain.fields[name] = field
        else:
            raise ValueError("Input field name must end with '_input'.")

    def add_output_field(self, name: str, size: int, motor_action: tuple[Any, ...]) -> None:
        """Add an output field to the Brain."""

        if self._main_brain is None:
            raise ValueError("Main Brain is not initialized. Please build the Brain first.")

        if name.endswith("_output"):
            field = OutputField(size=size, motor_action=motor_action)
            field.name = name
            self._trainer_output_fields.append(field)
            self._main_brain.fields[name] = field
        else:
            raise ValueError("Output field name must end with '_output'.")

    def add_column_field(self, name: str, num_columns: int, cells_per_column: int) -> None:
        """Add a column field to the Brain."""

        if self._main_brain is None:
            raise ValueError("Main Brain is not initialized. Please build the Brain first.")

        if name.endswith("_column"):
            field = ColumnField(
                input_fields=self._trainer_input_fields,
                non_spatial=True,
                num_columns=num_columns,
                cells_per_column=cells_per_column,
            )
            field.name = name
            self._trainer_column_fields.append(field)
            self._main_brain.fields[name] = field
        else:
            raise ValueError("Column field name must end with '_column'.")

    def train_column(self, brain: Brain | None, column: dict[str, list[Any]], steps: int) -> None:
        """Train the Brain on the specified dataset."""

        if brain is None:
            brain = self._main_brain
        if steps <= 1:
            steps = len(column)

        if len(column.keys()) == 0:
            raise ValueError("Dataset is empty. Cannot train on an empty dataset.")
        elif len(column.keys()) > 1:
            raise ValueError(
                "Dataset has more than one column. Use train_brain() to train on multiple columns."
            )
        else:

            self.logger.info(f"Training one column with {steps} data points.")

        for step in range(steps):
            # Get the current data point from the column
            name = list(column.keys())[0]
            value = column[name][step % len(column[name])]

            for field in self._trainer_input_fields:
                if field.name == f"{name}_input":
                    input_dict = {field.name: value}
                    break
            else:
                raise ValueError(f"No matching input field found for column '{name}'.")

            # Step the Brain with the prepared inputs
            brain.step(inputs=input_dict, learn=True)

    def train_brain(self, brain: Brain | None, dataset: dict[str, list[Any]], steps: int) -> None:
        """Train the Brain on the specified dataset."""
        if brain is None:
            brain = self._main_brain
        if steps <= 1:
            steps = min(len(column) for column in dataset.values())

        self.logger.info(f"Training on dataset with columns: {list(dataset.keys())}")

        for column_name, column_data in dataset.items():
            self.logger.debug(f"Column '{column_name}' has {len(column_data)} data points.")

        for step in range(steps):
            # Get the current data point from the dataset
            column_data = {key: dataset[key][step % len(dataset[key])] for key in dataset}
            # avoid index out of range by looping over dataset if steps exceed its size
            self.logger.debug(f"Step {step + 1}/{steps}, Data Point: {column_data}")

            # Prepare the input dictionary for the Brain
            input_dict = {}
            for field in self._trainer_input_fields:
                # Check if the field name exists in the data point and add it to the input dictionary
                if field.name in column_data:
                    input_dict[field.name] = column_data[field.name]
                else:
                    raise ValueError(f"Data point is missing required input field: {field.name}")

            # Step the Brain with the prepared inputs
            brain.step(inputs=input_dict, learn=True)

            # self.save_brain_state(brain, f"./model/brain_state_step_{step + 1}.joblib")

    def test(self, brain: Brain, test_dataset: Any, expected_data: Any) -> None:
        """Test the Brain on the specified dataset."""
        self.logger.info(f"Testing on dataset: {test_dataset}")
        # Implement testing logic as needed, e.g., evaluating predictions against expected outputs

    def build_full_brain(self, dataset: dict[Any, list[Any]], size: int) -> Brain:
        """Build a full Brain with all fields based on the dataset."""

        for key, values in dataset.items():
            if isinstance(values[0], (int, float)):
                encoder_params = RDSEParameters(size=size)
            elif isinstance(values[0], str):
                encoder_params = CategoryParameters(size=size)
            elif isinstance(values[0], (list, tuple, np.ndarray)):
                encoder_params = FourierEncoderParameters(size=size)
            elif isinstance(values[0], datetime):
                encoder_params = DateEncoderParameters(size=size)
            else:
                raise ValueError(f"Unsupported data type for field '{key}': {type(values[0])}")

            self._setup_io_fields([(f"{key}_input", size, encoder_params)])

        self._setup_column_fields(num_columns=size, cells_per_column=32)

        brain = self._create_brain()

        self._main_brain = brain
        self._brains.append(brain)

        return brain

    def save_brain_state(self, brain: Brain | None, path: str) -> None:
        """Save the Brain's state to the specified path."""
        if brain is None:
            brain = self._main_brain
        self.logger.info(f"Saving Brain state to: {path}")

        dump(brain, path)

    def load_brain_state(self, path: str) -> Brain:
        """Load the Brain's state from the specified path."""
        self.logger.info(f"Loading Brain state from: {path}")
        brain = load(path)
        self._main_brain = brain
        return brain
