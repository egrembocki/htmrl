import os

from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.train import Trainer
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.log import logger
from utils import DATA_PATH, PROJECT_ROOT


def fin_data_demo(column: str) -> None:
    """Demonstrate loading and visualizing data from the dataset."""

    ih = InputHandler()
    data = ih.input_data(os.path.join(PROJECT_ROOT, "data", "concat_ESData.xlsx"))

    brain = Brain()
    trainer = Trainer(brain)
    brain = trainer.fast_build_brain(data, 2048)

    if not column:
        for name, value in data.items():
            logger.info("Data column '%s': %d records", name, len(value))

            trainer.train_column(brain, column={name: value}, steps=10)
    else:
        trainer.train_column(brain, column={column: data[column]}, steps=10)

    trainer._main_brain.print_stats()


def sine_wave_demo() -> None:
    """Demonstrate encoding and learning on a simple sine wave dataset."""

    import numpy as np

    # Generate a sine wave dataset
    x = np.linspace(0, 1, 2048, endpoint=False)
    y = np.sin(2 * np.pi * 1 * x)
    data = {"sine_wave_input": y}
    brain = Brain()
    trainer = Trainer(brain)
    trainer.main_brain = trainer.build_brain([("sine_wave_input", 2048, RDSEParameters())])

    for name, value in data.items():
        logger.info("Data column '%s': %d records", name, len(value))

    column = {"sine_wave_input": y.tolist()}

    trainer.train_column(trainer.main_brain, column, steps=1000)

    trainer.show_active_columns(trainer.main_brain)
    trainer.show_heat_map(trainer.main_brain)

    trainer.main_brain.print_stats()


def show_input_data_demo(column: str) -> None:
    """Demonstrate loading and visualizing data from the dataset."""

    ih = InputHandler()
    data = ih.input_data(os.path.join(PROJECT_ROOT, "data", "concat_ESData.xlsx"))

    if not column:
        for name, value in data.items():
            logger.info("Data column '%s': %d records", name, len(value))
    else:
        logger.info("Data column '%s': %d records", column, len(data[column]))


def show_brain_creation_demo() -> None:
    """Demonstrate creating a Brain and inspecting its structure."""

    brain = Brain()
    trainer = Trainer(brain)

    # Example of building a brain with specific input fields
    input_fields = [
        ("temperature_input", 2048, RDSEParameters()),
        ("humidity_input", 2048, RDSEParameters()),
        ("energy_consumption_input", 2048, RDSEParameters()),
    ]
    trainer.main_brain = trainer.build_brain(input_fields)

    logger.info("Brain created with input fields:")
    for field in trainer.main_brain._input_fields.values():
        logger.info("- %s (size: %d)", field.name, field.size)

    print("Trainer Fields:")
    for field in trainer._trainer_input_fields:
        print(f"- {field.name}")

    print("Brains: Input Fields:")
    for field in trainer.main_brain._input_fields.values():
        print(f"- {field.name}")
    print("Brains: Column Fields:")
    for field in trainer.main_brain._column_fields.values():
        print(f"- {field.name}")


if __name__ == "__main__":
    # Example usage of the Brain and Trainer classes

    # show_brain_creation_demo()
    # fin_data_demo()
    # sine_wave_demo()
    show_input_data_demo(column="")
