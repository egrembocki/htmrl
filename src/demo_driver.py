import os
from typing import Any

from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.train import Trainer
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.log import logger
from utils import DATA_PATH, PROJECT_ROOT


def fin_data_demo() -> None:
    """Demonstrate loading and visualizing data from the dataset."""

    ih = InputHandler()
    data = ih.input_data(os.path.join(PROJECT_ROOT, "data", "concat_ESData.xlsx"))

    brain = Brain()
    trainer = Trainer(brain)
    brain = trainer.fast_build_brain(data, 2048)

    for name, value in data.items():
        logger.info("Data column '%s': %d records", name, len(value))

        trainer.train_column(brain, column={name: value}, steps=5)

    trainer._main_brain.print_stats()


def sine_wave_demo() -> None:
    """Demonstrate encoding and learning on a simple sine wave dataset."""

    import numpy as np

    # Generate a sine wave dataset
    x = np.linspace(0, 1, 100, endpoint=False)
    y = np.sin(2 * np.pi * 1 * x)
    data = {"sine_wave_input": y}
    brain = Brain()
    trainer = Trainer(brain)
    trainer.main_brain = trainer.build_brain([("sine_wave_input", 2048, RDSEParameters())])

    for name, value in data.items():
        logger.info("Data column '%s': %d records", name, len(value))

    column = {"sine_wave_input": y.tolist()}

    trainer.train_column(trainer.main_brain, column, steps=2048)

    trainer.show_active_columns(trainer.main_brain)

    trainer.main_brain.print_stats()


if __name__ == "__main__":
    # Example usage of the Brain and Trainer classes

    # fin_data_demo()
    sine_wave_demo()

    """
    print("Trainer Fields:")
    for field in trainer._trainer_input_fields:
        print(f"- {field.name}")

    print("Brains: Input Fields:")
    for field in trainer.main_brain._input_fields.values():
        print(f"- {field.name}")
    print("Brains: Column Fields:")
    for field in trainer.main_brain._column_fields.values():
        print(f"- {field.name}")
   """
