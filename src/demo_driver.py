import os
from typing import Any

from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.train import Trainer
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.log import logger
from utils import DATA_PATH, PROJECT_ROOT

if __name__ == "__main__":
    # Example usage of the Brain and Trainer classes
    brain = Brain()

    trainer = Trainer(brain)
    input_handler = InputHandler()

    # brain = trainer.load_brain_state("./model/full_initial_brain_state.joblib")

    PATH = os.path.join(DATA_PATH, "concat_ESData.xlsx")

    data = input_handler.input_data(PATH)
    brain = trainer.build_full_brain(data, 2048)

    for name, value in data.items():
        logger.info("Data column '%s': %d records", name, len(value))

        trainer.train_column(brain, column={name: value}, steps=5)

    trainer._main_brain.print_stats()

    """
    print("Trainer Fields:")
    for field in trainer._trainer_input_fields:
        print(f"- {field.name}")

    print("Brains: Input Fields:")
    for field in trainer._main_brain._input_fields.values():
        print(f"- {field.name}")
    print("Brains: Column Fields:")
    for field in trainer._main_brain._column_fields.values():
        print(f"- {field.name}")
   """
