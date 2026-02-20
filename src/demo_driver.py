import os
from typing import Any

from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.train import Trainer
from psu_capstone.input_layer.input_handler import InputHandler
from utils import DATA_PATH, PROJECT_ROOT

if __name__ == "__main__":
    # Example usage of the Brain and Trainer classes
    brain = Brain()
    trainer = Trainer(brain)
    input_handler = InputHandler()

    PATH = os.path.join(DATA_PATH, "concat_ESData.xlsx")

    data = input_handler.input_data(PATH)

    columns: list[str] = []
    values: list[list[Any]] = []

    for k, v in data.items():
        columns.append(k)
        values.append(v)

    trainer.main_brain = trainer.build_full_brain(data, 2048)

    print("Trainer Fields:")
    for field in trainer._trainer_input_fields:
        print(f"- {field.name}")

    print("Brains: Input Fields:")
    for field in trainer._main_brain._input_fields.values():
        print(f"- {field.name}")
    print("Brains: Column Fields:")
    for field in trainer._main_brain._column_fields.values():
        print(f"- {field.name}")
