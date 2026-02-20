import os

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

    for k in data.keys():
        columns.append(k)
        k = data[k] + "_input"
        print(f"Column: {k}, Sample Value: {data[k][0]} is of type {type(data[k][0])}")
