import os

import grapher
from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.train import Trainer
from psu_capstone.encoder_layer.base_encoder import ParentDataClass
from psu_capstone.encoder_layer.rdse import RDSEParameters
from psu_capstone.input_layer.input_handler import InputHandler
from psu_capstone.log import logger
from utils import DATA_PATH, PROJECT_ROOT

ESD = os.path.join(DATA_PATH, "concat_ESData.xlsx")
DATA_COLUMN_LOG_MESSAGE = "Data column '%s': %d records"


def fin_data_demo(column: str) -> None:
    """Demonstrate loading and visualizing data from the dataset."""

    ih = InputHandler()
    data = ih.input_data(ESD)

    brain = Brain()
    trainer = Trainer(brain)
    brain = trainer.fast_build_brain(data, 2048)

    if not column:
        for name, value in data.items():
            logger.info(DATA_COLUMN_LOG_MESSAGE, name, len(value))

            trainer.train_column(brain, column={name: value}, steps=100)

    elif column not in data:
        logger.error("Specified column '%s' not found in dataset.", column)
        return
    else:
        trainer.train_column(brain, column={f"{column}_input": data[column]}, steps=100)

    trainer._main_brain.print_stats()

    # trainer.test(trainer._main_brain, {f"{column}_input": data[column]}, steps=100)

    trainer.show_active_columns(trainer._main_brain)
    trainer.show_heat_map(trainer._main_brain)


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
        logger.info(DATA_COLUMN_LOG_MESSAGE, name, len(value))

    column = {"sine_wave_input": y.tolist()}

    trainer.train_column(trainer.main_brain, column, steps=100)

    # show predicted vs actual values for the last 100 steps
    trainer.test(trainer.main_brain, column, steps=100)

    trainer.show_active_columns(trainer.main_brain)
    trainer.show_heat_map(trainer.main_brain)

    trainer.main_brain.print_stats()


def show_input_data_demo() -> None:
    """Demonstrate loading and visualizing data from the dataset."""

    ih = InputHandler()
    data = ih.input_data(ESD)

    for name, value in data.items():
        logger.info(DATA_COLUMN_LOG_MESSAGE, name, len(value))


def show_brain_creation_demo() -> None:
    """Demonstrate creating a Brain and inspecting its structure."""

    brain = Brain()
    trainer = Trainer(brain)

    input_fields: list[tuple[str, int, ParentDataClass]] = []

    # Example of building a brain with specific input fields
    input_fields = [
        ("temperature_input", 2048, RDSEParameters()),
        ("humidity_input", 2048, RDSEParameters()),
        ("energy_consumption_input", 2048, RDSEParameters()),
    ]
    trainer.main_brain = trainer.build_brain(input_fields)

    print("Trainer Fields:")
    for field in trainer._trainer_input_fields:
        print(f"- {field.name}")

    print("Brains: Input Fields:")
    for field in trainer.main_brain._input_fields.values():
        print(f"- {field.name}")
    print("Brains: Column Fields:")
    for field in trainer.main_brain._column_fields.values():
        print(f"- {field.name}")


def show_field_encoding_demo() -> None:
    """Demonstrate encoding a sample input through the Brain's input fields."""

    brain = Brain()
    trainer = Trainer(brain)

    input_fields: list[tuple[str, int, ParentDataClass]] = []
    # Create a simple brain with one input field and one column field
    input_fields = [
        ("temperature_input", 2048, RDSEParameters()),
    ]
    trainer.main_brain = trainer.build_brain(input_fields)

    # Example input value to encode
    sample_input = {"temperature_input": 25.0}

    # Step the brain with the sample input
    output = trainer.main_brain._input_fields["temperature_input"].encode(
        sample_input["temperature_input"]
    )

    print("Encoded Output:")
    print(output)

    grapher.plot_sdr(output)


def show_input_to_encoder_demo() -> None:
    """Demonstrate the InputHandler's ability to convert raw input data to encoder-ready format."""

    ih = InputHandler()
    brain = Brain()
    trainer = Trainer(brain)

    ih.input_data(ESD)
    encoder_sequence = ih.get_column_data(column="Open")

    trainer.main_brain = trainer.build_brain([("Open_input", 2048, RDSEParameters())])

    values = encoder_sequence[:5]  # Take the first 5 values for demonstration

    field = trainer.main_brain._input_fields["Open_input"]
    encoder = field.encoder

    for value in values:
        encoded = field.encode(value)
        decoded_value, confidence = field.decode("active", field, encoder._encoding_cache)  # type: ignore
        print(f"Decoded: {decoded_value} (Confidence: {confidence})")
        grapher.plot_sdr(encoded)


if __name__ == "__main__":
    # Example usage of the Brain and Trainer classes

    # show_input_data_demo()
    # show_input_to_encoder_demo()
    # show_field_encoding_demo()
    # show_brain_creation_demo()
    # sine_wave_demo()
    fin_data_demo("Open")
