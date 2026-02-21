from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.HTM import ColumnField, Field, InputField, OutputField
from psu_capstone.encoder_layer.rdse import RDSEParameters


def create_brain_helper_multi_field() -> Brain:
    rdse_params = RDSEParameters(
        size=512,
        active_bits=0,
        sparsity=0.02,
        resolution=0.001,
        category=False,
        seed=5,
    )
    out_fi = OutputField(size=512, motor_action=(None,))
    input_field = InputField(size=512, encoder_params=rdse_params)
    column_field = ColumnField(
        input_fields=[input_field],
        non_spatial=True,
        num_columns=512,
        cells_per_column=16,
    )
    b = Brain(
        {
            "input": input_field,
            "output": out_fi,
            "column": column_field,
        }
    )
    return b


def create_brain_helper_single_field() -> Brain:
    rdse_params = RDSEParameters(
        size=512,
        active_bits=0,
        sparsity=0.02,
        resolution=0.001,
        category=False,
        seed=5,
    )
    input_field = InputField(size=512, encoder_params=rdse_params)
    column_field = ColumnField(
        input_fields=[input_field],
        non_spatial=True,
        num_columns=512,
        cells_per_column=16,
    )
    b = Brain(
        {
            "input": input_field,
            "column": column_field,
        }
    )
    return b


def test_initialize_brain():
    """Test initialize the brain with all field types."""
    b = create_brain_helper_multi_field()
    """
    Note: failing test, the output field is being counted as an output
    field and input field.
    """
    assert len(b._input_fields) == 1
    assert len(b._output_fields) == 1
    assert len(b._column_fields) == 1


def test_get_field():
    b = create_brain_helper_multi_field()
    input_field = b.__getitem__("input")
    output_field = b.__getitem__("output")
    column_field = b.__getitem__("column")
    assert isinstance(input_field, InputField)
    assert isinstance(output_field, OutputField)
    assert isinstance(column_field, ColumnField)


def test_step_and_prediction_method():
    b = create_brain_helper_single_field()
    b.step({"input": 10})
    prediction = b.prediction()["input"]
    assert prediction == 10
