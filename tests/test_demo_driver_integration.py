from demo_driver import build_demo_records
from psu_capstone.agent_layer.HTM import ColumnField, InputField
from psu_capstone.agent_layer.brain import Brain
from psu_capstone.encoder_layer.encoder_handler import EncoderHandler
from psu_capstone.sdr_layer.sdr import SDR


def _apply_sdr_to_input_field(sdr: SDR, input_field: InputField) -> None:
    dense = sdr.get_dense()
    assert len(dense) == len(input_field.cells)
    input_field.advance_states()
    for idx, cell in enumerate(input_field.cells):
        if dense[idx]:
            cell.set_active()  # type: ignore[attr-defined]


def test_demo_driver_stream_to_brain_and_htm():
    raw_records = build_demo_records()
    assert raw_records

    handler = EncoderHandler(raw_records)
    composite_sdrs = handler.build_composite_sdr(raw_records)

    input_field = InputField(size=composite_sdrs[0].size)
    column_field = ColumnField(
        input_fields=[input_field],
        cells_per_column=4,
        non_spatial=True,
    )
    brain = Brain({"input": input_field, "columns": column_field})

    for sdr in composite_sdrs:
        assert sdr.size == composite_sdrs[0].size
        assert sdr.get_sparsity() > 0.0
        _apply_sdr_to_input_field(sdr, input_field)
        brain.compute_only(learn=True)
        assert column_field.active_columns
