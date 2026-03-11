"""Tests for Trainer brain creation with proper InputFields, OutputFields, and ColumnFields."""

import pytest

import psu_capstone.encoder_layer as el
from psu_capstone.agent_layer.brain import Brain
from psu_capstone.agent_layer.HTM import ColumnField, InputField, OutputField
from psu_capstone.agent_layer.train import Trainer


@pytest.fixture
def trainer():
    brain = Brain()
    return Trainer(brain)


# ---------------------------------------------------------------------------
# build_brain
# ---------------------------------------------------------------------------


class TestBuildBrain:
    def test_single_input_field_is_present_in_brain(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert "value_input" in brain.fields

    def test_single_input_field_is_instance_of_input_field(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["value_input"], InputField)

    def test_column_field_is_created_automatically(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert len(brain.column_fields) == 1

    def test_column_field_is_instance_of_column_field(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.column_fields[0], ColumnField)

    def test_multiple_input_fields_all_present(self, trainer):
        fields = [
            ("temp_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
            ("humidity_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
        ]
        brain = trainer.build_brain(fields)
        assert "temp_input" in brain.fields
        assert "humidity_input" in brain.fields

    def test_multiple_input_fields_are_all_input_field_instances(self, trainer):
        fields = [
            ("temp_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
            ("humidity_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
        ]
        brain = trainer.build_brain(fields)
        assert all(
            isinstance(brain.fields[k], InputField) for k in ("temp_input", "humidity_input")
        )

    def test_input_field_size_matches_requested_size(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert len(brain.fields["value_input"].cells) == 512

    def test_brain_has_no_output_fields_when_none_defined(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert len(brain.output_fields) == 0

    def test_invalid_field_name_raises(self, trainer):
        fields = [("value", 512, el.RDSEParameters(size=512, resolution=1.0))]
        with pytest.raises(ValueError):
            trainer.build_brain(fields)

    def test_column_field_num_columns_equals_max_field_size(self, trainer):
        fields = [
            ("small_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
            ("large_input", 1024, el.RDSEParameters(size=1024, resolution=1.0)),
        ]
        brain = trainer.build_brain(fields)
        assert brain.column_fields[0].num_columns == 1024

    def test_build_brain_returns_brain_instance(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain, Brain)

    def test_built_brain_stored_as_main_brain(self, trainer):
        fields = [("value_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert trainer.main_brain is brain

    def test_category_encoder_input_field_is_created(self, trainer):
        fields = [
            ("label_input", 512, el.CategoryParameters(size=512, category_list=["a", "b", "c"]))
        ]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["label_input"], InputField)
        assert isinstance(brain.fields["label_input"].encoder, el.CategoryEncoder)

    def test_scalar_encoder_input_field_is_created(self, trainer):
        fields = [
            ("score_input", 512, el.ScalarEncoderParameters(size=512, minimum=0, maximum=100))
        ]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["score_input"], InputField)
        assert isinstance(brain.fields["score_input"].encoder, el.ScalarEncoder)

    def test_date_encoder_input_field_is_created(self, trainer):
        fields = [("date_input", 512, el.DateEncoderParameters(size=512))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["date_input"], InputField)
        assert isinstance(brain.fields["date_input"].encoder, el.DateEncoder)

    def test_fourier_encoder_input_field_is_created(self, trainer):
        fields = [("fourier_input", 512, el.FourierEncoderParameters(size=512))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["fourier_input"], InputField)
        assert isinstance(brain.fields["fourier_input"].encoder, el.FourierEncoder)

    def test_geospatial_encoder_input_field_is_created(self, trainer):
        fields = [("geo_input", 512, el.GeospatialParameters(size=512))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["geo_input"], InputField)
        assert isinstance(brain.fields["geo_input"].encoder, el.GeospatialEncoder)

    def test_coordinate_encoder_input_field_is_created(self, trainer):
        fields = [("coord_input", 512, el.CoordinateParameters(size=512))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["coord_input"], InputField)
        assert isinstance(brain.fields["coord_input"].encoder, el.CoordinateEncoder)

    def test_rdse_encoder_input_field_is_created(self, trainer):
        fields = [("rdse_input", 512, el.RDSEParameters(size=512, resolution=1.0))]
        brain = trainer.build_brain(fields)
        assert isinstance(brain.fields["rdse_input"], InputField)
        assert isinstance(brain.fields["rdse_input"].encoder, el.RandomDistributedScalarEncoder)

    def test_multiple_encoder_types_all_input_fields_created(self, trainer):
        fields = [
            ("category_input", 512, el.CategoryParameters(size=512, category_list=["a", "b", "c"])),
            ("scalar_input", 512, el.ScalarEncoderParameters(size=512, minimum=0, maximum=100)),
            ("date_input", 512, el.DateEncoderParameters(size=512)),
            ("fourier_input", 512, el.FourierEncoderParameters(size=512)),
            ("geo_input", 512, el.GeospatialParameters(size=512)),
            ("coord_input", 512, el.CoordinateParameters(size=512)),
            ("rdse_input", 512, el.RDSEParameters(size=512, resolution=1.0)),
        ]
        brain = trainer.build_brain(fields)
        for name in [
            "category_input",
            "scalar_input",
            "date_input",
            "fourier_input",
            "geo_input",
            "coord_input",
            "rdse_input",
        ]:
            assert isinstance(brain.fields[name], InputField)
            assert isinstance(
                brain.fields[name].encoder,
                (
                    el.CategoryEncoder,
                    el.ScalarEncoder,
                    el.DateEncoder,
                    el.FourierEncoder,
                    el.GeospatialEncoder,
                    el.CoordinateEncoder,
                    el.RandomDistributedScalarEncoder,
                ),
            )


# ---------------------------------------------------------------------------
# add_input_field / add_output_field / add_column_field
# ---------------------------------------------------------------------------


class TestAddFields:
    def test_add_input_field_adds_to_brain_fields(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        trainer.add_input_field("extra_input", 512, el.RDSEParameters(size=512, resolution=1.0))
        assert "extra_input" in trainer.main_brain.fields

    def test_add_input_field_is_input_field_instance(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        trainer.add_input_field("extra_input", 512, el.RDSEParameters(size=512, resolution=1.0))
        assert isinstance(trainer.main_brain.fields["extra_input"], InputField)

    def test_add_input_field_bad_name_raises(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        with pytest.raises(ValueError):
            trainer.add_input_field("extra", 512, el.RDSEParameters(size=512, resolution=1.0))

    def test_add_output_field_is_output_field_instance(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        trainer.add_output_field("motor_output", 128, motor_action=(None,))
        assert isinstance(trainer.main_brain.fields["motor_output"], OutputField)

    def test_add_output_field_bad_name_raises(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        with pytest.raises(ValueError):
            trainer.add_output_field("motor", 128, motor_action=(None,))

    def test_add_column_field_is_column_field_instance(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        trainer.add_column_field("extra_column", num_columns=512, cells_per_column=8)
        assert isinstance(trainer.main_brain.fields["extra_column"], ColumnField)

    def test_add_column_field_bad_name_raises(self, trainer):
        trainer.build_brain([("base_input", 512, el.RDSEParameters(size=512, resolution=1.0))])
        with pytest.raises(ValueError):
            trainer.add_column_field("extra", num_columns=512, cells_per_column=8)
