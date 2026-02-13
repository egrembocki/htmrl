import pytest

from psu_capstone.agent_layer import HTM
from psu_capstone.agent_layer.HTM import Column, InputField, OutputField, ProximalSynapse, Segment
from psu_capstone.encoder_layer.rdse import RDSEParameters

"""++++++++++Column Testing++++++++++"""


def make_input_field_helper() -> InputField:
    """Create an input field and get some cells active."""
    parameters = RDSEParameters()
    in_fi = InputField(parameters)
    in_fi.encode(15)
    return in_fi


def test_column_receptive_field_pct_sample_is_correct_size():
    """Checks that the receptive field is the correct size based on the HTM constant."""
    in_fi = make_input_field_helper()
    c = Column(in_fi)

    expected = int(len(in_fi.cells) * HTM.RECEPTIVE_FIELD_PCT)

    assert len(c.receptive_field) == expected


def test_column_potential_synapses_created_from_receptive_field():
    """Checks that the potential synapses are initialized from the receptive field."""
    in_fi = make_input_field_helper()
    c = Column(in_fi)

    assert len(c.potential_synapses) == len(c.receptive_field)

    receptive_cells = c.receptive_field
    synapse_sources = {syn.source_cell for syn in c.potential_synapses}

    assert synapse_sources == receptive_cells

    assert len(synapse_sources) == len(c.potential_synapses)

    assert all(isinstance(syn, ProximalSynapse) for syn in c.potential_synapses)


def test_column_wrong_field_entry():
    """Makes sure that the field is an input field when wrong entry."""
    in_fi = OutputField()
    c = Column(in_fi)
    assert isinstance(c.input_field, InputField)


def test_column_negative_cells_per_column():
    """Checks to make sure negative column entries are caught."""
    in_fi = make_input_field_helper()

    with pytest.raises(ValueError):
        Column(in_fi, -3)


def test_clear_state_resets_all_flags():
    """Checks that clear state works correctly."""
    in_fi = make_input_field_helper()
    c = Column(in_fi)

    c.clear_state()

    assert not c.active
    assert not c.prev_active
    assert not c.bursting
    assert not c.prev_bursting
    assert not c.predictive
    assert not c.prev_predictive


def test_compute_overlap_counts_active_sources():
    """Checks the overlap count for source cells and the connected synapses."""
    in_fi = make_input_field_helper()
    c = Column(in_fi)

    """Force synapses to be connected"""
    for i, syn in enumerate(c.potential_synapses[:5]):
        syn.permanence = HTM.CONNECTED_PERM + 0.1

    c._update_connected_synapses()

    for syn in c.connected_synapses[:5]:
        syn.source_cell.active = True
    """This should return the sum of the active source cells for the connected synapses."""
    c.compute_overlap()

    assert c.overlap == 5


def test_update_connected_synapses_with_negative_connected_perm():
    """Connected permanence should be a postive number."""
    in_fi = make_input_field_helper()
    c = Column(in_fi)
    with pytest.raises(ValueError):
        c._update_connected_synapses(-5)
