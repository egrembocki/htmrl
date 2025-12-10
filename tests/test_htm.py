import numpy as np
import pytest

from psu_capstone.agent_layer.htm import (
    cell,
    column,
    constants,
    distal_synapse,
    htm_utils,
    segment,
    spatial_pooler,
    synapse,
    temporal_memory,
)

# Spatial Pooler constants
CONNECTED_PERM = 0.5  # Permanence threshold for connected proximal synapse
MIN_OVERLAP = 3  # Minimum overlap to be considered during inhibition
PERMANENCE_INC = 0.01
PERMANENCE_DEC = 0.01
DESIRED_LOCAL_ACTIVITY = 10

# Temporal Memory constants
SEGMENT_ACTIVATION_THRESHOLD = 3  # Active connected distal synapses required for prediction
SEGMENT_LEARNING_THRESHOLD = 3  # For best matching segment selection (reserved)
INITIAL_DISTAL_PERM = 0.21  # Initial permanence for new distal synapses
NEW_SYNAPSE_MAX = 6  # New distal synapses to add on reinforcement


class FakeSegment:
    """fake segment for testing"""

    pass


class FakeSynapse:
    """fake synapse for testing"""

    def __init__(self, source_input, permanence):
        self.source_input = source_input
        self.permanence = permanence


class FakeCell:
    """fake cell for testing"""

    def __init__(self, id):
        self.id = id


"""Grabbing these from the constants file"""
CONNECTED_PERM = constants.CONNECTED_PERM
MIN_OVERLAP = constants.MIN_OVERLAP


"""Cell Tests"""


def test_cell_initialization():
    """Cell should be created with empty segment list."""
    c = cell.Cell()
    assert isinstance(c.segments, list)
    assert c.segments == []


def test_cell_store_segments():
    """Cell should store fake segments."""
    c = cell.Cell()
    s1 = FakeSegment()
    s2 = FakeSegment()

    c.segments.append(s1)
    c.segments.append(s2)

    assert len(c.segments) == 2
    assert c.segments[0] is s1
    assert c.segments[1] is s2


def test_cell_repr():
    """repr should return cell (id)"""
    c = cell.Cell()
    r = repr(c)
    assert r.startswith("Cell(id=")
    assert r.endswith(")")


"""Column Tests"""


def test_column_initialization_sets_connected_synapses():
    """Connected synapses should be those with permanence."""
    s1 = FakeSynapse(0, 0.4)
    s2 = FakeSynapse(1, 0.6)
    s3 = FakeSynapse(2, 0.7)

    col = column.Column([s1, s2, s3], position=(0, 0))

    assert col.connected_synapses == [s2, s3]
    assert len(col.connected_synapses) == 2


def test_compute_overlap_zero_when_below_min():
    """If the overlap < min_overlap, it should be zero."""
    s1 = FakeSynapse(0, 0.8)
    s2 = FakeSynapse(1, 0.8)
    s3 = FakeSynapse(2, 0.8)

    col = column.Column([s1, s2, s3], position=(1, 2))

    input_vector = np.array([1, 1, 0])
    col.compute_overlap(input_vector)

    assert col.overlap == 0.0


def test_compute_overlap_applies_boost_if_above_threshold():
    """If overlap_raw > min overlap, overlap = overlap_raw*boost."""
    s1 = FakeSynapse(0, 0.9)
    s2 = FakeSynapse(1, 0.9)
    s3 = FakeSynapse(2, 0.9)
    s4 = FakeSynapse(3, 0.9)

    col = column.Column([s1, s2, s3, s4], position=(5, 5))
    col.boost = 2

    input_vector = np.array([1, 1, 1, 0])
    col.compute_overlap(input_vector)

    assert col.overlap == 6


def test_compute_overlap_no_connected_synapse_is_zero():
    """If there are none connected, it should be zero overlap."""
    s1 = FakeSynapse(0, 0.1)
    s2 = FakeSynapse(1, 0.2)
    s3 = FakeSynapse(2, 0.3)

    col = column.Column([s1, s2, s3], position=(3, 3))
    col.boost = 5

    input_vector = np.array([1, 1, 1])
    col.compute_overlap(input_vector)

    assert col.overlap == 0


def test_compute_overlap_counts_only_active():
    """Make sure only active synapse counted."""
    s1 = FakeSynapse(0, 0.9)
    s2 = FakeSynapse(1, 0.9)
    s3 = FakeSynapse(2, 0.9)

    col = column.Column([s1, s2, s3], position=(9, 9))
    col.boost = 1

    input_vector = np.array([1, 1, 1])
    col.compute_overlap(input_vector)

    assert col.overlap == 3


"""Distal Synapse Tests"""


def test_distal_synapse_initialization():
    """Make sure it can be created."""
    c = FakeCell(id=1)
    s = distal_synapse.DistalSynapse(source_cell=c, permanence=0.5)

    assert s.source_cell is c
    assert s.permanence == 0.5


def test_distal_synapse_allows_high_permanence():
    """Test for high permanence entry."""
    c = FakeCell(id=2)
    s = distal_synapse.DistalSynapse(source_cell=c, permanence=1.0)

    assert s.permanence == 1.0


def test_distal_synapse_allows_zero_permanence():
    """Test for zero permanence entry."""
    c = FakeCell(id=3)
    s = distal_synapse.DistalSynapse(source_cell=c, permanence=0)

    assert s.permanence == 0


def test_distal_synapse_negative_permanence():
    """Test for negative permanence entry."""
    c = FakeCell(id=4)
    s = distal_synapse.DistalSynapse(source_cell=c, permanence=-0.1)

    assert s.permanence == -0.1


def test_distal_synapse_is_reference_not_copy():
    """Test reference vs copy"""
    c = FakeCell(id=5)
    s = distal_synapse.DistalSynapse(source_cell=c, permanence=0.2)

    c.new_attr = "test"

    assert s.source_cell.new_attr == "test"


"""Segment Tests"""


def test_segment_initialization():
    """Test initialization is empty"""
    s = segment.Segment()
    assert s.synapses == []
    assert s.sequence_segment is False


def test_segment_init_with_synapses():
    """Test for segment creation with two synapses."""
    s1 = distal_synapse.DistalSynapse(FakeCell(1), permanence=0.2)
    s2 = distal_synapse.DistalSynapse(FakeCell(2), permanence=0.3)
    s = segment.Segment([s1, s2])

    assert len(s.synapses) == 2
    assert s.synapses[0] is s1
    assert s.synapses[1] is s2


"""Synapse Tests"""


def test_active_synapse_return_connected_and_active():
    """This should only return synapses whose source cell is active."""
    fc1 = FakeCell(1)
    fc2 = FakeCell(2)
    fc3 = FakeCell(3)
    s1 = distal_synapse.DistalSynapse(fc1, permanence=CONNECTED_PERM + 0.2)
    s2 = distal_synapse.DistalSynapse(fc2, permanence=CONNECTED_PERM - 0.01)
    s3 = distal_synapse.DistalSynapse(fc3, permanence=CONNECTED_PERM + 0.3)

    s = segment.Segment([s1, s2, s3])

    active_cells = {fc1, fc3}

    result = s.active_synapses(active_cells)
    assert result == [s1, s3]


def test_active_synapses_empty_when_no_active_cells():
    """If we have no active cells, the active synapses should be none."""
    s1 = distal_synapse.DistalSynapse(FakeCell(1), permanence=CONNECTED_PERM + 0.5)
    s = segment.Segment([s1])

    result = s.active_synapses(active_cells=set())
    assert result == []


def test_active_synapses_empty_when_no_connected():
    """If we have no connected, the active zynapses should be none."""
    s1 = distal_synapse.DistalSynapse(FakeCell(1), permanence=CONNECTED_PERM - 0.1)
    """we have an active cell."""
    active_cells = {FakeCell(1)}

    s = segment.Segment([s1])
    assert s.active_synapses(active_cells) == []


def test_active_synapse_empty_on_empty_seg():
    """If the segment is empty, there should be no active synapse."""
    s = segment.Segment()
    assert s.active_synapses({FakeCell(1)}) == []


def test_matching_synapses_returns_all_with_active_source():
    """Test when prev_active matches with current source."""
    fc1 = FakeCell(1)
    fc2 = FakeCell(2)
    fc3 = FakeCell(3)
    s1 = distal_synapse.DistalSynapse(fc1, 0)
    s2 = distal_synapse.DistalSynapse(fc2, 1)
    s3 = distal_synapse.DistalSynapse(fc3, 0.5)

    s = segment.Segment([s1, s2, s3])

    prev_active = {fc2, fc3}

    result = s.matching_synapses(prev_active)
    assert result == [s2, s3]


def test_matching_synapses_empty_when_no_prev_match():
    """With no prev_active, the matching should return none."""
    fc1 = FakeCell(1)
    fc2 = FakeCell(2)
    s1 = distal_synapse.DistalSynapse(fc1, 0.5)
    s2 = distal_synapse.DistalSynapse(fc2, 0.1)

    s = segment.Segment([s1, s2])

    prev_active = {FakeCell(155)}
    assert s.matching_synapses(prev_active) == []


def test_matching_synapse_empty_on_empty_seg():
    """Test to make sure a blank segment should have no matching synapses."""
    s = segment.Segment()
    assert s.matching_synapses({FakeCell(1)}) == []


def test_matching_synapse_ignores_permanence():
    """Test to make sure matching synapses method ignores the permanence entered."""
    fc1 = FakeCell(1)
    s1 = distal_synapse.DistalSynapse(fc1, 0)

    s = segment.Segment([s1])

    prev_active = {fc1}

    assert s.matching_synapses(prev_active) == [s1]


def test_synapse_source_input():
    """Test for setting source input."""
    s = synapse.Synapse(1, 2)

    assert s.source_input == 1


def test_synapse_permanence_input():
    """Test for setting permanence."""
    s = synapse.Synapse(3, 6)

    assert s.permanence == 6


"""Spatial Pooler Tests"""


def test_combine_input_field_flat():

    sp = spatial_pooler.SpatialPooler(
        input_space_size=10,
        column_count=16,
        initial_synapses_per_column=2,
    )

    i = np.zeros(10, dtype=int)  # flat array with len 10
    """this should detect it is a 1d array and not change it"""
    o = sp.combine_input_fields(i)

    """same shape and datatype as before"""
    assert o.shape == (10,)
    assert o.dtype == int


def test_combine_fields_list_of_arrays():
    """This demonstrates the method combine input fields properly combining the inputs."""
    sp = spatial_pooler.SpatialPooler(
        input_space_size=6,
        column_count=4,
        initial_synapses_per_column=2,
    )

    o = sp.combine_input_fields([[1, 0, 1], [0, 1, 0]])

    assert o.tolist() == [1, 0, 1, 0, 1, 0]
    assert sp.field_ranges == {}


def test_combine_input_fields_metadata():
    """
    This test demonstrates that when taking in labeled input fields we will
    properly have the field ranges and field order correctly generated from
    method combine input fields.
    """
    sp = spatial_pooler.SpatialPooler(
        input_space_size=6,
        column_count=4,
        initial_synapses_per_column=2,
    )
    """"using dictionaries"""
    o = sp.combine_input_fields({"a": [1, 0], "b": [0, 1, 0, 0]})

    assert o.tolist() == [1, 0, 0, 1, 0, 0]
    assert sp.field_ranges == {"a": (0, 2), "b": (2, 6)}
    assert sp.field_order == ["a", "b"]
    assert len(sp.column_field_map) == len(sp.columns)


def test_columns_to_binary():
    """
    Test to make sure columns to binary properly activates correct columns.
    We have 4 columns in sp and activate :2 which is the first two.
    """
    sp = spatial_pooler.SpatialPooler(
        input_space_size=10,
        column_count=4,
        initial_synapses_per_column=2,
    )

    c = sp.columns[:2]
    m = sp.columns_to_binary(c)

    assert m.tolist() == [1, 1, 0, 0]


def test_columns_from_raw_input_detects_connected_synapses():
    """Test to make sure a raw input has correct connected synapse."""
    sp = spatial_pooler.SpatialPooler(
        input_space_size=5,
        column_count=4,
        initial_synapses_per_column=2,
    )

    c0 = sp.columns[0]
    c0.connected_synapses = [synapse.Synapse(2, CONNECTED_PERM + 0.01)]

    combined = np.array([0, 0, 1, 0, 0])
    c = sp._columns_from_raw_input(combined)

    assert sp.columns[0] in c


def test_compute_active_columns_basic():
    """Test to compute active columns. We force high permanence to make sure all are active."""
    sp = spatial_pooler.SpatialPooler(
        input_space_size=8,
        column_count=16,
        initial_synapses_per_column=3,
    )
    """forcing high permanence to get all active."""
    for c in sp.columns:
        c.potential_synapses = [synapse.Synapse(i % 8, 1.0) for i in range(3)]
        c.connected_synapses = c.potential_synapses

    i = np.ones(8, dtype=int)
    m, c = sp.compute_active_columns(i, inhibition_radius=0)

    assert m.sum() == len(sp.columns)
    assert len(c) == len(sp.columns)


DESIRED_LOCAL_ACTIVITY = 10


def test_inhibition_reduces_columns():
    sp = spatial_pooler.SpatialPooler(
        input_space_size=8,
        column_count=16,
        initial_synapses_per_column=3,
    )
    """Columns with increasing overlap"""
    for i, c in enumerate(sp.columns):
        c.overlap = i

    active = sp._inhibition(sp.columns, inhibition_radius=999)

    """"""
    assert len(active) == DESIRED_LOCAL_ACTIVITY


PERMANENCE_INC = 0.01
PERMANENCE_DEC = 0.01


def test_learning_phase_permanences():
    sp = spatial_pooler.SpatialPooler(
        input_space_size=5,
        column_count=4,
        initial_synapses_per_column=3,
    )

    c0 = sp.columns[0]
    c0.potential_synapses = [
        synapse.Synapse(0, 0.5),
        synapse.Synapse(1, 0.5),
        synapse.Synapse(2, 0.5),
    ]

    i = np.array([1, 0, 1, 0, 0])

    sp.learning_phase([c0], i)

    assert c0.potential_synapses[0].permanence == 0.5 + PERMANENCE_INC
    assert c0.potential_synapses[2].permanence == 0.5 + PERMANENCE_INC

    assert c0.potential_synapses[1].permanence == 0.5 - PERMANENCE_DEC

    for s in c0.connected_synapses:
        assert s.permanence > CONNECTED_PERM


def test_average_receptive_field_size():
    sp = spatial_pooler.SpatialPooler(
        input_space_size=10, column_count=4, initial_synapses_per_column=2
    )

    sp.columns[0].connected_synapses = [synapse.Synapse(2, 1), synapse.Synapse(5, 1)]
    sp.columns[1].connected_synapses = [synapse.Synapse(1, 1), synapse.Synapse(9, 1)]
    sp.columns[2].connected_synapses = []
    sp.columns[3].connected_synapses = [synapse.Synapse(0, 1), synapse.Synapse(4, 1)]

    avg = sp.average_receptive_field_size()
    assert pytest.approx(avg) == (3 + 8 + 4) / 3

    """Temportal Memory Tests"""


def test_tm_step():
    """
    A column with no predicted cells should
    activate all its cells and choose one winner cell.
    Looks to be chosen by _compute_active_state.
    """
    cols = []
    """Three columns with 4 cells each"""
    for _ in range(3):
        c = column.Column([], position=(0, 0))
        c.cells = [cell.Cell() for _ in range(4)]
        cols.append(c)

    tm = temporal_memory.TemporalMemory(cols, cells_per_column=4)

    """col 1 becomes active which is 4 cells"""
    result = tm.step([cols[1]])
    """All cells are active"""
    assert result["active_cells"].sum() == 4
    """One cell is the learning cell"""
    assert result["learning_cells"].sum() == 1


def test_tm_steap_predicted_column():
    """
    If a column was correctly predicted,
    only the predicted cell should become active and be the winner.
    """
    cols = []
    """Two columnes with 3 cells each"""
    for _ in range(2):
        c = column.Column([], position=(0, 0))
        c.cells = [cell.Cell() for _ in range(3)]
        cols.append(c)

    tm = temporal_memory.TemporalMemory(cols, 3)

    """attach a segment to cel"""
    cel = cols[0].cells[1]
    seg = segment.Segment()
    syn = distal_synapse.DistalSynapse(cel, permanence=1)
    seg.synapses.append(syn)
    cel.segments.append(seg)

    """make the cel active and predictive"""
    tm.active_cells[-1] = {cel}
    tm.predictive_cells[-1] = {cel}

    """this col was predicted so there should be 1 active and 1 learning"""
    result = tm.step([cols[0]])

    assert result["active_cells"].sum() == 1
    assert result["learning_cells"].sum() == 1


def test_predictive_cells_threshold():
    """Cells become predictive when a segment has >= threshold active synapses."""
    col = column.Column([], position=(0, 0))
    """One column with 4 cells"""
    col.cells = [cell.Cell() for _ in range(4)]
    tm = temporal_memory.TemporalMemory([col], 4)

    """add segment to cell 2"""
    cel = col.cells[2]
    seg = segment.Segment()
    cel.segments.append(seg)

    tm.active_cells[0] = set()

    """set active cells from 0 to threshold-1"""
    req = SEGMENT_ACTIVATION_THRESHOLD
    for i in range(req):
        c = col.cells[i]
        tm.active_cells[0].add(c)
        """add synapses from seg to these cells"""
        seg.synapses.append(distal_synapse.DistalSynapse(c, permanence=1))

    """count active synapses, count>=threshold become predictive"""
    tm._compute_predictive_state()

    """cel should be a predictive cell"""
    assert cel in tm.predictive_cells[0]


def test_predictive_cells_below_threshold():
    """
    Cells do not become predictive if a segment does
    not reach activation threshold.
    """
    col = column.Column([], position=(0, 0))
    """One column with 4 cells"""
    col.cells = [cell.Cell() for _ in range(4)]
    tm = temporal_memory.TemporalMemory([col], 4)

    """add segment to cell 1"""
    cel = col.cells[1]
    seg = segment.Segment()
    cel.segments.append(seg)

    """active on cell 0"""
    tm.active_cells[0] = {col.cells[0]}

    seg.synapses.append(distal_synapse.DistalSynapse(col.cells[0], permanence=1))

    tm._compute_predictive_state()

    """cel should not be in predictive cells since the threshold is higher"""
    assert cel not in tm.predictive_cells[0]


def test_reinforce_segment_updates_permanence():
    """
    _reinforce_segment should increase active synapses,
    decrease inactive ones, and mark the segment as sequence.
    """
    col = column.Column([], position=(0, 0))
    """one column with 3 cells"""
    col.cells = [cell.Cell() for _ in range(3)]
    tm = temporal_memory.TemporalMemory([col], 3)

    """make cell 1 active"""
    seg = segment.Segment()
    col.cells[0].segments.append(seg)
    tm.active_cells[-1] = {col.cells[1]}

    """add two synapses, one is connected to active, one is to inactive"""
    syn_active = distal_synapse.DistalSynapse(col.cells[1], permanence=0.3)
    syn_inactive = distal_synapse.DistalSynapse(col.cells[1], permanence=0.3)
    seg.synapses = [syn_active, syn_inactive]

    tm.current_t = 0
    """the permanence should gain for acitve and decrease for inactive."""
    tm._reinforce_segment(seg)

    assert syn_active.permanence == pytest.approx(0.3 + PERMANENCE_INC)
    """possible issue here on the approx for inactive permanence"""
    assert syn_inactive.permanence == pytest.approx(0.32 - PERMANENCE_DEC)
    assert seg.sequence_segment is True


def test_reinforce_segment_grows_new_synapses():
    """
    _reinforce_segment should grow new synapses to previously
    active cells up to NEW_SYNAPSE_MAX.
    """
    col = column.Column([], position=(0, 0))
    """one column with 5 cells"""
    col.cells = [cell.Cell() for _ in range(5)]
    tm = temporal_memory.TemporalMemory([col], 5)

    seg = segment.Segment()
    col.cells[0].segments.append(seg)
    """active cells for 1, 2, 3, 4"""
    tm.active_cells[-1] = {col.cells[i] for i in range(1, 5)}

    tm.current_t = 0
    """this should see active cells not in synapses"""
    tm._reinforce_segment(seg)

    """we expect the synapses to be at least 4"""
    assert len(seg.synapses) == min(4, NEW_SYNAPSE_MAX)
    """the permanence should equal INITIAL_DISTAL_PERM for each new one"""
    for syn in seg.synapses:
        assert syn.permanence == INITIAL_DISTAL_PERM


def test_punish_segment():
    """_punish_segment should decrement all synapse permanences by PERMANENCE_DEC."""
    col = column.Column([], position=(0, 0))
    col.cells = [cell.Cell() for _ in range(2)]
    tm = temporal_memory.TemporalMemory([col], 2)

    """one segment with .5 permanence"""
    seg = segment.Segment()
    syn = distal_synapse.DistalSynapse(col.cells[1], permanence=0.5)
    seg.synapses.append(syn)

    tm._punish_segment(seg)
    """
    since this is punishing the segment, we should see its
    permanences lower by PERMANENCE_DEC
    """
    assert syn.permanence == pytest.approx(0.5 - PERMANENCE_DEC)


def test_best_matching_cell_prefers_untrained_cell():
    """
    If no segments match, _best_matching_cell
    should return an unused (segmentless) cell.
    """
    col = column.Column([], position=(0, 0))
    """one column with 3 cells"""
    col.cells = [cell.Cell() for _ in range(3)]
    tm = temporal_memory.TemporalMemory([col], 3)
    """no segments means no best segment"""
    best_cell, best_seg = tm._best_matching_cell(col, prev_t=0)
    """we should see the best column cells but no segments"""
    assert best_cell in col.cells
    assert best_seg is None


def test_best_matching_cell_finds_best_segment():
    """
    _best_matching_cell should choose the cell
    whose segment has the most matching synapses.
    """
    col = column.Column([], position=(0, 0))
    """one column with 3 cells"""
    col.cells = [cell.Cell() for _ in range(3)]
    tm = temporal_memory.TemporalMemory([col], 3)

    """active cells 0 and 1"""
    tm.active_cells[-1] = {col.cells[0], col.cells[1]}

    cell1 = col.cells[0]
    cell2 = col.cells[1]

    """give cell1 a segment"""
    seg1 = segment.Segment()
    seg1.synapses.append(distal_synapse.DistalSynapse(col.cells[0], 1))
    cell1.segments.append(seg1)

    """give cell2 two segments"""
    seg2 = segment.Segment()
    seg2.synapses.append(distal_synapse.DistalSynapse(col.cells[0], 1))
    seg2.synapses.append(distal_synapse.DistalSynapse(col.cells[1], 1))
    cell2.segments.append(seg2)

    best_cell, best_seg = tm._best_matching_cell(col, prev_t=-1)

    """
    Cell 2 and Seg 2 should be the best since they have the most
    matching synapses
    """
    assert best_cell is cell2
    assert best_seg is seg2


def test_cells_to_binary():
    """
    cells_to_binary should output a flat
    binary vector marking active cells by their global index.
    """
    col1 = column.Column([], position=(0, 0))
    col2 = column.Column([], position=(0, 1))
    col1.cells = [cell.Cell() for _ in range(3)]
    col2.cells = [cell.Cell() for _ in range(3)]

    tm = temporal_memory.TemporalMemory([col1, col2], 3)
    tm.active_cells[0] = {col1.cells[1], col2.cells[0]}
    v = tm.cells_to_binary(tm.active_cells[0])
    assert v.tolist() == [0, 1, 0, 1, 0, 0]


def test_predictive_columns_mask():
    """
    get_predictive_columns_mask should mark columns
    that contain any predictive cell.
    """
    """three columns with two cells each"""
    col1 = column.Column([], position=(0, 0))
    col2 = column.Column([], position=(1, 0))
    col3 = column.Column([], position=(2, 0))
    col1.cells = [cell.Cell() for _ in range(2)]
    col2.cells = [cell.Cell() for _ in range(2)]
    col3.cells = [cell.Cell() for _ in range(2)]

    tm = temporal_memory.TemporalMemory([col1, col2, col3], 2)
    """make one predictive cell in middle column"""
    tm.predictive_cells[0] = {col2.cells[0]}
    """get the mask"""
    m = tm.get_predictive_columns_mask(t=0)
    """the mask should be the middle column with 1 active bit"""
    assert m.tolist() == [0, 1, 0]


def test_reset_state():
    """
    reset_state should clear all per-timestep
    activity while keeping learned synapses intact.
    """
    col = column.Column([], position=(0, 0))
    col.cells = [cell.Cell() for _ in range(3)]

    tm = temporal_memory.TemporalMemory([col], 3)
    tm.active_cells[0] = {col.cells[0]}
    tm.predictive_cells[0] = {col.cells[1]}

    tm.reset_state()

    assert tm.active_cells == {}
    assert tm.predictive_cells == {}
    assert tm.current_t == 0
