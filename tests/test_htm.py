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
    fc1 = FakeCell(1)
    fc2 = FakeCell(2)
    s1 = distal_synapse.DistalSynapse(fc1, 0.5)
    s2 = distal_synapse.DistalSynapse(fc2, 0.1)

    s = segment.Segment([s1, s2])

    prev_active = {FakeCell(155)}
    assert s.matching_synapses(prev_active) == []


def test_matching_synapse_empty_on_empty_seg():
    s = segment.Segment()
    assert s.matching_synapses({FakeCell(1)}) == []


def test_matching_synapse_ignores_permanence():
    fc1 = FakeCell(1)
    s1 = distal_synapse.DistalSynapse(fc1, 0)

    s = segment.Segment([s1])

    prev_active = {fc1}

    assert s.matching_synapses(prev_active) == [s1]


def test_synapse_source_input():
    s = synapse.Synapse(1, 2)

    assert s.source_input == 1


def test_synapse_permanence_input():
    s = synapse.Synapse(3, 6)

    assert s.permanence == 6
