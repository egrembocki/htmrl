import numpy as np
import pytest

from psu_capstone.agent_layer.HTM import InputField
from psu_capstone.encoder_layer.category_encoder import CategoryEncoder, CategoryParameters
from psu_capstone.encoder_layer.date_encoder import DateEncoder, DateEncoderParameters
from psu_capstone.encoder_layer.fourier_encoder import FourierEncoder, FourierEncoderParameters
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters


def test_input_field_correct_encoder_created():
    """
    This class uses dynamic instantiation. The test is designed
    to make sure the correct instance is created with the parameter
    types.
    """
    parameters = RDSEParameters()
    in_fi = InputField(parameters)
    assert isinstance(in_fi.encoder, RandomDistributedScalarEncoder)

    parameters = ScalarEncoderParameters()
    in_fi = InputField(parameters)
    assert isinstance(in_fi.encoder, ScalarEncoder)

    parameters = DateEncoderParameters()
    in_fi = InputField(parameters)
    assert isinstance(in_fi.encoder, DateEncoder)

    parameters = CategoryParameters()
    in_fi = InputField(parameters)
    assert isinstance(in_fi.encoder, CategoryEncoder)

    parameters = FourierEncoderParameters()
    in_fi = InputField(parameters)
    assert isinstance(in_fi.encoder, FourierEncoder)

    # when geospatial is added, add it here.


def test_input_field_check_cell_count():
    """Test to check that the number of cells is equal to the size in the encoder."""
    parameters = RDSEParameters()
    in_fi = InputField(parameters)
    assert len(in_fi.cells) == in_fi.encoder.size


def test_input_field_with_no_parameters():
    """Test to make sure the input field defaults to rdse if no parameters are given."""
    in_fi = InputField()
    assert isinstance(in_fi.encoder, RandomDistributedScalarEncoder)


def test_input_field_check_cells_are_active_after_encode():
    """Test that we have active cells after encoding."""
    in_fi = InputField()
    a = in_fi.encode(10)
    cells = in_fi.cells
    sparse = np.nonzero(a)[0]
    active_cells = [i for i, cell in enumerate(cells) if cell.active]
    assert np.array_equal(sparse, active_cells)


def test_input_field_advance_cell_states():
    """This test makes sure cell states are being adjusted properly."""
    in_fi = InputField()
    in_fi.encode(10)
    cells = in_fi.cells
    """Find the current active, winner, and predictive cells."""
    active_cells = [i for i, cell in enumerate(cells) if cell.active]
    winner_cells = [i for i, cell in enumerate(cells) if cell.winner]
    predictive_cells = [i for i, cell in enumerate(cells) if cell.predictive]
    assert active_cells != 0
    in_fi.advance_states()
    """With advancing states, these should be empty for a new encoding to come in."""
    new_active_cells = [i for i, cell in enumerate(cells) if cell.active]
    assert new_active_cells == []
    """All previous cell designations should equal what it had been."""
    prev_active = [i for i, cell in enumerate(cells) if cell.prev_active]
    assert prev_active == active_cells
    prev_winner = [i for i, cell in enumerate(cells) if cell.prev_winner]
    assert prev_winner == winner_cells
    prev_predictive = [i for i, cell in enumerate(cells) if cell.prev_predictive]
    assert prev_predictive == predictive_cells
