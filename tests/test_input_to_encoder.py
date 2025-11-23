import pandas as pd
import pytest

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR
from psu_capstone.input_layer.input_handler import InputHandler


class DummyEncoder(BaseEncoder):
    """Temporary placeholder encoder for tests."""

    def __init__(self, dimensions=None):
        # For tests that only care about DataFrame wiring, we don't really
        # need a meaningful SDR shape, so default to [1] if not given.
        if dimensions is None:
            dimensions = [1]

        super().__init__(dimensions)
        self.input_df: pd.DataFrame | None = None

    def attach_input(self, df: pd.DataFrame):
        """Store a reference to the incoming DataFrame (no copy)."""
        self.input_df = df

    def encode(self, input_value: float, output_sdr: SDR) -> bool:
        """
        Dummy implementation to satisfy the abstract method requirement.

        Real encoders would write into output_sdr based on input_value.
        For this stub, we don't modify output_sdr and just return True.
        """
        return True


def test_input_to_encoder_passes_same_dataframe_object():
    """
    This test verifies that InputHandler.to_dataframe(...) returns a pandas DataFrame
    and that passing that DataFrame into an encoder keeps the *same object*.

    The purpose is to ensure we are not copying, re-wrapping, or rebuilding
    the DataFrame — the encoder should receive the identical object produced
    by the handler.
    """

    # Arrange
    data_list = [0, 1, 1, 2, 3, 5, 8, 13]

    # Create the InputHandler and convert raw data → DataFrame
    handler = InputHandler()
    df = handler._to_dataframe(data_list)

    # handler returns a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Act
    # Attach the returned DataFrame to a dummy encoder
    encoder = DummyEncoder()
    encoder.attach_input(df)

    # Assert
    # Confirm encoder.input_df is *the same object*, not a copy
    assert encoder.input_df is df  # identity (same memory reference)
    pd.testing.assert_frame_equal(encoder.input_df, df)
