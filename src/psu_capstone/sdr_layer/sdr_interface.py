"""Interface for SDR layer components."""

from typing import Any, Iterable, Protocol, runtime_checkable

from psu_capstone.sdr_layer.sdr import SDR


@runtime_checkable
class SDRInterface(Protocol):
    """Defines the interface for SDR layer components."""

    def zero(self) -> None:
        """Sets all bits in the SDR to 0."""
        ...

    def set_sparse(self, sparse: Iterable[int]) -> None:
        """Sets the sparse representation of the SDR.

        Args:
            sparse (Any): The sparse representation to set.
        """
        ...

    def get_sparse(self) -> Any:
        """Gets the sparse representation of the SDR.

        Returns:
            Any: The sparse representation of the SDR.
        """
        ...

    def set_dense(self, dense: Iterable[int]) -> None:
        """Sets the dense representation of the SDR.

        Args:
            dense (Iterable[int]): The dense representation to set.
        """
        ...

    def get_dense(self) -> Any:
        """Gets the dense representation of the SDR.

        Returns:
            Any: The dense representation of the SDR.
        """
        ...

    def set_sparsity(self, sparsity: float) -> None:
        """Sets the sparsity of the SDR.

        Args:
            sparsity (float): The sparsity value to set.
        """
        ...

    def get_sparsity(self) -> float:
        """Gets the sparsity of the SDR.

        Returns:
            float: The sparsity of the SDR.
        """
        ...

    def set_sdr(self, other: "SDR") -> None:
        """Sets the SDR from another SDR.

        Args:
            other (SDR): The other SDR to copy from.
        """
        ...

    def get_sdr(self) -> "SDR":
        """Gets the SDR.

        Returns:
            SDR: The SDR instance.
        """
        ...

    def sdr_to_type(self, type: Any, is_sparse: bool) -> Any:
        """Converts the SDR to a specific type.

        Returns:
            Any: The converted SDR type.
        """
        ...
