"""Interface for SDR layer components."""

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from src.psu_capstone.sdr_layer.sdr import SDR

T = TypeVar("T")


@runtime_checkable
class SDRInterface(Protocol, Generic[T]):
    """Defines the interface for SDR layer components."""

    def set_sdr(self, sdr: SDR) -> None:
        """Sets the SDR data.

        Args:
            sdr (SDR): The SDR data to set.
        """
        ...

    def set_sparse(self, sparse: Any) -> None:
        """Sets the sparse representation of the SDR.

        Args:
            sparse (Any): The sparse representation to set.
        """
        ...

    def set_dense(self, dense: Any) -> None:
        """Sets the dense representation of the SDR.

        Args:
            dense (Any): The dense representation to set.
        """
        ...

    def set_sparsity(self, sparsity: float) -> None:
        """Sets the sparsity of the SDR.

        Args:
            sparsity (float): The sparsity value to set.
        """
        ...

    def get_sparse(self) -> Any:
        """Gets the sparse representation of the SDR.

        Returns:
            Any: The sparse representation of the SDR.
        """
        ...

    def get_dense(self) -> Any:
        """Gets the dense representation of the SDR.

        Returns:
            Any: The dense representation of the SDR.
        """
        ...

    def get_sdr(self) -> SDR:
        """Gets the SDR object.

        Returns:
            SDR: The SDR object.
        """
        ...

    def get_sparsity(self) -> float:
        """Gets the sparsity of the SDR.

        Returns:
            float: The sparsity of the SDR.
        """
        ...

    def sdr_to_type(self) -> T:
        """Converts the SDR to a specific type.

        Returns:
            T: The converted SDR type.
        """
        ...
