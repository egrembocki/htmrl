"""Sparse Distributed Representation (SDR) utilities.

This module mirrors NuPIC's C++ SDR behaviour using idiomatic Python.  It
exposes the same public surface while providing type aliases, validation
helpers, and callback hooks that keep dense, sparse, and coordinate caches
in sync with each other. -- Renamed file
"""

from __future__ import annotations

import random
from math import prod
from typing import Any, Callable, Iterable, List, Optional, Sequence

# Type aliases mirroring the C++ implementation

elem_dense = int  #: Dense element type used for storing SDR bits.
elem_sparse = int  #: Sparse index type mirroring the C++ implementation.
sdr_dense_t = List[elem_dense]  #: Alias for the dense SDR container type.
sdr_sparse_t = List[elem_sparse]  #: Alias for the sparse SDR container type.
sdr_coordinate_t = List[List[int]]  #: Alias representing coordinates grouped per dimension.
sdr_callback_t = Callable[[], None]  #: Callback signature invoked on SDR state changes.


INPUT_SDR_NONE_MSG = "Input SDR cannot be None."  #: Common error message for null SDR inputs.


class SDR:
    """Python counterpart of NuPIC's SparseDistributedRepresentation.

    The instance maintains dense, sparse, and coordinate views that lazily
    materialise from whichever representation is currently authoritative.
    Callbacks can be registered to observe mutations or destruction events.

    Attributes:
        __dimensions: Shape of the SDR as a list of ints.
        __size: Total number of bits in the SDR.
        _dense: Backing dense bit vector representing active elements.
        _sparse: Cached list of active indices in sparse form.
        _coordinates: Cached coordinates for each active bit broken per dimension.
        _dense_valid: Flag indicating whether the dense buffer is authoritative.
        _sparse_valid: Flag indicating whether the sparse buffer is authoritative.
        _coordinates_valid: Flag indicating whether the coordinate cache is valid.
        __callbacks: Registered change callbacks invoked after value updates.
        __destroy_callbacks: Callbacks invoked during ``destroy``.
    """

    def __init__(self, dimensions: List[int]) -> None:
        """Create a new SDR with the given dimensions.

        Args:
            dimensions: Iterable defining the length of each SDR dimension.

        Raises:
            AssertionError: If no dimensions are provided.
        """
        try:
            if dimensions is None or len(dimensions) == 0:
                raise ValueError("SDR must have at least one dimension.")
            else:
                self._dimensions: List[int] = [int(dim) for dim in dimensions]

        except Exception as e:
            raise ValueError("Invalid dimensions provided for SDR.") from e

        self._size: int = prod(int(dim) for dim in self._dimensions)

        self._dense: sdr_dense_t = [elem_dense(0)] * int(self._size)
        self._sparse: sdr_sparse_t = []
        self._coordinates: sdr_coordinate_t = [[] for _ in self._dimensions]
        self._sparsity: float = 0.0
        self._interface: Any = None

        self._dense_valid = True
        self._sparse_valid = False
        self._coordinates_valid = False

        self.__callbacks: List[Optional[sdr_callback_t]] = []
        self.__destroy_callbacks: List[Optional[sdr_callback_t]] = []

    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        """Return the total number of bits in the SDR."""
        return self._size

    @size.setter
    def size(self, size: int) -> None:
        """Set the total number of bits in the SDR.

        Args:
            size: New size for the SDR.

        Raises:
            AssertionError: If the new size is not positive.
        """
        new_size = int(size)
        assert new_size > 0, "SDR size must be positive."

        self._size = new_size
        self._dense = [elem_dense(0)] * int(self._size)
        self._sparse = []
        self._coordinates = [[] for _ in self._dimensions]

        self.clear()
        self._dense_valid = True
        self.do_callbacks()

    @property
    def dimensions(self) -> List[int]:
        """Return the dimensions of the SDR."""
        return self._dimensions

    @property
    def interface(self) -> Any:
        """Get the SDR's interface for environment interaction.

        Returns:
            The current interface object.
        """
        return self._interface

    @interface.setter
    def interface(self, interface: Any) -> None:
        """Set the SDR's interface for environment interaction.

        Args:
            interface: An object that defines how the SDR interacts with the environment.
        """
        self._interface = interface

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Invalidate cached representations without mutating underlying buffers."""

        self._dense_valid = False
        self._sparse_valid = False
        self._coordinates_valid = False

    def do_callbacks(self) -> None:
        """Notify registered watchers that the SDR value has changed."""

        for callback in self.__callbacks:
            if callback is not None:
                callback()

    # ------------------------------------------------------------------
    # Setters that operate on in-place modifications
    # ------------------------------------------------------------------
    def set_dense_inplace(self) -> None:
        """Mark the dense buffer as authoritative after in-place edits.

        Verifies the dense array size, coerces values into ``elem_dense``, and
        invalidates cached sparse/coordinate views before notifying callbacks.
        """

        assert len(self._dense) == int(self._size), "Dense buffer size does not match SDR size."

        self._dense = [elem_dense(int(val)) for val in self._dense]

        self.clear()
        self._dense_valid = True
        self.do_callbacks()

    def set_sparse_inplace(self) -> None:
        """Mark the sparse buffer as authoritative after in-place edits.

        Ensures sparse indices are sorted, unique, and within bounds, then
        refreshes cached dense/coordinate views and triggers callbacks.
        """

        assert all(
            int(self._sparse[i]) <= int(self._sparse[i + 1]) for i in range(len(self._sparse) - 1)
        ), "Sparse data must be sorted!"
        if self._sparse:
            assert int(self._sparse[-1]) < int(self._size), "Sparse index out of bounds!"

        previous = None
        for idx in self._sparse:
            assert previous is None or int(idx) != int(
                previous
            ), "Sparse data must not contain duplicates!"
            previous = idx

        self._sparse = [elem_sparse(int(val)) for val in self._sparse]

        self.clear()
        self._sparse_valid = True
        self.do_callbacks()

    def set_coordinates_inplace(self) -> None:
        """Mark the coordinate buffers as authoritative after edits.

        Validates that every dimension shares the same active count, that
        indices respect per-dimension bounds, and that caches reflect the new
        canonical coordinate ordering.
        """

        assert len(self._coordinates) == len(
            self._dimensions
        ), "Coordinate data must match SDR dimensionality!"

        expected_length = len(self._coordinates[0]) if self._coordinates else 0
        for dim_index, coord_vec in enumerate(self._coordinates):
            assert (
                len(coord_vec) == expected_length
            ), "All coordinate vectors must share the same length!"
            limit = int(self._dimensions[dim_index])
            for idx in coord_vec:
                assert int(idx) < limit, "Coordinate index out of bounds!"

        self._coordinates = [
            [elem_sparse(int(idx)) for idx in coord_vec] for coord_vec in self._coordinates
        ]

        self.clear()
        self._coordinates_valid = True
        self.do_callbacks()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Reset the SDR to an empty state and fire destroy callbacks."""

        self.clear()
        self._dense.clear()
        self._sparse.clear()
        self._coordinates.clear()
        self._size = 0
        self._dimensions.clear()

        for callback in self.__destroy_callbacks:
            if callback is not None:
                callback()

        self.__callbacks.clear()
        self.__destroy_callbacks.clear()

    # ------------------------------------------------------------------
    # Dimension helpers
    # ------------------------------------------------------------------
    def get_dimensions(self) -> List[int]:
        """Return a copy of the SDR dimensionality."""
        return list(self._dimensions)

    def reshape(self, new_dimensions: Sequence[int]) -> None:
        """Reshape the SDR while preserving the total number of bits.

        Lazily ensures a sparse view exists, verifies that the flattened size
        stays constant, then updates dimensional metadata while invalidating
        coordinate caches.
        """

        if not (self._dense_valid or self._sparse_valid):
            self.get_sparse()

        new_dims = [int(dim) for dim in new_dimensions]
        new_size = prod(int(dim) for dim in new_dims)
        assert new_size == int(self._size), "Total size must remain constant when reshaping SDR."

        self._dimensions = new_dims
        self._coordinates_valid = False
        self._coordinates = [[] for _ in self._dimensions]

    # ------------------------------------------------------------------
    # Protocols: Setters and Getters
    # ------------------------------------------------------------------

    def zero(self) -> None:
        """Clear all active bits, reset caches, and mark the dense buffer canonical."""
        self._dense = [elem_dense(0)] * int(self._size)
        self._sparse = []
        self._coordinates = [[] for _ in self._dimensions]

        self.clear()
        self._dense_valid = True
        self.do_callbacks()

    def set_dense(self, dense: Iterable[int]) -> None:
        """Replace contents with a dense iterable after validating its length."""
        dense_list = list(dense)
        assert len(dense_list) == int(self._size), "Input dense array size does not match SDR size."

        temp = [elem_dense(int(val)) for val in dense_list]
        self._dense, temp = temp, self._dense
        self.set_dense_inplace()

    def get_dense(self) -> sdr_dense_t:  # alias: -> List[int]
        """Return the dense representation, materialising it from sparse data if required."""
        if not self._dense_valid:
            if not self._sparse_valid:
                self.get_sparse()
            self._dense = [elem_dense(0)] * int(self._size)
            for index in self._sparse:
                self._dense[int(index)] = elem_dense(1)
            self._dense_valid = True
        return self._dense

    def get_sparsity(self) -> float:
        """Return the fraction of active bits relative to the configured size."""
        return len(self.get_sparse()) / float(int(self._size))

    def set_sparsity(self, sparsity: float) -> None:
        """Sets the sparsity of the SDR.

        Args:
            sparsity (float): The sparsity value to set.
        """
        self._sparsity = (
            sparsity  # this will need to get changed change the active bits accordingly
        )

    def set_sparse(self, sparse: Iterable[int]) -> None:
        """Replace the SDR contents with sparse indices and recompute caches."""
        self._sparse = [elem_sparse(int(idx)) for idx in sparse]
        self.set_sparse_inplace()

    def get_sparse(self) -> sdr_sparse_t:
        """Return sparse indices, creating them from dense or coordinate caches as needed."""
        if not self._sparse_valid:
            if self._dense_valid:
                self._sparse = [
                    elem_sparse(idx) for idx, value in enumerate(self._dense) if int(value) != 0
                ]
            elif self._coordinates_valid:
                self._sparse = []
                length = len(self._coordinates[0]) if self._coordinates else 0
                for idx in range(length):
                    flat_index = 0
                    stride = 1
                    for dim_idx in range(len(self._dimensions) - 1, -1, -1):
                        coord = self._coordinates[dim_idx][idx]
                        flat_index += int(coord) * stride
                        stride *= int(self._dimensions[dim_idx])
                    self._sparse.append(elem_sparse(flat_index))
            else:
                self._sparse = []
            self._sparse.sort(key=int)
            self._sparse_valid = True
        return self._sparse

    def sdr_to_type(self, type: Any, is_sparse: bool) -> Any:
        """Convert the SDR to a specific type, either dense or sparse."""
        if isinstance(type, list):
            if is_sparse:
                return [elem_sparse(int(idx)) for idx in self.get_sparse()]
            else:
                return [elem_dense(int(val)) for val in self.get_dense()]
        else:
            raise NotImplementedError("Conversion to the specified type is not implemented.")

    def set_sdr(self, other: "SDR") -> None:
        """Copy shape and active bits from another SDR, reshaping if necessary."""
        other_dims = other.get_dimensions()
        if not self._dimensions:
            self._dimensions = [int(dim) for dim in other_dims]
            self._size = prod(int(dim) for dim in self._dimensions)
        else:
            self.reshape(other_dims)
        self.set_sparse(int(idx) for idx in other.get_sparse())

    def get_sdr(self) -> "SDR":
        """Return a copy of this SDR instance."""
        copy = SDR(self._dimensions)
        copy.set_sparse(int(idx) for idx in self.get_sparse())
        return copy

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def at_byte(self, coordinates: Sequence[int]) -> int:
        """Return the value stored at the provided multidimensional coordinate.

        Performs bounds checking, computes the flattened index, and returns
        the stored dense byte without mutating caches.
        """
        assert len(coordinates) == len(
            self._dimensions
        ), "Number of coordinates must match dimensionality."

        flat_index = 0
        stride = 1
        for dim_size, coord in zip(reversed(self._dimensions), reversed(coordinates)):
            assert int(coord) < int(dim_size), "Coordinate out of bounds."
            flat_index += int(coord) * stride
            stride *= int(dim_size)
        return self.get_dense()[flat_index]

    def set_coordinates(self, coordinates: Iterable[Iterable[int]]) -> None:
        """Replace the SDR contents with explicit coordinates per dimension."""
        self._coordinates = [
            [elem_sparse(int(idx)) for idx in coord_vec] for coord_vec in coordinates
        ]
        self.set_coordinates_inplace()

    def get_coordinates(self) -> sdr_coordinate_t:
        """Return coordinate lists, deriving them from the sparse view when stale."""
        if not self._coordinates_valid:
            for coord in self._coordinates:
                coord.clear()

            for index in self.get_sparse():
                flat_index = int(index)
                for dim in range(len(self._dimensions) - 1, 0, -1):
                    dim_size = int(self._dimensions[dim])
                    self._coordinates[dim].append(flat_index % dim_size)
                    flat_index //= dim_size
                self._coordinates[0].append(flat_index)

            self._coordinates_valid = True
        return self._coordinates

    def get_sum(self) -> int:
        """Return the number of active bits, delegating to the sparse representation."""
        return len(self.get_sparse())

    def get_overlap(self, other: "SDR") -> int:
        """Compute the overlap between this SDR and another with matching dimensions.

        Args:
            other: SDR to compare against.

        Returns:
            Number of shared active indices.

        Raises:
            AssertionError: If the SDRs do not share the same dimensions.
        """
        assert (
            self._dimensions == other.get_dimensions()
        ), "SDRs must have matching dimensions to compute overlap."

        self_sparse = set(map(int, self.get_sparse()))
        other_sparse = set(map(int, other.get_sparse()))
        return len(self_sparse & other_sparse)

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------
    def intersection(self, sdrs: List["SDR"]) -> None:
        """Compute the bitwise intersection of multiple SDRs and store it in-place.

        Args:
            sdrs: Collection of SDRs to intersect with this instance.

        Raises:
            AssertionError: If fewer than two SDRs are provided or if any SDR
            has incompatible dimensions.
        """
        assert len(sdrs) >= 2, "Intersection requires at least two SDRs."

        inputs = list(sdrs)
        inplace = False
        i = 0
        while i < len(inputs):
            sdr = inputs[i]
            assert sdr is not None, INPUT_SDR_NONE_MSG
            assert (
                sdr.get_dimensions() == self._dimensions
            ), "All SDRs must share dimensions for intersection."
            if sdr is self:
                inplace = True
                inputs[i] = inputs[-1]
                inputs.pop()
                continue
            i += 1

        if inplace:
            dense_buffer = self.get_dense()
        else:
            dense_buffer = [elem_dense(int(val)) for val in inputs[-1].get_dense()]
            self._dense = dense_buffer
            inputs.pop()

        for sdr in inputs:
            data = sdr.get_dense()
            for idx, val in enumerate(data):
                dense_buffer[idx] = elem_dense(1 if int(dense_buffer[idx]) and int(val) else 0)

        self.set_dense_inplace()

    def _validate_concatenate_inputs(self, inputs: List["SDR"], axis_index: int) -> int:
        """Validate concatenate inputs and return the combined size along the chosen axis."""
        concat_axis_size = 0
        for sdr in inputs:
            assert sdr is not None, "Input SDR cannot be None."
            dims = sdr.get_dimensions()
            assert len(dims) == len(self._dimensions), "Input dimensionality mismatch."
            for dim_idx, (dim_in, dim_self) in enumerate(zip(dims, self._dimensions)):
                if dim_idx == axis_index:
                    concat_axis_size += int(dim_in)
                else:
                    assert int(dim_in) == int(
                        dim_self
                    ), "All non-axis dimensions must match for concatenate."
        return concat_axis_size

    def set_union(self, sdrs: List["SDR"]) -> None:
        """Compute the bitwise union of multiple SDRs and store it in-place.

        Args:
            sdrs: Collection of SDRs to union with this instance.

        Raises:
            AssertionError: If fewer than two SDRs are provided or if any SDR
            has incompatible dimensions.
        """
        assert len(sdrs) >= 2, "Union requires at least two SDRs."

        inputs = list(sdrs)
        inplace = False
        i = 0
        while i < len(inputs):
            sdr = inputs[i]
            assert sdr is not None, INPUT_SDR_NONE_MSG
            assert (
                sdr.get_dimensions() == self._dimensions
            ), "All SDRs must share dimensions for union."
            if sdr is self:
                inplace = True
                inputs[i] = inputs[-1]
                inputs.pop()
                continue
            i += 1

        if inplace:
            dense_buffer = self.get_dense()
        else:
            dense_buffer = [elem_dense(int(val)) for val in inputs[-1].get_dense()]
            self._dense = dense_buffer
            inputs.pop()

        for sdr in inputs:
            data = sdr.get_dense()
            for idx, val in enumerate(data):
                dense_buffer[idx] = elem_dense(1 if int(dense_buffer[idx]) or int(val) else 0)

        self.set_dense_inplace()

    def concatenate(self, inputs: List["SDR"], axis: int = 0) -> None:
        """Concatenate SDRs along a chosen axis, writing the dense result into this instance.

        Args:
            inputs: SDRs to concatenate into this SDR.
            axis: Axis index along which to concatenate.

        Raises:
            AssertionError: If fewer than two inputs are provided, if the axis
            is invalid, or if input dimensions are incompatible with ``self``.
        """
        assert len(inputs) >= 2, "Not enough inputs to concatenate."

        axis_index = int(axis)
        assert 0 <= axis_index < len(self._dimensions), "Axis out of bounds."

        concat_axis_size = self._validate_concatenate_inputs(inputs, axis_index)

        assert concat_axis_size == int(
            self._dimensions[axis_index]
        ), "Concatenation axis dimensions do not sum to output size."

        buffers = [list(sdr.get_dense()) for sdr in inputs]
        row_lengths: List[int] = []
        for sdr in inputs:
            row = 1
            dims = sdr.get_dimensions()
            for dim_idx in range(axis_index, len(dims)):
                row *= int(dims[dim_idx])
            row_lengths.append(row)

        total_size = int(self._size)
        self._dense = [elem_dense(0)] * total_size
        positions = [0] * len(inputs)
        out_pos = 0

        while out_pos < total_size:
            for buf_idx, buf in enumerate(buffers):
                row = row_lengths[buf_idx]
                start = positions[buf_idx]
                end = start + row
                for value in buf[start:end]:
                    self._dense[out_pos] = elem_dense(int(value))
                    out_pos += 1
                positions[buf_idx] = end

        self.set_dense_inplace()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def add_on_change_callback(self, callback: sdr_callback_t) -> int:
        """Register a mutation callback and return a reusable handle.

        Args:
            callback: Callable that takes no arguments.

        Returns:
            Handle that can be used to remove the callback later.
        """
        for index, existing in enumerate(self.__callbacks):
            if existing is None:
                self.__callbacks[index] = callback
                return index
        self.__callbacks.append(callback)
        return len(self.__callbacks) - 1

    def remove_on_change_callback(self, index: int) -> None:
        """Remove a previously registered change callback via its handle.

        Args:
            index: Handle returned by :meth:`add_on_change_callback`.

        Raises:
            AssertionError: If the handle is invalid or already removed.
        """
        idx = int(index)
        assert 0 <= idx < len(self.__callbacks), (
            "SparseDistributedRepresentation::removeCallback, Invalid Handle!",
        )
        assert self.__callbacks[idx] is not None, (
            "SparseDistributedRepresentation::removeCallback, Callback already removed!",
        )
        self.__callbacks[idx] = None

    def add_destroy_callback(self, callback: sdr_callback_t) -> int:
        """Register a destroy-time callback and return a reusable handle.

        Args:
            callback: Callable that takes no arguments.

        Returns:
            Handle that can be used to remove the callback later.
        """
        for index, existing in enumerate(self.__destroy_callbacks):
            if existing is None:
                self.__destroy_callbacks[index] = callback
                return index
        self.__destroy_callbacks.append(callback)
        return len(self.__destroy_callbacks) - 1

    def remove_destroy_callback(self, index: int) -> None:
        """Remove a destroy callback associated with the provided handle.

        Args:
            index: Handle returned by :meth:`add_destroy_callback`.

        Raises:
            AssertionError: If the handle is invalid or already removed.
        """
        idx = int(index)
        assert 0 <= idx < len(self.__destroy_callbacks), (
            "SparseDistributedRepresentation::removeDestroyCallback, Invalid Handle!",
        )
        assert self.__destroy_callbacks[idx] is not None, (
            "SparseDistributedRepresentation::removeDestroyCallback, " "Callback already removed!",
        )
        self.__destroy_callbacks[idx] = None

    # ------------------------------------------------------------------
    # Randomised operations
    # ------------------------------------------------------------------
    def randomize(self, sparsity: float, rng: Optional[random.Random] = None) -> None:
        """Populate the SDR with random active bits drawn at the requested sparsity."""
        assert 0.0 <= sparsity <= 1.0, "Sparsity must be within [0, 1]."

        size = int(self._size)
        nbits = max(0, min(size, int(round(size * float(sparsity)))))
        rng = rng or random.Random(0)

        if nbits:
            selected = sorted(rng.sample(range(size), nbits))
        else:
            selected = []

        self._sparse = [elem_sparse(idx) for idx in selected]
        self.set_sparse_inplace()

    def add_noise(self, fraction_noise: float, rng: Optional[random.Random] = None) -> None:
        """Stochastically move active bits while preserving the overall sparsity."""
        assert 0.0 <= fraction_noise <= 1.0, "Noise fraction must be within [0, 1]."
        assert (
            1.0 + fraction_noise
        ) * self.get_sparsity() <= 1.0, "Noise would exceed SDR capacity."

        num_move_bits = int(round(fraction_noise * int(self.get_sum())))
        if num_move_bits == 0:
            return

        rng = rng or random.Random()

        sparse_values = list(map(int, self.get_sparse()))
        assert len(sparse_values) >= num_move_bits, "Not enough active bits to turn off."
        turn_off = rng.sample(sparse_values, num_move_bits)

        dense = self.get_dense()
        off_population = [idx for idx in range(int(self._size)) if int(dense[idx]) == 0]
        assert len(off_population) >= num_move_bits, "Not enough inactive bits to turn on."
        turn_on = rng.sample(off_population, num_move_bits)

        for idx in turn_on:
            dense[idx] = elem_dense(1)
        for idx in turn_off:
            dense[idx] = elem_dense(0)

        self.set_dense_inplace()

    def kill_cells(self, fraction: float, seed: int = 0) -> None:
        """Deactivate a random subset of bits, seeded for deterministic selection."""
        assert 0.0 <= fraction <= 1.0, "Kill fraction must be within [0, 1]."
        size = int(self._size)
        nkill = int(round(size * fraction))
        if nkill == 0:
            return

        rng = random.Random(int(seed))
        dense = self.get_dense()
        indices = list(range(size))
        to_kill = rng.sample(indices, nkill)

        for idx in to_kill:
            dense[idx] = elem_dense(0)

        self.set_dense(int(val) for val in dense)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        """Return ``True`` when both SDRs share shape and identical dense content."""
        if not isinstance(other, SDR):
            return NotImplemented

        if len(self._dimensions) != len(other._dimensions):
            return False
        for left, right in zip(self._dimensions, other._dimensions):
            if int(left) != int(right):
                return False

        dense_self = self.get_dense()
        dense_other = other.get_dense()
        if len(dense_self) != len(dense_other):
            return False

        return all(int(a) == int(b) for a, b in zip(dense_self, dense_other))

    def __repr__(self) -> str:
        """Return a concise, developer-friendly summary of the SDR state."""
        return (
            f"SDR(dimensions={self._dimensions}, size={int(self._size)}, "
            f"active={len(self.get_sparse())})"
        )


if __name__ == "__main__":
    sdr_one = SDR([10, 10])
    sdr_two = SDR([10, 10])
    sdr_three = SDR([10, 10])
    sdr_cat = SDR([30, 10])

    for label, sdr in (
        ("SDR One", sdr_one),
        ("SDR Two", sdr_two),
        ("SDR Three", sdr_three),
    ):
        sdr.randomize(0.02)
        print(f"{label}: {sdr}")

    sdr_cat.concatenate([sdr_two, sdr_one, sdr_three], axis=0)
    print("Union of SDR One, SDR Two, and SDR Three:", sdr_cat)

    sdr_sparse = SDR([32, 64])
    sdr_sparse.randomize(0.02)
    print("Sparse SDR:", sdr_sparse)
