"""
@Nemunta - NuPic
* Parameters for the RandomDistributedScalarEncoder (RDSE)
*
* Members "activeBits" & "sparsity" are mutually exclusive, specify exactly one
* of them.
*
* Members "radius", "resolution", & "category" are mutually exclusive, specify
* exactly one of them.
"""

import copy
import math
import random
import struct
from dataclasses import dataclass
from typing import Iterable, List, Tuple, override

import mmh3
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.sdr_layer.sdr import SDR


@dataclass
class RDSEParameters:

    size: int = 2048
    """
    * Member "size" is the total number of bits in the encoded output SDR.
    """
    active_bits: int = 40
    """
    * Member "activeBits" is the number of true bits in the encoded output SDR.
    """
    sparsity: float = 0.0
    """
    * Member "sparsity" is the fraction of bits in the encoded output which this
    * encoder will activate. This is an alternative way to specify the member
    * "activeBits".
    """
    radius: float = 0.0
    """
    * Member "radius" Two inputs separated by more than the radius have
    * non-overlapping representations. Two inputs separated by less than the
    * radius will in general overlap in at least some of their bits. You can
    * think of this as the radius of the input.
    """
    resolution: float = 1.0
    """
    * The representation will only "shift" or change significantly every 1 unit.
    """
    category: bool = False
    """
    * Member "category" means that the inputs are enumerated categories.
    * If true then this encoder will only encode unsigned integers, and all
    * inputs will have unique / non-overlapping representations.
    """
    seed: int = 42
    """
    * Member "seed" forces different encoders to produce different outputs, even
    * if the inputs and all other parameters are the same.  Two encoders with the
    * same seed, parameters, and input will produce identical outputs.
    *
    * The seed 0 is special.  Seed 0 is replaced with a random number.
    """


"""
 * Encodes a real number as a set of randomly generated activations.
 *
 * Description:
 * The RandomDistributedScalarEncoder (RDSE) encodes a numeric scalar (floating
 * point) value into an SDR.  The RDSE is more flexible than the ScalarEncoder.
 * This encoder does not need to know the minimum and maximum of the input
 * range.  It does not assign an input->output mapping at construction.  Instead
 * the encoding is determined at runtime.

"""


class RandomDistributedScalarEncoder(BaseEncoder[float]):
    """Builds a Random Distributed Scalar Encoder (RDSE), with mmhr3 hashing."""

    def __init__(
        self, parameters: RDSEParameters = RDSEParameters(), dimensions: list[int] | None = None
    ):
        self._parameters = copy.deepcopy(parameters)
        self._parameters = self.check_parameters(self._parameters)

        self._size = self._parameters.size
        self._active_bits = self._parameters.active_bits
        self._sparsity = self._parameters.sparsity
        self._radius = self._parameters.radius
        self._resolution = self._parameters.resolution
        self._category = self._parameters.category
        self._seed = self._parameters.seed
        self._encoding_cache: dict[float, List[int]] = {}
        self.knn: KNeighborsRegressor
        self.encoding: bool = False

        super().__init__(dimensions, self._size)

    @override
    def encode(self, input_value: float) -> List[int]:
        """Encode the input value into a binary vector."""
        self.register_encoding(input_value)
        return self._compute_encoding(input_value)

    def register_encoding(self, input_value: float, encoded: List[int] | None = None) -> List[int]:
        """Caches an encoding so decode_closest can compare against it."""
        vector = encoded if encoded is not None else self._compute_encoding(input_value)
        if len(vector) != self.size:
            raise ValueError("Stored encoding must be the same length as encoder size")
        self._encoding_cache[input_value] = vector
        return vector

    def clear_registered_encodings(self) -> None:
        """Clears cached encodings used for nearest-neighbor decoding."""
        self._encoding_cache.clear()

    def _compute_encoding(self, input_value: float) -> List[int]:
        """
        Uses murmurhash3 algorithm to encode a float value.

        :param self: Description
        :param input_value: The value we want encoded.
        :type input_value: float
        :return: Returns a list of integers that signify an SDR.
        :rtype: List[int]
        """
        if self._category:
            if input_value != int(input_value) or input_value < 0:
                raise ValueError("Input to category encoder must be an unsigned integer")
        self.encoding = True
        data = [0] * self.size

        index = int(input_value / self._resolution)

        for offset in range(self._active_bits):
            hash_buffer = index + offset

            """
                The lower case i in the struct.pack makes this take in signed 32 bit integers.
                It is important to note the previous iteration used an upper case I which
                made this not take negative values. The struct.pack converts an integer
                into a byte representation.
            """
            bucket = mmh3.hash(struct.pack("i", hash_buffer), self._seed, signed=False)
            bucket = bucket % self.size
            """
                Don't worry about hash collisions.  Instead measure the critical
                properties of the encoder in unit tests and quantify how significant
                the hash collisions are.  This encoder can not fix the collisions
                because it does not record past encodings.  Collisions cause small
                deviations in the sparsity or semantic similarity, depending on how
                they're handled.
            """
            data[bucket] = 1
        return data

    @staticmethod
    def _overlap(first: List[int], second: List[int]) -> int:
        """
        Checks for overlapping bits in two Lists of integers.

        :param first: The first list in checking.
        :type first: List[int]
        :param second: The second list in checking.
        :type second: List[int]
        :return: Returns the overlapping bits.
        :rtype: int
        """
        if len(first) != len(second):
            raise ValueError("Vectors must be the same length to compute overlap")
        return sum(1 for a, b in zip(first, second) if a == 1 and b == 1)

    def sparsify(self, vector: List[int]) -> List[int]:
        """Converts a sparse activity vector to a activation list."""
        return [i for i, bit in enumerate(vector) if bit == 1]

    def decode(
        self, encoded: List[int], candidates: Iterable[float] | None = None
    ) -> Tuple[float | None, float]:
        """Returns the value whose encoding overlaps the most with the provided SDR."""
        if len(encoded) != self.size:
            raise ValueError(
                f"Encoded input size ({len(encoded)}) does not match encoder size ({self.size})"
            )

        search_values = (
            list(candidates) if candidates is not None else list(self._encoding_cache.keys())
        )
        if not search_values:
            raise ValueError("No candidate encodings available for decoding")

        best_value: float | None = None
        best_overlap = -1

        for candidate in search_values:
            candidate_encoding = self._encoding_cache.get(candidate)
            if candidate_encoding is None:
                candidate_encoding = self.register_encoding(candidate)
            overlap = self._overlap(encoded, candidate_encoding)
            if overlap > best_overlap:
                best_overlap = overlap
                best_value = candidate

        confidence = (
            best_overlap / self._active_bits if best_overlap >= 0 and self._active_bits else 0.0
        )
        return best_value, confidence

    def make_knn(self) -> None:
        """
        Alternate method to decode where it employs a knn regressor model.

        :param self: Description
        """
        if not self._encoding_cache:
            raise ValueError("No registered encodings available to build KNN")

        x = np.array(list(self._encoding_cache.values()), dtype=np.uint8)
        y = np.array(list(self._encoding_cache.keys()), dtype=np.float32)

        knn = KNeighborsRegressor(
            n_neighbors=min(2, len(y)),
            weights="distance",
            metric="hamming",
        )
        knn.fit(x, y)
        self.knn = knn

    def decode_knn(self, encoded: List[int]) -> float:
        """Returns the value whose encoding overlaps the most with the provided SDR."""
        if len(encoded) != self.size:
            raise ValueError("Encoded input must match encoder size")
        if self.encoding:
            self.make_knn()
            self.encoding = False
        query = np.asarray(encoded, dtype=np.int8).reshape(1, -1)
        result = self.knn.predict(query)
        return result.item()

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: RDSEParameters):
        """Method to check mutually exclusive parameters and fill in missing values.

        Args:
            parameters (RDSEParameters): The parameters to check and fill in.
        Returns:
            RDSEParameters: The checked and filled in parameters.
        Raises:
            AssertionError: If the parameters are invalid.

        """

        assert parameters.size > 0

        num_active_args = 0

        # sparisty XOR active bits
        # Check active bits / sparsity mutual exclusivity

        if parameters.active_bits > 0:
            num_active_args += 1
        if parameters.sparsity > 0.0:
            num_active_args += 1

        assert num_active_args != 0, "Missing argument, need one of: 'activeBits' or 'sparsity'."
        assert (
            num_active_args == 1
        ), "Too many arguments, choose only one of: 'activeBits' or 'sparsity'."

        # radius XOR resolution XOR category
        # Check radius / resolution / category mutual exclusivity
        num_resolution_args = 0
        if parameters.radius > 0.0:
            num_resolution_args += 1
        if parameters.category:
            num_resolution_args += 1
        if parameters.resolution > 0.0:
            num_resolution_args += 1

        assert (
            num_resolution_args != 0
        ), "Missing argument, need one of: 'radius', 'resolution', 'category'."
        assert (
            num_resolution_args == 1
        ), "Too many arguments, choose only one of: 'radius', 'resolution', 'category'."

        # Fill in missing active bits / sparsity

        if parameters.sparsity > 0 and parameters.active_bits == 0:
            assert 0 <= parameters.sparsity <= 1
            parameters.active_bits = int(round(parameters.size * parameters.sparsity))
            assert parameters.active_bits > 0
            assert parameters.sparsity > 0.0

        # category XOR radius XOR resolution
        # Fill in missing radius / resolution / category
        if parameters.category and parameters.radius <= 0.0 and parameters.resolution <= 0.0:
            parameters.radius = 1.0

            assert parameters.radius > 0.0
            assert parameters.resolution <= 0.0
            assert parameters.category

        if parameters.radius > 0.0 and parameters.resolution <= 0.0 and not parameters.category:
            assert parameters.active_bits > 0
            parameters.resolution = parameters.radius / parameters.active_bits
            parameters.category = False

            assert parameters.resolution > 0.0
            assert not parameters.category
            assert parameters.radius > 0.0

        if parameters.resolution > 0.0 and parameters.radius <= 0.0 and not parameters.category:
            assert parameters.active_bits > 0
            parameters.radius = float(parameters.active_bits) * parameters.resolution
            parameters.category = False

            assert parameters.radius > 0.0
            assert not parameters.category
            assert parameters.resolution > 0.0

        # Handle seed == 0 case
        while parameters.seed == 0:
            parameters.seed = random.getrandbits(32)

        return parameters


if __name__ == "__main__":
    params = RDSEParameters(
        size=2048,
        sparsity=0.02,
        radius=1.0,
        active_bits=0,
        category=False,
        seed=12345,
    )
    encoder = RandomDistributedScalarEncoder(params)
    a = encoder.encode(0.1)
    b = encoder.encode(5.1)

    def _overlap_count(first: list[int], second: list[int]) -> int:
        return int(np.count_nonzero(first == second))

    print(_overlap_count(a, b))
    print(encoder.decode(a))
    print(encoder.decode(b))
    # Tests
    """
    params = RDSEParameters(
        size=2048, active_bits=40, sparsity=0.0, radius=0.0, resolution=1.0, category=False, seed=42
    )
    e1 = RandomDistributedScalarEncoder(params)
    o1 = SDR([e1.size])
    e1.encode(10.0, o1)
    print("Encoded 10.0 \n")
    print("Decode prediction is: \n")

    e2 = RandomDistributedScalarEncoder(params)

    o2 = SDR([e2.size])
    e2.encode(1.0, o2)
    print("Sparse is: \n")

    params2 = RDSEParameters(
        size=2048, active_bits=40, sparsity=0.0, radius=0.0, resolution=1.0, category=False, seed=42
    )
    e3 = RandomDistributedScalarEncoder(params2)
    o3 = SDR([e3.size])
    e3.encode(10.0, o3)
    print("Sparse is: \n")

    encoder = RandomDistributedScalarEncoder()
    output = SDR([encoder.size])
    encoder.encode(10.0, output)
    print("Sparse is: \n")
    output2 = SDR([encoder.size])
    encoder.encode(10.0, output2)
    print("Sparse is: \n")

    encoder = RandomDistributedScalarEncoder()
    output = SDR([encoder.size])

    train_values = [random.uniform(0.0, 100.0) for _ in range(200)]
    for v in train_values:
        encoder.encode(v, output)
    test_values = [random.uniform(0.0, 100.0) for _ in range(20)]
    print("__value_____predicted___error")
    y_true = []
    y_pred = []
    for v in test_values:
        encoder.encode(v, output)
        pred = encoder.decode(output)
        y_true.append(v)
        y_pred.append(pred)
        error = pred - v
        print(f"{v:7.3f}   {pred:9.3f}   {error:+7.3f}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    print("RMSE:", rmse)
    """
