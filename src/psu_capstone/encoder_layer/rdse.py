import copy
import math
import random
import struct
from dataclasses import dataclass
from typing import override

import mmh3

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.sdr_layer.sdr import SDR

"""
 * Parameters for the RandomDistributedScalarEncoder (RDSE)
 *
 * Members "activeBits" & "sparsity" are mutually exclusive, specify exactly one
 * of them.
 *
 * Members "radius", "resolution", & "category" are mutually exclusive, specify
 * exactly one of them.
"""


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
    radius: float = 1.0
    """
    * Member "radius" Two inputs separated by more than the radius have
    * non-overlapping representations. Two inputs separated by less than the
    * radius will in general overlap in at least some of their bits. You can
    * think of this as the radius of the input.
    """
    resolution: float = 0.0
    """
    * Member "resolution" Two inputs separated by greater than, or equal to the
    * resolution will in general have different representations.
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
    """Random Distributed Scalar Encoder (RDSE) implementation."""

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

        super().__init__(dimensions, self._size)

    """
    Encodes an input value into an SDR with a random distributed scalar encoder.
    We employ the murmur hashing.
    """

    @override
    def encode(self, input_value: float, output_sdr: SDR) -> list[int]:
        assert output_sdr.size == self._size, "Output SDR size does not match encoder size."
        if math.isnan(input_value):
            output_sdr.zero()
            return []
        if self._category:
            if input_value != int(input_value) or input_value < 0:
                raise ValueError("Input to category encoder must be an unsigned integer")

        data = [0] * self.size
        assert self._resolution > 0.0, "Resolution must be greater than 0."
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

        output_sdr.set_dense(data)

        return data

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
    # Tests
    params = RDSEParameters(
        size=2048, active_bits=40, sparsity=0.0, radius=0.0, resolution=1.0, category=False, seed=42
    )
    e1 = RandomDistributedScalarEncoder(params)
    o1 = SDR([e1.size])
    e1.encode(10.0, o1)
    print(o1.get_sparse())

    e2 = RandomDistributedScalarEncoder(params)

    o2 = SDR([e2.size])
    e2.encode(10.0, o2)
    print(o2.get_sparse())

    params2 = RDSEParameters(
        size=2048, active_bits=40, sparsity=0.0, radius=0.0, resolution=1.0, category=False, seed=42
    )
    e3 = RandomDistributedScalarEncoder(params2)
    o3 = SDR([e3.size])
    e3.encode(10.0, o3)
    print(o3.get_sparse())

    encoder = RandomDistributedScalarEncoder()
    output = SDR([encoder.size])
    encoder.encode(10.0, output)
    print(output.get_sparse())
    output2 = SDR([encoder.size])
    encoder.encode(10.0, output2)
    print(output2.get_sparse())
