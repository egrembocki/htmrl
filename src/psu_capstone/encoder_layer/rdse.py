import copy
import math
import random
import struct
from dataclasses import dataclass
from typing import override

import mmh3
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

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
    radius: float = 0.0
    """
    * Member "radius" Two inputs separated by more than the radius have
    * non-overlapping representations. Two inputs separated by less than the
    * radius will in general overlap in at least some of their bits. You can
    * think of this as the radius of the input.
    """
    resolution: float = 1.0
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
        self._sdrs_encoded: list[np.ndarray] = []
        self._input_values_encoded: list[float] = []
        self.knn: KNeighborsRegressor
        self.enc: bool = False

        super().__init__(dimensions, self._size)

    """
    Encodes an input value into an SDR with a random distributed scalar encoder.
    We employ the murmur hashing.
    """

    # Properties
    @property
    def size(self) -> int:
        return self._size

    @size.setter
    def size(self, value: int) -> None:
        self._size = value

    @property
    def active_bits(self) -> int:
        return self._active_bits

    @active_bits.setter
    def active_bits(self, value: int) -> None:
        self._active_bits = value

    @property
    def sparsity(self) -> float:
        return self._sparsity

    @sparsity.setter
    def sparsity(self, value: float) -> None:
        self._sparsity = value

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = value

    @property
    def resolution(self) -> float:
        return self._resolution

    @resolution.setter
    def resolution(self, value: float) -> None:
        self._resolution = value

    @property
    def category(self) -> bool:
        return self._category

    @category.setter
    def category(self, value: bool) -> None:
        self._category = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value

    @override
    def encode(self, input_value: float) -> list[int]:
        # assert output_sdr.size == self._size, "Output SDR size does not match encoder size."
        # if math.isnan(input_value):
        #    output_sdr.zero()
        #    return
        if self._category:
            if input_value != int(input_value) or input_value < 0:
                raise ValueError("Input to category encoder must be an unsigned integer")
        # I am appendding every successful input_value to the local list for knn regressor use.
        self._input_values_encoded.append(input_value)
        self.enc = True

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
        # add the density to the sdrs encoder list for knn regressor use
        self._sdrs_encoded.append(data)
        # output_sdr.set_dense(data)
        return data

    def make_knn(self):
        x = np.array(self._sdrs_encoded, dtype=np.uint8)
        y = np.array(self._input_values_encoded, dtype=np.float32)
        knn = KNeighborsRegressor(n_neighbors=2, weights="distance", metric="hamming")
        knn.fit(x, y)

    def decode(self, input_sdr: SDR) -> float:
        # x = np.array(self._sdrs_encoded, dtype=np.uint8)
        # y = np.array(self._input_values_encoded, dtype=np.float32)
        # knn = KNeighborsRegressor(n_neighbors=2, weights="distance", metric="hamming")
        # knn.fit(x, y)
        if self.enc:
            self.makeKnn()
            self.enc = False
        query = np.asarray(input_sdr.get_dense(), dtype=np.int8).reshape(1, -1)
        result = self.knn.predict(query)
        return result.item()

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: RDSEParameters):
        assert parameters.size > 0

        num_active_args = 0
        if parameters.active_bits > 0:
            num_active_args += 1
        if parameters.sparsity > 0:
            num_active_args += 1

        assert num_active_args != 0, "Missing argument, need one of: 'activeBits' or 'sparsity'."
        assert (
            num_active_args == 1
        ), "Too many arguments, choose only one of: 'activeBits' or 'sparsity'."

        num_resolution_args = 0
        if parameters.radius > 0:
            num_resolution_args += 1
        if parameters.category:
            num_resolution_args += 1
        if parameters.resolution > 0:
            num_resolution_args += 1

        assert (
            num_resolution_args != 0
        ), "Missing argument, need one of: 'radius', 'resolution', 'category'."
        assert (
            num_resolution_args == 1
        ), "Too many arguments, choose only one of: 'radius', 'resolution', 'category'."

        args = parameters

        if args.sparsity > 0:
            assert 0 <= args.sparsity <= 1
            args.active_bits = int(round(args.size * args.sparsity))
            assert args.active_bits > 0

        if args.category:
            args.radius = 1

        if args.radius > 0:
            args.resolution = args.radius / args.active_bits
        elif args.resolution > 0:
            args.radius = args.active_bits * args.resolution

        while args.seed == 0:
            args.seed = random.getrandbits(32)

        return args


if __name__ == "__main__":
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
