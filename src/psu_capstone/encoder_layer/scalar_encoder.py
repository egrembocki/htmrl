"""Scalar Encoder Module implementation. From NuPic Numenta Cpp ported to python
/**
 * These four (4) members define the total number of bits in the output:
 *      size,
 *      radius,
 *      category,
 *      resolution.
 *
 * These are mutually exclusive and only one of them should be non-zero when
 * constructing the encoder. -- Need to refactor ScalarEncoder to take in only params
 */

"""

import copy
import math
from dataclasses import dataclass
from typing import List, Union

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.sdr import SDR


@dataclass
class ScalarEncoderParameters:

    minimum: float
    maximum: float
    """Min and Max
     * Members "minimum" and "maximum" define the range of the input signal.
     * These endpoints are inclusive.
     */"""

    clip_input: bool
    """Whether to clip inputs outside the min/max range.
       /**
     * Member "clipInput" determines whether to allow input values outside the
     * range [minimum, maximum].
     * If true, the input will be clipped into the range [minimum, maximum].
     * If false, inputs outside of the range will raise an error.
     */"""

    periodic: bool
    """Whether the encoder is periodic (circular) or not.
    /**
     * Member "periodic" controls what happens near the edges of the input
     * range.
     *
     * If true, then the minimum & maximum input values are the same and the
     * first and last bits of the output SDR are adjacent.  The contiguous
     * block of 1's wraps around the end back to the beginning.
     *
     * If false, then minimum & maximum input values are the endpoints of the
     * input range, are not adjacent, and activity does not wrap around.
     */
    """

    category: bool
    """Whether the encoder is a category encoder.
    /**
     * Member "category" means that the inputs are enumerated categories.
     * If true then this encoder will only encode unsigned integers, and all
     * inputs will have unique / non-overlapping representations.
     */
    """
    active_bits: int
    sparsity: float
    """Number of active bits or sparsity level.
     /**
     * Member "activeBits" is the number of true bits in the encoded output SDR.
     * The output encodings will have a contiguous block of this many 1's.
     */

      /**
     * Member "sparsity" is an alternative way to specify the member "activeBits".
     * Sparsity requires that the size to also be specified.
     * Specify only one of: activeBits or sparsity.
     */

    """
    size: int
    """Total number of bits in the output SDR.
     /**
     * Member "size" is the total number of bits in the encoded output SDR.
     */
    """
    radius: float
    """Approximate input range (width) covered by the active bits.
    /**
     * Member "radius" Two inputs separated by more than the radius have
     * non-overlapping representations. Two inputs separated by less than the
     * radius will in general overlap in at least some of their bits. You can
     * think of this as the radius of the input.
     */
    """
    resolution: float
    """The smallest difference between two inputs that produces different outputs.
      /**
     * Member "resolution" Two inputs separated by greater than, or equal to the
     * resolution are guaranteed to have different representations.
     */"""


class ScalarEncoder(BaseEncoder):
    """
    /**
     * Encodes a real number as a contiguous block of 1's.
     *
     * Description:
     * The ScalarEncoder encodes a numeric (floating point) value into an array
     * of bits. The output is 0's except for a contiguous block of 1's. The
     * location of this contiguous block varies continuously with the input value.
     *
     * To inspect this run:
     * $ python -m htm.examples.encoders.scalar_encoder --help
     */"""

    def __init__(self, parameters: ScalarEncoderParameters, dimensions: List[int] | None = None):
        self._parameters = copy.deepcopy(parameters)
        self._parameters = self.check_parameters(self._parameters)

        self._minimum = self._parameters.minimum
        self._maximum = self._parameters.maximum
        self._clip_input = self._parameters.clip_input
        self._periodic = self._parameters.periodic
        self._category = self._parameters.category
        self._active_bits = self._parameters.active_bits
        self._sparsity = self._parameters.sparsity
        self._size = self._parameters.size
        self._radius = self._parameters.radius
        self._resolution = self._parameters.resolution

        super().__init__(dimensions, self._size)

    """
        Encodes an input value into an SDR with a block of 1's.

        Description:
        The encode method is responsible for transforming the supplied SDR data structure into
        an SDR that has the encoding of the input value.
    """

    def encode(self, input_value: float, output_sdr: SDR) -> bool:
        assert output_sdr.size == self.size, "Output SDR size does not match encoder size."

        if math.isnan(input_value):
            output_sdr.zero()
            return False

        elif self._clip_input:
            if self._periodic:
                """TODO: implement modlus to inputs"""
                input_value = input_value % self._maximum
                # raise NotImplementedError("Periodic input clipping not implemented.")
            else:
                input_value = max(input_value, self._minimum)
                input_value = min(input_value, self._maximum)
        else:
            if self._category and input_value != float(int(input_value)):
                raise ValueError("Input to category encoder must be an unsigned integer!")
            if not (self._minimum <= input_value <= self._maximum):
                raise ValueError(
                    f"Input must be within range [{self._minimum}, {self._maximum}]! "
                    f"Received {input_value}"
                )

        start = int(round((input_value - self._minimum) / self._resolution))

        """Handle edge case where start + active_bits exceeds output size.
          // The endpoints of the input range are inclusive, which means that the
          // maximum value may round up to an index which is outside of the SDR. Correct
          // this by pushing the endpoint (and everything which rounds to it) onto the
          // last bit in the SDR.
        """
        if not self._periodic:
            start = min(start, output_sdr.size - self._active_bits)

        sparse = output_sdr.get_sparse()
        sparse[:] = range(start, start + self._active_bits)

        if self._periodic:
            for i, bit in enumerate(sparse):
                if bit >= output_sdr.size:
                    sparse[i] = bit - output_sdr.size
            sparse.sort()

        output_sdr.set_sparse(sparse)

        self.__sdr = output_sdr

        return self.__sdr == output_sdr

    # After encode we may need a check_parameters method since most of the encoders have this
    def check_parameters(self, parameters: ScalarEncoderParameters):
        """
        Check parameters method is used to verify that the correct parameters were entered
        and reject the user when they are not.

        Description: This changes and transforms the input that the user has with the parameters
        dataclass. There are many aspects such as the active bit and sparsity being mutually exclusive
        and the size, radius, resolution, and category also being muturally exclusive with each other.
        The user will have an assert that rejects when these are violated.
        """
        assert parameters.minimum <= parameters.maximum
        num_active_args = sum([parameters.active_bits > 0, parameters.sparsity > 0])
        assert num_active_args != 0, "Missing argument, need one of: 'active_bits', 'sparsity'."
        # print(str(parameters.sparsity))
        # print(str(parameters.active_bits))
        assert (
            num_active_args == 1
        ), "Specified both: 'active_bits', 'sparsity'. Specify only one of them." + str(
            num_active_args
        )
        num_size_args = sum(
            [
                parameters.size > 0,
                parameters.radius > 0,
                parameters.category,
                parameters.resolution > 0,
            ]
        )
        assert (
            num_size_args != 0
        ), "Missing argument, need one of: 'size', 'radius', 'resolution', 'category'."
        assert num_size_args == 1, (
            "Too many arguments specified: 'size', 'radius', 'resolution', 'category'. Choose only one of them."
            + str(num_size_args)
            + "     "
            + str(parameters.size)
            + "     "
            + str(parameters.radius)
            + "      "
            + str(parameters.category)
            + "     "
            + str(parameters.resolution)
        )
        if parameters.periodic:
            assert (
                not parameters.clip_input
            ), "Will not clip periodic inputs.  Caller must apply modulus."
        if parameters.category:
            assert not parameters.clip_input, "Incompatible arguments: category & clip_input."
            assert not parameters.periodic, "Incompatible arguments: category & periodic."
            assert parameters.minimum == float(
                int(parameters.minimum)
            ), "Minimum input value of category encoder must be an unsigned integer!"
            assert parameters.maximum == float(
                int(parameters.maximum)
            ), "Maximum input value of category encoder must be an unsigned integer!"

        args = parameters
        if args.category:
            args.radius = 1.0
        if args.sparsity:
            assert 0.0 <= args.sparsity <= 1.0
            assert args.size > 0, "Argument 'sparsity' requires that the 'size' also be given."
            args.active_bits = round(args.size * args.sparsity)
            assert (
                args.active_bits > 0
            ), "sparsity and size must be given so that sparsity * size > 0!"
        if args.periodic:
            extent_width = args.maximum - args.minimum
        else:
            max_inclusive = math.nextafter(args.maximum, math.inf)
            extent_width = max_inclusive - args.minimum
        if args.size > 0:
            if args.periodic:
                args.resolution = extent_width / args.size
            else:
                n_buckets = args.size - (args.active_bits - 1)
                args.resolution = extent_width / (n_buckets - 1)
        else:
            if args.radius > 0.0:
                args.resolution = args.radius / args.active_bits

            needed_bands = math.ceil(extent_width / args.resolution)
            if args.periodic:
                args.size = needed_bands
            else:
                args.size = needed_bands + (args.active_bits - 1)

        # Sanity check the parameters.
        assert args.size > 0
        assert args.active_bits > 0
        assert args.active_bits < args.size

        args.radius = args.active_bits * args.resolution
        assert args.radius > 0

        args.sparsity = args.active_bits / float(args.size)
        assert args.sparsity > 0

        return args


"""p = ScalarEncoderParameters(
        minimum=10.0,
        maximum=20.0,
        clip_input=False,
        periodic=False,
        category=False,
        active_bits=2,
        sparsity=0.0,
        size= 10,
        radius=0.0,
        resolution=0.0,
        size_or_radius_or_category_or_resolution=0,
        active_bits_or_sparsity=0
)
encoder3 = ScalarEncoder(p ,dimensions=[p.size])
output = SDR(dimensions=[p.size])
encoder3.encode(10.0, output)
encoder3.encode(20.0, output)
print(output)
#encoder3.encode(9.9, output) ValueError
#encoder3.encode(20.1, output) ValueError
print(output)"""
