"""Category Encoder implementation"""

import copy
from dataclasses import dataclass
from typing import List

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR


@dataclass
class CategoryParameters:
    """
    The w is the width in bits per category. So, if you have 5 categories and w=3
    we will have 5*3+3=18 bits total. The extra 3 comes from the unknown category.
    """

    w: int
    """
    List of categories to use.
    """
    category_list: List[str]
    """
    This is the name of the encoder, can probably be removed.
    """
    """
    This is an optional default true bool. The category encoder will use the
    RDSE for each category encoded unless this is false, then it will use a
    basic scalar encoder like the htm core implementation.
    """
    rdse_used: bool = True


class CategoryEncoder(BaseEncoder):
    """
    Encodes a list of discrete categories (described by strings), that aren't
    related to each other, so we never emit a mixture of categories.

    The value of zero is reserved for "unknown category"

    Internally we use a :class:`.ScalarEncoder` with a radius of 1, but since we
    only encode integers, we never get mixture outputs.

    The :class:`.SDRCategoryEncoder` uses a different method to encode categories.

    :param categoryList: list of discrete string categories
    :param forced: if True, skip checks for parameters' settings; see
                    :class:`.ScalarEncoder` for details. (default False)
    """

    def __init__(self, parameters: CategoryParameters, dimensions: List[int] | None = None):

        self._parameters = copy.deepcopy(parameters)
        self._w = self._parameters.w
        self._category_list = self._parameters.category_list
        self._RDSEused = self._parameters.rdse_used
        self._num_categories = len(self._category_list) + 1
        self._size = self._num_categories * self._w

        super().__init__(dimensions, self._size)
        """
        If we want the RDSE to be used this will set our encoder object equal to an RDSE with the proper paremeters.
        """
        if self._RDSEused:
            self.rdsep = RDSEParameters(
                size=self._num_categories * self._w,
                active_bits=self._w,
                sparsity=0.0,
                radius=1.0,
                resolution=0.0,
                category=False,
                seed=0,
            )
            self.encoder = RandomDistributedScalarEncoder(self.rdsep, dimensions=[self.rdsep.size])
            self._dimensions = [self.rdsep.size]
            """
            This means we want the scalar encoder to be used and this sets our encoder object to a Scalar encoder with proper parameters.
            """
        else:
            self.sp = ScalarEncoderParameters(
                minimum=0,
                maximum=float(self._num_categories - 1),
                clip_input=False,
                periodic=False,
                category=False,
                active_bits=self._w,
                sparsity=0.0,
                size=0,
                radius=1.0,
                resolution=0.0,
            )
            self.encoder = ScalarEncoder(self.sp, dimensions=[self.sp.size])
            self._dimensions = [self.sp.size]

    def encode(self, input_value: str, output_sdr: SDR) -> None:
        if input_value not in self._category_list:
            index = 0
        else:
            index = self._category_list.index(input_value) + 1
        self.encoder.encode(float(index), output_sdr)

    def check_parameters(self, parameters: CategoryParameters):
        if parameters.w <= 0:
            raise ValueError("Parameter 'w' must be positive.")
        if not parameters.category_list:
            raise ValueError("category_list cannot be empty.")
        if len(set(parameters.category_list)) != len(parameters.category_list):
            raise ValueError("category_list contains duplicate entries.")
        return parameters


"""# This tests the Scalar and RDSE versions to make sure we are getting correct encodings
categories = ["ES", "GB", "US"]
parameters = CategoryParameters(w=3, category_list=categories, RDSEused=False)
e = CategoryEncoder(parameters=parameters)
a = SDR([1, 12])
e.encode("US", a)
assert a.get_dense() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
print("Encoding is correct:", a.get_dense())
e.encode("ES", a)
assert a.get_dense() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
print("Encoding is correct:", a.get_dense())
e.encode("US", a)
assert a.get_dense() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
print("Encoding is correct:", a.get_dense())
e.encode("NA", a)
assert a.get_dense() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print("Encoding is correct:", a.get_dense())

categories = ["ES", "GB", "US"]
parameters = CategoryParameters(w=3, category_list=categories)
e1 = CategoryEncoder(parameters=parameters)
a1 = SDR([1, 12])
a2 = SDR([1, 12])
e1.encode("ES", a1)
e1.encode("ES", a2)
print(a1.get_dense())
print(a2.get_dense())
assert a1.get_dense() == a2.get_dense()
e1.encode("GB", a1)
e1.encode("GB", a2)
print(a1.get_dense())
print(a2.get_dense())
assert a1.get_dense() == a2.get_dense()
e1.encode("US", a1)
e1.encode("US", a2)
print(a1.get_dense())
print(a2.get_dense())
assert a1.get_dense() == a2.get_dense()
e1.encode("NA", a1)
e1.encode("NA", a2)
print(a1.get_dense())
print(a2.get_dense())
assert a1.get_dense() == a2.get_dense()"""
