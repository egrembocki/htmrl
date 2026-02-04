import hashlib
import random

import numpy as np

from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder

scalarEncoder = ScalarEncoder()


random.seed(32)
key_vector = np.zeros(256, dtype=np.uint8)

shift_one = 0xC57B
shift_two = 0x1A3F


for i in range(len(key_vector)):
    rand_int = random.randint(0, 255)
    key_vector[i] = rand_int

print(f"Initial bit 0 value: {key_vector[0]:b}")
