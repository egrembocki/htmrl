import numpy as np

from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters


def hamming_distance(sdr1: np.ndarray, sdr2: np.ndarray) -> int:

    return int(np.count_nonzero(sdr1 != sdr2))


def overlap(sdr1: np.ndarray, sdr2: np.ndarray) -> int:

    return int(np.sum(np.logical_and(sdr1, sdr2)))


if __name__ == "__main__":

    encoder = RandomDistributedScalarEncoder()

    sdr_5 = encoder.encode(5)
    sdr_6 = encoder.encode(6)
    sdr_59 = encoder.encode(59)
    sdr_60 = encoder.encode(60)

    sdr_5 = np.array(sdr_5)
    sdr_6 = np.array(sdr_6)
    sdr_59 = np.array(sdr_59)
    sdr_60 = np.array(sdr_60)

    print(f"Hamming distance between 5 and 6: {hamming_distance(sdr_5, sdr_6)}")
    print(f"Overlap between 5 and 6: {overlap(sdr_5, sdr_6)}")
    print(f"Hamming distance between 59 and 60: {hamming_distance(sdr_59, sdr_60)}")
    print(f"Overlap between 59 and 60: {overlap(sdr_59, sdr_60)}")
    print(f"Hamming distance between 5 and 59: {hamming_distance(sdr_5, sdr_59)}")
    print(f"Overlap between 5 and 59: {overlap(sdr_5, sdr_59)}")
    print(f"Hamming distance between 6 and 60: {hamming_distance(sdr_6, sdr_60)}")
    print(f"Overlap between 6 and 60: {overlap(sdr_6, sdr_60)}")
