import argparse
import numpy as np
import struct

# Random seed to use for random generator.
seed = 42


# Writes values to binary file.
def to_binary(data, filename, uint32):
    if uint32:
        filename += "_uint32"
    else:
        filename += "_uint64"

    with open(filename, "wb") as f:
        # Write size.
        f.write(struct.pack("Q", len(data)))
        # Write values.
        data.tofile(f)

    print("wrote to binary " + filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse", help="sparse (default: no)", action="store_true")
    parser.add_argument("--uint32", help="uint32 instead of uint64 (default: no)", action="store_true")
    parser.add_argument("--many", help="200M instead of 1K keys (default: no)", action="store_true")
    args = parser.parse_args()

    if args.many:
        num_keys = 200000000
        num_keys_str = "200M"
    else:
        num_keys = 1000
        num_keys_str = "1K"

    np.random.seed(seed)

    if args.sparse:
        print("Generating sparse uniform data")
        if args.uint32:
            data = np.random.randint(0, 4294967295, size=num_keys, dtype="uint32")
        else:
            data = np.random.randint(0, 18446744073709551615, size=num_keys, dtype="uint64")

        data.sort()
        to_binary(data, "data/uniform_sparse_" + num_keys_str, args.uint32)
    else:
        print("Generating dense uniform data")
        # Make dense keys not start at 0
        offset = 42
        if args.uint32:
            data = np.arange(offset, num_keys + offset, dtype="uint32")
        else:
            data = np.arange(offset, num_keys + offset, dtype="uint64")

        to_binary(data, "data/uniform_dense_" + num_keys_str, args.uint32)


if __name__ == "__main__":
    main()

