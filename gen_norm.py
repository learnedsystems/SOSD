import numpy as np
import struct
from scipy.stats import norm, lognorm
import os

# any arbitrary seed value will do, but this one is clearly the best.
np.random.seed(seed=42) 

NUM_KEYS = 200_000_000

print("Generating normal data...")
if not os.path.exists("data/normal_200M_uint32"):
    print("32 bit...")
    keys = np.linspace(0, 1, NUM_KEYS + 2)[1:-1]

    # for some reason, the PPF function seems to use quadratic memory
    # with the size of its input.
    keys = np.array_split(keys, 1000)
    keys = [norm.ppf(x) for x in keys]
    keys = np.array(keys).flatten()
    
    keys = (keys - np.min(keys)) / (np.max(keys) - np.min(keys))
    keys *= 2**32 - 1
    keys = keys.astype(np.uint32)

    with open("data/normal_200M_uint32", "wb") as f:
        f.write(struct.pack("Q", len(keys)))
        keys.tofile(f)

if not os.path.exists("data/normal_200M_uint64"):
    print("64 bit...")
    keys = np.linspace(0, 1, NUM_KEYS + 2)[1:-1]

    # for some reason, the PPF function seems to use quadratic memory
    # with the size of its input.
    keys = np.array_split(keys, 1000)
    keys = [norm.ppf(x) for x in keys]
    keys = np.array(keys).flatten()

    keys = (keys - np.min(keys)) / (np.max(keys) - np.min(keys))
    keys *= 2**63 - 1
    keys = keys.astype(np.uint64)

    with open("data/normal_200M_uint64", "wb") as f:
        f.write(struct.pack("Q", len(keys)))
        keys.tofile(f)

print("Generating log normal data...")
if not os.path.exists("data/lognormal_200M_uint32"):
    print("32 bit...")
    keys = np.linspace(0, 1, NUM_KEYS + 2)[1:-1]

    # using a sigma of 2 for the 32 bit keys produces WAY too many
    # duplicates, so we will deviate from the RMI paper
    # and use 1.
    keys = np.array_split(keys, 1000)
    keys = [lognorm.ppf(x, 1) for x in keys]
    keys = np.array(keys).flatten()

    keys = (keys - np.min(keys)) / (np.max(keys) - np.min(keys))
    keys *= 2**32 - 1
    keys = keys.astype(np.uint32)

    with open("data/lognormal_200M_uint32", "wb") as f:
        f.write(struct.pack("Q", len(keys)))
        keys.tofile(f)

if not os.path.exists("data/lognormal_200M_uint64"):
    print("64 bit...")
    keys = np.linspace(0, 1, NUM_KEYS + 2)[1:-1]

    # use a sigma of 2 to match the LIS paper.
    keys = np.array_split(keys, 1000)
    keys = [lognorm.ppf(x, 2) for x in keys]
    keys = np.array(keys).flatten()

    keys = (keys - np.min(keys)) / (np.max(keys) - np.min(keys))
    keys *= 2**63 - 1
    keys = keys.astype(np.uint64)

    with open("data/lognormal_200M_uint64", "wb") as f:
        f.write(struct.pack("Q", len(keys)))
        keys.tofile(f)
