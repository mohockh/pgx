"""
Constants from the C ACPC implementation.
Extracted from evalHandTables_shrunk.h
"""

import jax.numpy as jnp

# Hand class definitions from C code
HANDCLASS_SINGLE_CARD = 0
HANDCLASS_PAIR = 1287
HANDCLASS_TWO_PAIR = 5005
HANDCLASS_TRIPS = 8606
HANDCLASS_STRAIGHT = 9620
HANDCLASS_FLUSH = 9633
HANDCLASS_FULL_HOUSE = 10920
HANDCLASS_QUADS = 11934
HANDCLASS_STRAIGHT_FLUSH = 12103

# Quad values for each rank (2=0, 3=1, ..., A=12)
QUADS_VAL = jnp.array([
    11934, 11947, 11960, 11973, 11986, 11999, 12012,
    12025, 12038, 12051, 12064, 12077, 12090
], dtype=jnp.uint16)

# Trips values for each rank (2=0, 3=1, ..., A=12)
TRIPS_VAL = jnp.array([
    8606, 8684, 8762, 8840, 8918, 8996, 9074,
    9152, 9230, 9308, 9386, 9464, 9542
], dtype=jnp.uint16)

# Full house other value offset
FULL_HOUSE_OTHER_VAL = HANDCLASS_FULL_HOUSE - HANDCLASS_TRIPS

# Pair values for each rank (2=0, 3=1, ..., A=12)
PAIRS_VAL = jnp.array([
    1287, 1573, 1859, 2145, 2431, 2717, 3003,
    3289, 3575, 3861, 4147, 4433, 4719
], dtype=jnp.uint16)

# Two pair other values for second pair rank
TWO_PAIR_OTHER_VAL = jnp.array([
    3718, 3731, 3744, 3757, 3770, 3783, 3796,
    3809, 3822, 3835, 3848, 3861, 3874
], dtype=jnp.uint16)