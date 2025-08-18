"""
JAX-compatible poker hand evaluator that exactly mimics the C ACPC rankCardset implementation.

This version uses the same lookup tables and algorithm flow as the original C code
for perfect compatibility and correctness.
"""

import jax
import jax.numpy as jnp
from .cardset import cards_to_suit_patterns
from .tables.constants import (
    HANDCLASS_STRAIGHT_FLUSH, HANDCLASS_QUADS, HANDCLASS_FULL_HOUSE,
    HANDCLASS_FLUSH, HANDCLASS_STRAIGHT, HANDCLASS_TRIPS,
    QUADS_VAL, TRIPS_VAL, FULL_HOUSE_OTHER_VAL, PAIRS_VAL, TWO_PAIR_OTHER_VAL
)
from .tables.one_suit_val import ONE_SUIT_VAL
from .tables.any_suit_val import ANY_SUIT_VAL
from .tables.pair_other_val import PAIR_OTHER_VAL
from .tables.trips_other_val import TRIPS_OTHER_VAL

@jax.jit
def rank_cardset_jax(cards: jnp.ndarray, num_suits: int = 4) -> int:
    """
    Flattened JAX implementation with reduced conditionals.
    
    Args:
        cards: Array of card IDs (0-51), can be padded with -1
        
    Returns:
        Hand strength value (higher = better), exactly matching C implementation
    """
    # Convert cards to C-style bySuit representation
    by_suit = cards_to_suit_patterns(cards)
    
    # Step 1: Pre-compute all values
    by_suit_values = jnp.take(by_suit, jnp.arange(num_suits), unique_indices=True, indices_are_sorted=True)
    one_suit_values = jnp.take(ONE_SUIT_VAL, by_suit_values)
    postponed = jnp.max(one_suit_values)
    
    # Early return for straight flush
    is_straight_flush = postponed >= HANDCLASS_STRAIGHT_FLUSH
    
    # Calculate sets exactly like C (lines 82-90)
    sets_0 = by_suit[0] | by_suit[1]
    sets_1 = by_suit[0] & by_suit[1]
    sets_2 = sets_1 & by_suit[2]
    sets_1 |= sets_0 & by_suit[2]
    sets_0 |= by_suit[2]
    sets_3 = sets_2 & by_suit[3]
    sets_2 |= sets_1 & by_suit[3]
    sets_1 |= sets_0 & by_suit[3]
    sets_0 |= by_suit[3]
    
    # Pre-compute common values
    postponed_any = ANY_SUIT_VAL[sets_0]
    
    # Pre-compute condition flags
    has_quads = sets_3 > 0
    has_trips = sets_2 > 0
    has_flush = postponed > 0
    has_straight = postponed_any >= HANDCLASS_STRAIGHT
    has_pairs = sets_1 > 0
    
    # Calculate rank values first
    rank_values = jnp.array([sets_1, sets_2, sets_3], dtype=jnp.uint16)
    rank_topbits = jnp_top_bit(rank_values)
    pair_rank, trips_rank, quads_rank = rank_topbits[0], rank_topbits[1], rank_topbits[2]
    
    # Pre-compute power values (expensive operation)
    power_quads = jnp.uint16(1) << quads_rank
    power_trips = jnp.uint16(1) << trips_rank
    power_pair = jnp.uint16(1) << pair_rank
    
    # Pre-compute all XOR operations
    sets_0_xor_quads = sets_0 ^ power_quads
    sets_1_for_trips = sets_1 ^ power_trips
    sets_0_for_trips = sets_0 ^ power_trips
    sets_0_for_pair = sets_0 ^ power_pair
    sets_1_for_pair = sets_1 ^ power_pair
    
    # Calculate remaining top_bit values at once
    kicker_values = jnp.array([
        sets_0_xor_quads,                 # for quads_val
        sets_1_for_trips,                 # for full_house_val
        sets_1_for_pair,                  # second_pair_rank
    ], dtype=jnp.uint16)
    
    kicker_topbits = jnp_top_bit(kicker_values)
    quads_kicker = kicker_topbits[0]
    fullhouse_kicker = kicker_topbits[1] 
    second_pair_rank = kicker_topbits[2]
    
    # Calculate sets_0_final and its top_bit
    sets_0_final = sets_0_for_pair ^ (jnp.uint16(1) << second_pair_rank)
    two_pair_kicker = jnp_top_bit(sets_0_final)
    
    # Pre-compute all possible return values
    straight_flush_val = postponed
    quads_val = QUADS_VAL[quads_rank] + quads_kicker
    
    # Full house check
    has_full_house = has_trips & (sets_1_for_trips > 0)
    full_house_val = TRIPS_VAL[trips_rank] + FULL_HOUSE_OTHER_VAL + fullhouse_kicker
    
    # Trips value
    trips_val = TRIPS_VAL[trips_rank] + TRIPS_OTHER_VAL[sets_0_for_trips].astype(jnp.uint16)
    
    # Flush value
    flush_val = postponed
    
    # Straight value
    straight_val = postponed_any
    
    # Two pair check
    has_two_pair = has_pairs & (sets_1_for_pair > 0)
    two_pair_val = (PAIRS_VAL[pair_rank] + 
                   TWO_PAIR_OTHER_VAL[second_pair_rank] + 
                   two_pair_kicker)
    
    # One pair value
    one_pair_val = PAIRS_VAL[pair_rank] + PAIR_OTHER_VAL[sets_0_for_pair]
    
    # High card value
    high_card_val = postponed_any
    
    # Flattened selection logic using jax.lax.select
    # Start from bottom and work up
    result = high_card_val
    result = jax.lax.select(has_pairs & ~has_two_pair, one_pair_val, result)
    result = jax.lax.select(has_two_pair, two_pair_val, result)
    result = jax.lax.select(has_straight & ~has_flush & ~has_trips, straight_val, result)
    result = jax.lax.select(has_flush & ~has_trips, flush_val, result)
    result = jax.lax.select(has_trips & ~has_full_house & ~has_flush & ~has_straight, trips_val, result)
    result = jax.lax.select(has_trips & has_straight & ~has_flush & ~has_full_house, straight_val, result)
    result = jax.lax.select(has_trips & has_flush & ~has_full_house, flush_val, result)
    result = jax.lax.select(has_full_house, full_house_val, result)
    result = jax.lax.select(has_quads, quads_val, result)
    result = jax.lax.select(is_straight_flush, straight_flush_val, result)
    
    return result

@jax.jit
def jnp_top_bit(value: jnp.uint16) -> jnp.uint16:
    """
    Using bit arithmetic instead of topBit lookup table: log2(int) - 1.
    Works on both scalars and arrays.
    """
    value |= value >> 1
    value |= value >> 2
    value |= value >> 4
    value |= value >> 8
    return (jnp.bitwise_count(value) - 1).astype(jnp.uint16)

# Main evaluation function that matches the original interface
@jax.jit
def evaluate_hand_jax(cards: jnp.ndarray) -> int:
    """
    Main hand evaluation function using C ACPC algorithm.
    
    Args:
        cards: Array of card IDs (0-51), can be padded with -1
        
    Returns:
        Hand strength value (higher = better)
    """
    return rank_cardset_jax(cards)
