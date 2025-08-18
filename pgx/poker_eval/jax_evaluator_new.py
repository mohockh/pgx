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
from .tables.top_bit import TOP_BIT
from .tables.trips_other_val import TRIPS_OTHER_VAL

@jax.jit
def rank_cardset_jax(cards: jnp.ndarray, num_suits: int = 4) -> int:
    """
    JAX implementation of the C rankCardset function.
    
    Args:
        cards: Array of card IDs (0-51), can be padded with -1
        
    Returns:
        Hand strength value (higher = better), exactly matching C implementation
    """
    # Convert cards to C-style bySuit representation
    by_suit = cards_to_suit_patterns(cards)
    
    # Step 1: Check for straight flush (mimics C lines 66-80)
    # Use jnp.take to get the values of by_suit, for all four suits.
    by_suit_values = jnp.take(by_suit, jnp.arange(num_suits), unique_indices=True, indices_are_sorted=True)

    # Use jnp.take to look up each suit's by_suit_value in the ONE_SUIT_VAL
    # table.
    one_suit_values = jnp.take(ONE_SUIT_VAL, by_suit_values)

    # Get the maximum ONE_SUIT_VAL across all suits.
    postponed = jnp.max(one_suit_values)
    
    # If straight flush found, return immediately (C lines 76-80)
    def straight_flush_return():
        return postponed
    
    def continue_evaluation():
        # Step 2: Calculate intersection patterns exactly like C (lines 82-90)
        sets_0 = by_suit[0] | by_suit[1]
        sets_1 = by_suit[0] & by_suit[1]
        sets_2 = sets_1 & by_suit[2]
        sets_1 |= sets_0 & by_suit[2]
        sets_0 |= by_suit[2]
        sets_3 = sets_2 & by_suit[3]
        sets_2 |= sets_1 & by_suit[3]
        sets_1 |= sets_0 & by_suit[3]
        sets_0 |= by_suit[3]
        
        # Step 3: Check for quads (C lines 92-97)
        def quads_return():
            r = jnp_top_bit(sets_3)
            return QUADS_VAL[r] + jnp_top_bit(sets_0 ^ jnp.power(2, r))
        
        def no_quads():
            # Step 4: Check for trips or full house (C lines 99-126)
            def trips_or_full_house():
                r = jnp_top_bit(sets_2)
                sets_1_modified = sets_1 ^ jnp.power(2, r)
                
                # Check for full house (C lines 104-109)
                def full_house_return():
                    return TRIPS_VAL[r] + FULL_HOUSE_OTHER_VAL + jnp_top_bit(sets_1_modified)
                
                def not_full_house():
                    # Check if we have flush (C lines 111-115)
                    def flush_return():
                        return postponed
                    
                    def check_straight():
                        postponed_any = ANY_SUIT_VAL[sets_0]
                        
                        # Check for straight (C lines 118-122)
                        def straight_return():
                            return postponed_any
                        
                        def trips_return():
                            # Trips (C lines 124-126)
                            sets_0_modified = sets_0 ^ jnp.power(2, r)
                            return TRIPS_VAL[r] + TRIPS_OTHER_VAL[sets_0_modified].astype(jnp.uint16)
                        
                        return jax.lax.cond(
                            postponed_any >= HANDCLASS_STRAIGHT,
                            straight_return,
                            trips_return
                        )
                    
                    return jax.lax.cond(
                        postponed > 0,
                        flush_return,
                        check_straight
                    )
                
                return jax.lax.cond(
                    sets_1_modified > 0,
                    full_house_return,
                    not_full_house
                )
            
            def no_trips():
                # No trips case (C lines 128-141)
                def flush_return():
                    return postponed
                
                def check_straight():
                    postponed_any = ANY_SUIT_VAL[sets_0]
                    
                    def straight_return():
                        return postponed_any
                    
                    def check_pairs():
                        # Check for pairs (C lines 143-159)
                        def pairs_found():
                            r = jnp_top_bit(sets_1)
                            sets_0_modified = sets_0 ^ jnp.power(2, r)
                            sets_1_modified = sets_1 ^ jnp.power(2, r)
                            
                            # Check for two pair (C lines 149-155)
                            def two_pair_return():
                                sets_0_final = sets_0_modified ^ jnp.power(2, jnp_top_bit(sets_1_modified))
                                return (PAIRS_VAL[r] + 
                                       TWO_PAIR_OTHER_VAL[jnp_top_bit(sets_1_modified)] + 
                                       jnp_top_bit(sets_0_final))
                            
                            def one_pair_return():
                                # One pair (C line 158)
                                return PAIRS_VAL[r] + PAIR_OTHER_VAL[sets_0_modified]
                            
                            return jax.lax.cond(
                                sets_1_modified > 0,
                                two_pair_return,
                                one_pair_return
                            )
                        
                        def high_card_return():
                            # High card (C line 161)
                            return postponed_any
                        
                        return jax.lax.cond(
                            sets_1 > 0,
                            pairs_found,
                            high_card_return
                        )
                    
                    return jax.lax.cond(
                        postponed_any >= HANDCLASS_STRAIGHT,
                        straight_return,
                        check_pairs
                    )
                
                return jax.lax.cond(
                    postponed > 0,
                    flush_return,
                    check_straight
                )
            
            return jax.lax.cond(
                sets_2 > 0,
                trips_or_full_house,
                no_trips
            )
        
        return jax.lax.cond(
            sets_3 > 0,
            quads_return,
            no_quads
        )
    
    return jax.lax.cond(
        postponed >= HANDCLASS_STRAIGHT_FLUSH,
        straight_flush_return,
        continue_evaluation
    )

#@jax.jit
def jnp_top_bit(value: jnp.uint16) -> jnp.uint16:
    """
    JAX implementation using lookup table to match C topBit exactly.
    """
    value |= value >> 1
    value |= value >> 2
    value |= value >> 4
    value |= value >> 8
    return (jnp.bitwise_count(value) - 1).astype(jnp.uint16)
    #return TOP_BIT[value].astype(jnp.uint16)

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
