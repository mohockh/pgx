"""
Core poker hand evaluation functions using ACPC-style lookup tables.

Provides fast, accurate hand evaluation compatible with JAX transformations.
"""

import jax
import jax.numpy as jnp
from typing import Union
from .cardset import cards_to_cardset, extract_suit_ranks, count_suit_cards, get_rank_counts
from .tables import (
    HANDCLASS_HIGH_CARD, HANDCLASS_PAIR, HANDCLASS_TWO_PAIR, 
    HANDCLASS_THREE_OF_A_KIND, HANDCLASS_STRAIGHT, HANDCLASS_FLUSH,
    HANDCLASS_FULL_HOUSE, HANDCLASS_FOUR_OF_A_KIND, HANDCLASS_STRAIGHT_FLUSH,
    TOP_BIT, STRAIGHT_PATTERNS
)

@jax.jit
def evaluate_hand(cards: jnp.ndarray) -> int:
    """
    Evaluate poker hand strength from card IDs.
    
    Args:
        cards: Array of 5-7 card IDs (0-51)
        
    Returns:
        Hand strength value (higher = better)
    """
    # Convert to cardset representation
    cardset = cards_to_cardset(cards)
    
    # Extract suit and rank information
    suit_ranks = extract_suit_ranks(cardset)
    rank_counts = get_rank_counts(cardset)
    suit_counts = count_suit_cards(cardset)
    
    # Check for flush
    is_flush = jnp.any(suit_counts >= 5)
    flush_suit = jnp.argmax(suit_counts)
    
    # Check for straight
    straight_rank, is_straight = check_straight(suit_ranks, rank_counts)
    
    # Check for straight flush
    if is_flush and is_straight:
        flush_ranks = suit_ranks[flush_suit]
        sf_rank, is_sf = check_straight_in_suit(flush_ranks)
        if is_sf:
            return HANDCLASS_STRAIGHT_FLUSH + sf_rank
    
    # Count pairs, trips, quads
    pair_ranks, trip_ranks, quad_ranks = analyze_rank_counts(rank_counts)
    
    # Four of a kind
    if jnp.any(quad_ranks > 0):
        quad_rank = get_highest_rank(quad_ranks)
        kicker = get_highest_rank(rank_counts == 1)
        return HANDCLASS_FOUR_OF_A_KIND + (quad_rank << 4) + kicker
    
    # Full house
    if jnp.any(trip_ranks > 0) and jnp.any(pair_ranks > 0):
        trip_rank = get_highest_rank(trip_ranks)
        pair_rank = get_highest_rank(pair_ranks)
        return HANDCLASS_FULL_HOUSE + (trip_rank << 4) + pair_rank
    
    # Flush
    if is_flush:
        flush_ranks = suit_ranks[flush_suit]
        flush_value = get_flush_value(flush_ranks)
        return HANDCLASS_FLUSH + flush_value
    
    # Straight
    if is_straight:
        return HANDCLASS_STRAIGHT + straight_rank
    
    # Three of a kind
    if jnp.any(trip_ranks > 0):
        trip_rank = get_highest_rank(trip_ranks)
        kickers = get_kickers(rank_counts == 1, 2)
        return HANDCLASS_THREE_OF_A_KIND + (trip_rank << 8) + kickers
    
    # Two pair
    pair_count = jnp.sum(pair_ranks > 0)
    if pair_count >= 2:
        high_pair, low_pair = get_two_pairs(pair_ranks)
        kicker = get_highest_rank(rank_counts == 1)
        return HANDCLASS_TWO_PAIR + (high_pair << 8) + (low_pair << 4) + kicker
    
    # One pair
    if jnp.any(pair_ranks > 0):
        pair_rank = get_highest_rank(pair_ranks)
        kickers = get_kickers(rank_counts == 1, 3)
        return HANDCLASS_PAIR + (pair_rank << 12) + kickers
    
    # High card
    kickers = get_kickers(rank_counts == 1, 5)
    return HANDCLASS_HIGH_CARD + kickers

@jax.jit
def check_straight(suit_ranks: jnp.ndarray, rank_counts: jnp.ndarray) -> tuple:
    """
    Check for straight in any combination of suits.
    
    Returns:
        Tuple of (straight_rank, is_straight)
    """
    # Create combined rank pattern
    any_ranks = jnp.zeros(13, dtype=jnp.int32)
    for i in range(13):
        if rank_counts[i] > 0:
            any_ranks = any_ranks.at[i].set(1)
    
    # Convert to bit pattern
    rank_pattern = 0
    for i in range(13):
        if any_ranks[i]:
            rank_pattern |= (1 << i)
    
    # Check each straight pattern
    best_straight = -1
    
    for i, pattern in enumerate(STRAIGHT_PATTERNS):
        if (rank_pattern & pattern) == pattern:
            best_straight = i
    
    is_straight = best_straight >= 0
    straight_rank = jnp.where(is_straight, best_straight, 0)
    
    return straight_rank, is_straight

@jax.jit
def check_straight_in_suit(suit_ranks: int) -> tuple:
    """
    Check for straight in specific suit (for straight flush).
    
    Returns:
        Tuple of (straight_rank, is_straight_flush)
    """
    best_straight = -1
    
    for i, pattern in enumerate(STRAIGHT_PATTERNS):
        if (suit_ranks & pattern) == pattern:
            best_straight = i
    
    is_sf = best_straight >= 0
    sf_rank = jnp.where(is_sf, best_straight, 0)
    
    return sf_rank, is_sf

@jax.jit
def analyze_rank_counts(rank_counts: jnp.ndarray) -> tuple:
    """
    Analyze rank counts to find pairs, trips, quads.
    
    Returns:
        Tuple of (pair_ranks, trip_ranks, quad_ranks) as binary patterns
    """
    pair_ranks = jnp.zeros(13, dtype=jnp.int32)
    trip_ranks = jnp.zeros(13, dtype=jnp.int32)
    quad_ranks = jnp.zeros(13, dtype=jnp.int32)
    
    for i in range(13):
        count = rank_counts[i]
        if count == 2:
            pair_ranks = pair_ranks.at[i].set(1)
        elif count == 3:
            trip_ranks = trip_ranks.at[i].set(1)
        elif count == 4:
            quad_ranks = quad_ranks.at[i].set(1)
    
    return pair_ranks, trip_ranks, quad_ranks

@jax.jit
def get_highest_rank(rank_pattern: jnp.ndarray) -> int:
    """Get highest rank from binary pattern."""
    for i in range(12, -1, -1):
        if rank_pattern[i] > 0:
            return i
    return 0

@jax.jit
def get_two_pairs(pair_ranks: jnp.ndarray) -> tuple:
    """Get highest two pair ranks."""
    high_pair = 0
    low_pair = 0
    found_pairs = 0
    
    for i in range(12, -1, -1):
        is_pair = pair_ranks[i] > 0
        # First pair found
        high_pair = jnp.where((found_pairs == 0) & is_pair, i, high_pair)
        # Second pair found  
        low_pair = jnp.where((found_pairs == 1) & is_pair, i, low_pair)
        # Update count
        found_pairs = jnp.where(is_pair, found_pairs + 1, found_pairs)
    
    return high_pair, low_pair

@jax.jit
def get_kickers(kicker_pattern: jnp.ndarray, num_kickers: int) -> int:
    """Get kicker value from highest cards."""
    kickers = 0
    count = 0
    
    for i in range(12, -1, -1):
        if kicker_pattern[i] > 0 and count < num_kickers:
            kickers |= (i << (4 * count))
            count += 1
    
    return kickers

@jax.jit
def get_flush_value(flush_ranks: int) -> int:
    """Calculate flush value from rank pattern."""
    value = 0
    count = 0
    
    for i in range(12, -1, -1):
        if (flush_ranks & (1 << i)) and count < 5:
            value |= (i << (4 * count))
            count += 1
    
    return value

@jax.jit
def batch_evaluate(hands: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate multiple hands in parallel.
    
    Args:
        hands: Array of shape (batch_size, num_cards) with card IDs
        
    Returns:
        Array of hand strength values
    """
    return jax.vmap(evaluate_hand)(hands)

@jax.jit
def hand_vs_hand(hand1: jnp.ndarray, hand2: jnp.ndarray) -> int:
    """
    Compare two hands.
    
    Args:
        hand1: First hand (array of card IDs)
        hand2: Second hand (array of card IDs)
        
    Returns:
        1 if hand1 wins, -1 if hand2 wins, 0 if tie
    """
    strength1 = evaluate_hand(hand1)
    strength2 = evaluate_hand(hand2)
    
    return jnp.sign(strength1 - strength2)