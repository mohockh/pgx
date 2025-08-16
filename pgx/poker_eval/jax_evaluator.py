"""
JAX-compatible poker hand evaluator.

This version uses only JAX operations and avoids Python control flow.
"""

import jax
import jax.numpy as jnp

@jax.jit
def evaluate_hand_jax(cards: jnp.ndarray) -> int:
    """
    JAX-compatible hand evaluation.
    
    Args:
        cards: Array of card IDs (0-51), can be padded with -1
        
    Returns:
        Hand strength value (higher = better)
    """
    # Filter out invalid cards (-1)
    valid_mask = cards >= 0
    valid_cards = jnp.where(valid_mask, cards, 0)
    num_valid = jnp.sum(valid_mask)
    
    # Extract suits and ranks
    suits = valid_cards // 13
    ranks = valid_cards % 13
    
    # Count ranks (only for valid cards)
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    for i in range(cards.shape[0]):
        # Only count if card is valid
        is_valid = valid_mask[i]
        rank = ranks[i]
        rank_counts = rank_counts.at[rank].add(jnp.where(is_valid, 1, 0))
    
    # Count suits (only for valid cards)  
    suit_counts = jnp.zeros(4, dtype=jnp.int32)
    for i in range(cards.shape[0]):
        is_valid = valid_mask[i]
        suit = suits[i]
        suit_counts = suit_counts.at[suit].add(jnp.where(is_valid, 1, 0))
    
    # Check for flush
    is_flush = jnp.any(suit_counts >= 5)
    flush_suit = jnp.argmax(suit_counts)
    
    # Check for straight
    is_straight, straight_rank = check_straight_jax(rank_counts)
    
    # Count pairs/trips/quads
    pairs = jnp.sum(rank_counts == 2)
    trips = jnp.sum(rank_counts == 3)
    quads = jnp.sum(rank_counts == 4)
    
    # Get kickers (ranks that appear exactly once)
    kicker_ranks = get_sorted_kickers(rank_counts, ranks, valid_mask)
    
    # Get highest pair/trip/quad rank
    quad_rank = get_max_rank_with_count(rank_counts, 4)
    trip_rank = get_max_rank_with_count(rank_counts, 3)
    pair_rank = get_max_rank_with_count(rank_counts, 2)
    
    # For two pair, get both pair ranks
    pair_ranks_sorted = get_sorted_pairs(rank_counts)
    
    # Check for straight flush
    is_straight_flush = is_straight & is_flush & check_straight_flush(suits, ranks, valid_mask, flush_suit)
    
    # Hand classification using JAX select with proper kicker encoding
    strength = jnp.select([
        is_straight_flush,       # Straight flush
        quads > 0,              # Four of a kind
        (trips > 0) & (pairs > 0),  # Full house
        is_flush,               # Flush
        is_straight,            # Straight
        trips > 0,              # Three of a kind
        pairs >= 2,             # Two pair
        pairs >= 1,             # One pair
    ], [
        8000000 + straight_rank * 1000,  # Straight flush: 8,000,000 - 8,012,000
        7000000 + quad_rank * 1000 + get_kicker_value(kicker_ranks, 1),  # Four of a kind: 7,000,000 - 7,999,999
        6000000 + trip_rank * 1000 + pair_rank * 100,  # Full house: 6,000,000 - 6,999,999
        5000000 + get_flush_kicker_value(suits, ranks, valid_mask, flush_suit),  # Flush: 5,000,000 - 5,999,999
        4000000 + straight_rank * 1000,  # Straight: 4,000,000 - 4,012,000
        3000000 + trip_rank * 1000 + get_kicker_value(kicker_ranks, 2),  # Three of a kind: 3,000,000 - 3,999,999
        2000000 + pair_ranks_sorted[0] * 1000 + pair_ranks_sorted[1] * 100 + get_kicker_value(kicker_ranks, 1),  # Two pair: 2,000,000 - 2,999,999
        1000000 + pair_rank * 1000 + get_kicker_value(kicker_ranks, 3),  # One pair: 1,000,000 - 1,999,999
    ], default=get_kicker_value(kicker_ranks, 5))  # High card: 0 - 999,999
    
    return strength

@jax.jit
def check_straight_jax(rank_counts: jnp.ndarray) -> tuple:
    """Check for straight using JAX operations."""
    has_rank = rank_counts > 0
    
    # Check each possible straight
    straights = jnp.array([
        # Regular straights (5 consecutive) - highest rank as identifier
        jnp.all(has_rank[0:5]),   # 2-6 -> rank 4
        jnp.all(has_rank[1:6]),   # 3-7 -> rank 5
        jnp.all(has_rank[2:7]),   # 4-8 -> rank 6
        jnp.all(has_rank[3:8]),   # 5-9 -> rank 7
        jnp.all(has_rank[4:9]),   # 6-10 -> rank 8
        jnp.all(has_rank[5:10]),  # 7-J -> rank 9
        jnp.all(has_rank[6:11]),  # 8-Q -> rank 10
        jnp.all(has_rank[7:12]),  # 9-K -> rank 11
        jnp.all(has_rank[8:13]),  # 10-A -> rank 12
        # Wheel (A-5) -> rank 3 (5-high, but ace low)
        has_rank[12] & jnp.all(has_rank[0:4]),  # A,2,3,4,5
    ])
    
    # Get highest straight (prefer non-wheel straights)
    straight_ranks = jnp.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 3])  # Wheel is lowest
    has_straight = jnp.any(straights)
    
    # Find highest straight rank (scan from highest to lowest)
    best_straight = 0
    for i in range(9, -1, -1):  # Start from highest non-wheel straight
        best_straight = jnp.where(straights[i], straight_ranks[i], best_straight)
    
    return has_straight, best_straight

@jax.jit
def get_max_rank(ranks: jnp.ndarray, valid_mask: jnp.ndarray) -> int:
    """Get maximum rank from valid cards."""
    # Set invalid cards to -1 so they don't affect max
    masked_ranks = jnp.where(valid_mask, ranks, -1)
    return jnp.max(masked_ranks)

@jax.jit  
def get_max_rank_with_count(rank_counts: jnp.ndarray, target_count: int) -> int:
    """Get highest rank that appears exactly target_count times."""
    # Create mask for ranks with target count
    target_mask = rank_counts == target_count
    # Set non-target ranks to -1
    masked_ranks = jnp.where(target_mask, jnp.arange(13), -1)
    # Return max (or 0 if no matches)
    return jnp.maximum(0, jnp.max(masked_ranks))

@jax.jit
def get_sorted_kickers(rank_counts: jnp.ndarray, ranks: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    """Get kicker ranks (ranks that appear exactly once) sorted highest to lowest."""
    # Create array of kicker ranks (appear exactly once)
    kicker_mask = rank_counts == 1
    kicker_ranks = jnp.where(kicker_mask, jnp.arange(13), -1)
    
    # Sort kickers highest to lowest (use negative for descending sort)
    sorted_indices = jnp.argsort(-kicker_ranks)
    sorted_kickers = kicker_ranks[sorted_indices]
    
    # Filter out -1 values and keep only valid kickers
    valid_kickers = jnp.where(sorted_kickers >= 0, sorted_kickers, -1)
    
    return valid_kickers

@jax.jit  
def get_sorted_pairs(rank_counts: jnp.ndarray) -> jnp.ndarray:
    """Get pair ranks sorted highest to lowest."""
    # Create array of pair ranks (appear exactly twice)
    pair_mask = rank_counts == 2
    pair_ranks = jnp.where(pair_mask, jnp.arange(13), -1)
    
    # Sort pairs highest to lowest
    sorted_indices = jnp.argsort(-pair_ranks)
    sorted_pairs = pair_ranks[sorted_indices]
    
    # Return first two (highest pairs), pad with 0 if needed
    result = jnp.array([0, 0])
    valid_count = jnp.sum(sorted_pairs >= 0)
    result = result.at[0].set(jnp.where(valid_count >= 1, jnp.maximum(0, sorted_pairs[0]), 0))
    result = result.at[1].set(jnp.where(valid_count >= 2, jnp.maximum(0, sorted_pairs[1]), 0))
    
    return result

@jax.jit
def get_kicker_value(kicker_ranks: jnp.ndarray, num_kickers: int) -> int:
    """Encode kicker ranks into a single value."""
    value = 0
    # Use static loop since num_kickers is usually small
    for i in range(5):  # Max 5 kickers
        # Only include if i < num_kickers and valid kicker exists
        include_kicker = (i < num_kickers) & (i < kicker_ranks.shape[0]) & (kicker_ranks[i] >= 0)
        kicker = jnp.where(include_kicker, kicker_ranks[i], 0)
        value += kicker * (13 ** jnp.maximum(0, num_kickers - 1 - i))
    return value

@jax.jit
def get_flush_kicker_value(suits: jnp.ndarray, ranks: jnp.ndarray, valid_mask: jnp.ndarray, flush_suit: int) -> int:
    """Get kicker value for flush (top 5 cards of flush suit)."""
    # Get ranks in flush suit
    flush_mask = (suits == flush_suit) & valid_mask
    flush_ranks = jnp.where(flush_mask, ranks, -1)
    
    # Sort flush ranks highest to lowest
    sorted_indices = jnp.argsort(-flush_ranks)
    sorted_flush_ranks = flush_ranks[sorted_indices]
    
    # Take top 5 cards and encode
    value = 0
    for i in range(5):
        rank = jnp.where((i < sorted_flush_ranks.shape[0]) & (sorted_flush_ranks[i] >= 0), 
                        sorted_flush_ranks[i], 0)
        value += rank * (13 ** (4 - i))
    return value

@jax.jit
def check_straight_flush(suits: jnp.ndarray, ranks: jnp.ndarray, valid_mask: jnp.ndarray, flush_suit: int) -> bool:
    """Check if the flush is also a straight."""
    # Get ranks in flush suit
    flush_mask = (suits == flush_suit) & valid_mask
    flush_rank_counts = jnp.zeros(13, dtype=jnp.int32)
    
    # Count ranks in flush suit (use static loop)
    for i in range(7):  # Max 7 cards
        is_valid = i < suits.shape[0]
        is_flush_card = flush_mask[i] & is_valid
        rank = ranks[i]
        flush_rank_counts = flush_rank_counts.at[rank].add(jnp.where(is_flush_card, 1, 0))
    
    # Check for straight in flush suit
    is_sf, _ = check_straight_jax(flush_rank_counts)
    return is_sf

def hand_class_jax(strength: int) -> int:
    """Get hand class from strength."""
    return jnp.select([
        strength >= 8000000,  # Straight flush
        strength >= 7000000,  # Four of a kind
        strength >= 6000000,  # Full house
        strength >= 5000000,  # Flush
        strength >= 4000000,  # Straight
        strength >= 3000000,  # Three of a kind
        strength >= 2000000,  # Two pair
        strength >= 1000000,  # One pair
    ], [8, 7, 6, 5, 4, 3, 2, 1], default=0)