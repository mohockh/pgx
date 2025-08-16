"""
Simplified poker hand evaluator for initial testing.

This version avoids complex JAX control flow and focuses on basic functionality.
"""

import jax
import jax.numpy as jnp
from .cardset import parse_card

def evaluate_hand_simple(cards: jnp.ndarray) -> int:
    """
    Simple hand evaluation - not JIT compiled initially.
    
    Args:
        cards: Array of card IDs (0-51)
        
    Returns:
        Hand strength value (higher = better)
    """
    # Convert cards to suits and ranks
    suits = cards // 13
    ranks = cards % 13
    
    # Count ranks
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    for i in range(len(cards)):
        rank = ranks[i]
        rank_counts = rank_counts.at[rank].add(1)
    
    # Count suits  
    suit_counts = jnp.zeros(4, dtype=jnp.int32)
    for i in range(len(cards)):
        suit = suits[i]
        suit_counts = suit_counts.at[suit].add(1)
    
    # Check for flush
    is_flush = jnp.any(suit_counts >= 5)
    
    # Check for straight
    is_straight = check_straight_simple(rank_counts)
    
    # Count pairs/trips/quads
    pairs = jnp.sum(rank_counts == 2)
    trips = jnp.sum(rank_counts == 3) 
    quads = jnp.sum(rank_counts == 4)
    
    # Classify hand
    if is_straight and is_flush:
        return 8000 + jnp.max(ranks)  # Straight flush
    elif quads > 0:
        return 7000 + jnp.max(jnp.where(rank_counts == 4, jnp.arange(13), 0))  # Four of a kind
    elif trips > 0 and pairs > 0:
        return 6000 + jnp.max(jnp.where(rank_counts == 3, jnp.arange(13), 0))  # Full house
    elif is_flush:
        return 5000 + jnp.max(ranks)  # Flush
    elif is_straight:
        return 4000 + jnp.max(ranks)  # Straight
    elif trips > 0:
        return 3000 + jnp.max(jnp.where(rank_counts == 3, jnp.arange(13), 0))  # Three of a kind
    elif pairs >= 2:
        return 2000 + jnp.max(jnp.where(rank_counts == 2, jnp.arange(13), 0))  # Two pair
    elif pairs >= 1:
        return 1000 + jnp.max(jnp.where(rank_counts == 2, jnp.arange(13), 0))  # One pair
    else:
        return jnp.max(ranks)  # High card

def check_straight_simple(rank_counts: jnp.ndarray) -> bool:
    """Check for straight in rank counts."""
    # Convert to binary pattern
    has_rank = rank_counts > 0
    
    # Check for 5 consecutive ranks
    for start in range(9):  # 0-8 (up to 9-high straight)
        if jnp.all(has_rank[start:start+5]):
            return True
    
    # Check for wheel (A-5)
    if jnp.all(has_rank[jnp.array([12, 0, 1, 2, 3])]):  # A,2,3,4,5
        return True
        
    return False

def hand_class_simple(strength: int) -> int:
    """Get hand class from strength."""
    if strength >= 8000:
        return 8  # Straight flush
    elif strength >= 7000:
        return 7  # Four of a kind
    elif strength >= 6000:
        return 6  # Full house
    elif strength >= 5000:
        return 5  # Flush
    elif strength >= 4000:
        return 4  # Straight
    elif strength >= 3000:
        return 3  # Three of a kind
    elif strength >= 2000:
        return 2  # Two pair
    elif strength >= 1000:
        return 1  # One pair
    else:
        return 0  # High card

def cards_from_string(cards_str: str) -> jnp.ndarray:
    """Convert card string to array of card IDs."""
    card_strs = cards_str.split()
    card_ids = [parse_card(card) for card in card_strs]
    return jnp.array(card_ids, dtype=jnp.int32)