"""
Card representation and conversion utilities for poker hand evaluation.

Provides functions to convert between different card representations and
work with the ACPC cardset format for fast evaluation.

Now uses uint32[2] arrays for memory efficiency while maintaining full compatibility.
"""

import jax
import jax.numpy as jnp
from typing import Union, List, Tuple
from .cardset_ops import (
    create_empty_cardset, create_cardset_from_value, cardset_or,
    cardset_and, cardset_not, set_bit, get_bit, 
    cardset_to_uint64, or_reduce
)

# Maximum number of cards that can be extracted from a cardset
MAX_CARDS_IN_CARDSET = 10

# Pre-computed constants for vectorized suit extraction (computed once at module load)
_SUITS_GRID, _RANKS_GRID = jnp.meshgrid(jnp.arange(4), jnp.arange(13), indexing='ij')
_BIT_POSITIONS_GRID = (_SUITS_GRID << 4) + _RANKS_GRID  # Shape: (4, 13)
_RANK_POWERS_GRID = jnp.uint16(1) << _RANKS_GRID  # Shape: (4, 13)

def card_to_id(suit: int, rank: int) -> int:
    """
    Convert suit and rank to card ID.
    
    Args:
        suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
        rank: 0=2, 1=3, ..., 11=K, 12=A
        
    Returns:
        Card ID (0-51)
    """
    return suit * 13 + rank

def id_to_card(card_id: int) -> Tuple[int, int]:
    """
    Convert card ID to suit and rank.
    
    Args:
        card_id: Card ID (0-51)
        
    Returns:
        Tuple of (suit, rank)
    """
    return card_id // 13, card_id % 13

@jax.jit
def cards_to_cardset(cards: jnp.ndarray) -> jnp.ndarray:
    """
    Convert array of card IDs to ACPC cardset representation using fully vectorized operations.
    Assumes all cards are valid (0-51).
    
    Args:
        cards: Array of valid card IDs (0-51)
        
    Returns:
        uint32[2] cardset array
    """
    cards = cards.astype(jnp.uint32)
    
    # Convert to suits and ranks (vectorized)
    suits = cards // 13
    ranks = cards % 13
    
    # Calculate bit positions: (suit << 4) + rank
    bit_positions = (suits << 4) + ranks
    
    # Create bit masks for all cards simultaneously (vectorized)
    # For low 32 bits (bit_pos < 32)
    low_masks = jnp.where(bit_positions < 32, 
                          jnp.uint32(1) << bit_positions, jnp.uint32(0))
    
    # For high 32 bits (bit_pos >= 32)  
    high_masks = jnp.where(bit_positions >= 32,
                           jnp.uint32(1) << (bit_positions - 32), jnp.uint32(0))
    
    # Combine all masks using vectorized OR operations
    low_combined = jnp.bitwise_or.reduce(low_masks)
    high_combined = jnp.bitwise_or.reduce(high_masks)
    
    return jnp.array([low_combined, high_combined], dtype=jnp.uint32)

@jax.jit
def extract_suit_ranks(cardset: jnp.ndarray) -> jnp.ndarray:
    """
    Extract rank patterns for each suit from cardset using pure bit operations.
    
    Args:
        cardset: uint32[2] cardset array
        
    Returns:
        Array of 4 integers, each with 13 bits representing ranks in that suit
    """
    # Use pre-computed bit positions grid
    bit_positions = _BIT_POSITIONS_GRID  # Shape: (4, 13)
    
    # Extract all bits using bit shifts (no conditionals)
    # For low word: shift cardset[0] right by bit_pos, mask with 1
    low_bits = (cardset[0] >> bit_positions) & 1  # Shape: (4, 13)
    
    # For high word: shift cardset[1] right by (bit_pos - 32), mask with 1
    # When bit_pos < 32, this gives us bits from very high positions (which are 0)
    high_bits = (cardset[1] >> (bit_positions - 32)) & 1  # Shape: (4, 13)
    
    # Select appropriate bits: use high_bits when bit_pos >= 32, else use low_bits
    # This is equivalent to: bit_pos >= 32 ? high_bits : low_bits
    valid_bits = jnp.where(bit_positions >= 32, high_bits, low_bits)  # Shape: (4, 13)
    
    # Convert to rank patterns using pre-computed powers
    suit_patterns = jnp.sum(valid_bits * _RANK_POWERS_GRID, axis=1, dtype=jnp.uint16)
    
    return suit_patterns

@jax.jit
def cards_to_suit_patterns(cards: jnp.ndarray) -> jnp.ndarray:
    """
    Fully vectorized conversion of card IDs to C-style bySuit representation.
    This matches the C Cardset.bySuit[4] format exactly.
    
    Args:
        cards: Array of card IDs (0-51), padded with -1 for invalid cards
        
    Returns:
        Array of shape (4,) representing bySuit[4] - 13-bit patterns for each suit
    """
    # Filter valid cards
    valid_mask = cards >= 0
    valid_cards = jnp.where(valid_mask, cards, 0)
    
    # Convert to suits and ranks vectorized
    suits = valid_cards // 13
    ranks = valid_cards % 13
    
    # Create bit patterns for each valid card: 2^rank, only for valid cards
    bit_patterns = jnp.where(valid_mask, jnp.uint16(1) << ranks, jnp.uint16(0))
    
    # Fully vectorized approach using broadcasting
    # Create a 4x7 matrix where each row represents a suit and columns are card positions
    suit_indicators = jnp.arange(4)[:, None] == suits[None, :]  # Shape: (4, num_cards)
    valid_indicators = valid_mask[None, :] & suit_indicators     # Shape: (4, num_cards)
    
    # Sum bit patterns for each suit
    suit_patterns = jnp.sum(
        jnp.where(valid_indicators, bit_patterns[None, :], jnp.uint16(0)), 
        axis=1, 
        dtype=jnp.uint16
    )
    
    return suit_patterns

@jax.jit
def count_suit_cards(cardset: jnp.ndarray) -> jnp.ndarray:
    """
    Count number of cards in each suit.
    
    Args:
        cardset: uint32[2] cardset array
        
    Returns:
        Array of 4 integers with card counts per suit
    """
    suit_counts = jnp.zeros(4, dtype=jnp.int32)
    
    for suit in range(4):
        count = jnp.int32(0)
        for rank in range(13):
            bit_pos = (suit << 4) + rank
            # Use uint32[2] operations
            bit_set = get_bit(cardset, bit_pos)
            count += jnp.int32(bit_set)
        suit_counts = suit_counts.at[suit].set(count)
    
    return suit_counts

@jax.jit
def get_rank_counts(cardset: jnp.ndarray) -> jnp.ndarray:
    """
    Count occurrences of each rank across all suits.
    
    Args:
        cardset: uint32[2] cardset array
        
    Returns:
        Array of 13 integers with count of each rank
    """
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    
    for rank in range(13):
        count = jnp.int32(0)
        for suit in range(4):
            bit_pos = (suit << 4) + rank
            # Use uint32[2] operations
            bit_set = get_bit(cardset, bit_pos)
            count += jnp.int32(bit_set)
        rank_counts = rank_counts.at[rank].set(count)
    
    return rank_counts

def hand_class(strength: int) -> int:
    """
    Get hand class (0-8) from hand strength value using C ACPC thresholds.
    
    Args:
        strength: Hand strength value from evaluator
        
    Returns:
        Hand class: 0=high card, 1=pair, ..., 8=straight flush
    """
    # Use ACPC hand class thresholds from constants.py
    from .tables.constants import (HANDCLASS_SINGLE_CARD, HANDCLASS_PAIR, HANDCLASS_TWO_PAIR,
                                   HANDCLASS_TRIPS, HANDCLASS_STRAIGHT, HANDCLASS_FLUSH,
                                   HANDCLASS_FULL_HOUSE, HANDCLASS_QUADS, HANDCLASS_STRAIGHT_FLUSH)
    
    if strength >= HANDCLASS_STRAIGHT_FLUSH:
        return 8  # Straight flush
    elif strength >= HANDCLASS_QUADS:
        return 7  # Four of a kind
    elif strength >= HANDCLASS_FULL_HOUSE:
        return 6  # Full house
    elif strength >= HANDCLASS_FLUSH:
        return 5  # Flush
    elif strength >= HANDCLASS_STRAIGHT:
        return 4  # Straight
    elif strength >= HANDCLASS_TRIPS:
        return 3  # Three of a kind
    elif strength >= HANDCLASS_TWO_PAIR:
        return 2  # Two pair
    elif strength >= HANDCLASS_PAIR:
        return 1  # One pair
    else:
        return 0  # High card

def hand_description(strength: int) -> str:
    """
    Get human-readable description of hand.
    
    Args:
        strength: Hand strength value from evaluator
        
    Returns:
        String description of hand type
    """
    hand_classes = {
        0: "High Card",
        1: "Pair", 
        2: "Two Pair",
        3: "Three of a Kind",
        4: "Straight",
        5: "Flush", 
        6: "Full House",
        7: "Four of a Kind",
        8: "Straight Flush"
    }
    class_id = hand_class(strength)
    return hand_classes.get(class_id, "Unknown")

def format_card(card_id: int) -> str:
    """
    Format card ID as human-readable string.
    
    Args:
        card_id: Card ID (0-51)
        
    Returns:
        String like "As" (Ace of spades) or "2c" (2 of clubs)
    """
    suit, rank = id_to_card(card_id)
    
    rank_chars = "23456789TJQKA"
    suit_chars = "cdhs"
    
    return rank_chars[rank] + suit_chars[suit]

def parse_card(card_str: str) -> int:
    """
    Parse card string to card ID.
    
    Args:
        card_str: String like "As" or "2c"
        
    Returns:
        Card ID (0-51)
    """
    if len(card_str) != 2:
        raise ValueError(f"Invalid card string: {card_str}")
    
    rank_chars = "23456789TJQKA"
    suit_chars = "cdhs"
    
    rank_char, suit_char = card_str[0].upper(), card_str[1].lower()
    
    try:
        rank = rank_chars.index(rank_char)
        suit = suit_chars.index(suit_char)
        return card_to_id(suit, rank)
    except ValueError:
        raise ValueError(f"Invalid card string: {card_str}")

def format_hand(cards: List[int]) -> str:
    """
    Format list of card IDs as readable string.
    
    Args:
        cards: List of card IDs
        
    Returns:
        String like "As Kh Qd Jc Ts"
    """
    return " ".join(format_card(card) for card in cards)

@jax.jit
def cardset_to_cards(cardset: jnp.ndarray) -> jnp.ndarray:
    """
    Convert cardset back to array of card IDs.
    
    Args:
        cardset: uint32[2] cardset array
        
    Returns:
        Array of card IDs, padded with -1 for empty slots
        Always returns MAX_CARDS_IN_CARDSET cards
    """
    cards = jnp.full(MAX_CARDS_IN_CARDSET, -1, dtype=jnp.int32)
    card_count = jnp.int32(0)
    
    # Check each possible card (0-51)
    for card_id in range(52):
        bit_pos = (card_id // 13) * 16 + (card_id % 13)  # ACPC bit position
        has_card = get_bit(cardset, bit_pos)
        
        # If card is present and we have space, add it
        new_count = card_count + jnp.int32(has_card)
        cards = jnp.where(
            (has_card > 0) & (card_count < MAX_CARDS_IN_CARDSET),
            cards.at[card_count].set(card_id),
            cards
        )
        card_count = jnp.where(new_count <= MAX_CARDS_IN_CARDSET, new_count, card_count)
    
    return cards

@jax.jit
def add_card_to_cardset(cardset: jnp.ndarray, card_id: int) -> jnp.ndarray:
    """
    Add a single card to a cardset.
    
    Args:
        cardset: Existing uint32[2] cardset array
        card_id: Card ID to add (0-51)
        
    Returns:
        Updated cardset with card added
    """
    # ACPC format: bit position = (suit << 4) + rank
    suit = card_id // 13
    rank = card_id % 13
    bit_pos = (suit << 4) + rank
    
    return set_bit(cardset, bit_pos)


# ============================================================================
# COMPATIBILITY AND CONVENIENCE FUNCTIONS
# ============================================================================

# Note: create_empty_cardset is now imported directly from cardset_ops


# Note: cardset_or is now imported directly from cardset_ops


# Note: cardset_and is now imported directly from cardset_ops


# Note: cardset_not is now imported directly from cardset_ops


def cardset_to_int(cardset: jnp.ndarray) -> int:
    """Convert uint32[2] cardset to Python int for debugging/comparison."""
    return cardset_to_uint64(cardset)


def cardset_from_int(value: int) -> jnp.ndarray:
    """Convert Python int to uint32[2] cardset."""
    return create_cardset_from_value(value)
