"""
Card representation and conversion utilities for poker hand evaluation.

Provides functions to convert between different card representations and
work with the ACPC cardset format for fast evaluation.
"""

import jax
import jax.numpy as jnp
from typing import Union, List, Tuple

# Maximum number of cards that can be extracted from a cardset
MAX_CARDS_IN_CARDSET = 10

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
def cards_to_cardset(cards: jnp.ndarray) -> jnp.uint64:
    """
    Convert array of card IDs to ACPC cardset representation using vectorized operations.
    
    Args:
        cards: Array of card IDs
        
    Returns:
        64-bit cardset integer
    """
    # Filter out invalid cards (-1)
    valid_mask = cards >= 0
    valid_cards = jnp.where(valid_mask, cards, 0).astype(jnp.uint64)
    
    # Convert to suits and ranks (vectorized)
    suits = valid_cards // 13
    ranks = valid_cards % 13
    
    # Calculate bit positions: (suit << 4) + rank
    bit_positions = (suits << 4) + ranks
    
    # Create bit masks for each card
    bit_masks = jnp.where(valid_mask, jnp.uint64(1) << bit_positions, jnp.uint64(0))
    
    # OR all bit masks together to create final cardset
    cardset = jnp.bitwise_or.reduce(bit_masks)
    
    return cardset

@jax.jit
def extract_suit_ranks(cardset: int) -> jnp.ndarray:
    """
    Extract rank patterns for each suit from cardset.
    
    Args:
        cardset: 64-bit cardset integer
        
    Returns:
        Array of 4 integers, each with 13 bits representing ranks in that suit
    """
    suit_ranks = jnp.zeros(4, dtype=jnp.uint16)
    
    # Extract ranks for each suit using JAX operations
    for suit in range(4):
        ranks = 0
        for rank in range(13):
            bit_pos = (suit << 4) + rank
            # Use JAX operations instead of Python if
            bit_set = (cardset >> bit_pos) & 1
            ranks |= bit_set << rank
        suit_ranks = suit_ranks.at[suit].set(ranks)
    
    return suit_ranks

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
def count_suit_cards(cardset: int) -> jnp.ndarray:
    """
    Count number of cards in each suit.
    
    Args:
        cardset: 64-bit cardset integer
        
    Returns:
        Array of 4 integers with card counts per suit
    """
    suit_counts = jnp.zeros(4, dtype=jnp.int32)
    
    for suit in range(4):
        count = 0
        for rank in range(13):
            bit_pos = (suit << 4) + rank
            # Use JAX operations instead of Python if
            bit_set = (cardset >> bit_pos) & 1
            count += bit_set
        suit_counts = suit_counts.at[suit].set(count)
    
    return suit_counts

@jax.jit
def get_rank_counts(cardset: int) -> jnp.ndarray:
    """
    Count occurrences of each rank across all suits.
    
    Args:
        cardset: 64-bit cardset integer
        
    Returns:
        Array of 13 integers with count of each rank
    """
    rank_counts = jnp.zeros(13, dtype=jnp.int32)
    
    for rank in range(13):
        count = 0
        for suit in range(4):
            bit_pos = (suit << 4) + rank
            # Use JAX operations instead of Python if
            bit_set = (cardset >> bit_pos) & 1
            count += bit_set
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
def cardset_to_cards(cardset: jnp.uint64) -> jnp.ndarray:
    """
    Convert cardset back to array of card IDs.
    
    Args:
        cardset: 64-bit cardset integer
        
    Returns:
        Array of card IDs, padded with -1 for empty slots
        Always returns MAX_CARDS_IN_CARDSET cards
    """
    cards = jnp.full(MAX_CARDS_IN_CARDSET, -1, dtype=jnp.int32)
    card_count = 0
    
    # Check each possible card (0-51)
    for card_id in range(52):
        bit_pos = (card_id // 13) * 16 + (card_id % 13)  # ACPC bit position
        has_card = (cardset >> bit_pos) & 1
        
        # If card is present and we have space, add it
        new_count = card_count + has_card
        cards = jnp.where(
            (has_card > 0) & (card_count < MAX_CARDS_IN_CARDSET),
            cards.at[card_count].set(card_id),
            cards
        )
        card_count = jnp.where(new_count <= MAX_CARDS_IN_CARDSET, new_count, card_count)
    
    return cards

@jax.jit
def add_card_to_cardset(cardset: jnp.uint64, card_id: int) -> jnp.uint64:
    """
    Add a single card to a cardset.
    
    Args:
        cardset: Existing cardset
        card_id: Card ID to add (0-51)
        
    Returns:
        Updated cardset with card added
    """
    # ACPC format: bit position = (suit << 4) + rank
    suit = card_id // 13
    rank = card_id % 13
    bit_pos = (suit << 4) + rank
    
    return cardset | jnp.uint64(1 << bit_pos)
