"""
Card representation and conversion utilities for poker hand evaluation.

Provides functions to convert between different card representations and
work with the ACPC cardset format for fast evaluation.
"""

import jax
import jax.numpy as jnp
from typing import Union, List, Tuple
from .tables import get_hand_class, HAND_CLASSES

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
def cards_to_cardset(cards: jnp.ndarray) -> int:
    """
    Convert array of card IDs to ACPC cardset representation.
    
    Args:
        cards: Array of card IDs
        
    Returns:
        64-bit cardset integer
    """
    cardset = 0
    
    def add_card(carry, card_id):
        cardset, = carry
        # Convert card_id to suit and rank
        suit = card_id // 13
        rank = card_id % 13
        
        # ACPC format: bit position = (suit << 4) + rank
        bit_pos = (suit << 4) + rank
        new_cardset = cardset | (1 << bit_pos)
        
        return (new_cardset,), None
    
    (final_cardset,), _ = jax.lax.scan(add_card, (cardset,), cards)
    return final_cardset

@jax.jit
def extract_suit_ranks(cardset: int) -> jnp.ndarray:
    """
    Extract rank patterns for each suit from cardset.
    
    Args:
        cardset: 64-bit cardset integer
        
    Returns:
        Array of 4 integers, each with 13 bits representing ranks in that suit
    """
    suit_ranks = jnp.zeros(4, dtype=jnp.int32)
    
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
    Get hand class (0-8) from hand strength value.
    
    Args:
        strength: Hand strength value from evaluator
        
    Returns:
        Hand class: 0=high card, 1=pair, ..., 8=straight flush
    """
    return get_hand_class(strength)

def hand_description(strength: int) -> str:
    """
    Get human-readable description of hand.
    
    Args:
        strength: Hand strength value from evaluator
        
    Returns:
        String description of hand type
    """
    class_id = get_hand_class(strength)
    return HAND_CLASSES.get(class_id, "Unknown")

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