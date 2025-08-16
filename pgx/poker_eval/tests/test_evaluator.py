"""
Comprehensive unit tests for poker hand evaluation.

Tests all hand types, edge cases, and evaluation accuracy.
"""

import pytest
import jax.numpy as jnp
import sys
import os

from pgx.poker_eval.jax_evaluator import evaluate_hand_jax as evaluate_hand, hand_class_jax
from pgx.poker_eval.cardset import parse_card
# For now, create simple implementations for missing functions
def batch_evaluate(hands):
    import jax
    return jax.vmap(evaluate_hand)(hands)

def hand_vs_hand(hand1, hand2):
    import jax.numpy as jnp
    strength1 = evaluate_hand(hand1)
    strength2 = evaluate_hand(hand2) 
    return jnp.sign(strength1 - strength2)

# Use JAX version of hand_class
def hand_class(strength):
    return int(hand_class_jax(strength))  # Convert to Python int

# Create hand_description function for JAX evaluator
def hand_description(strength):
    class_names = {
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
    return class_names.get(hand_class(strength), "Unknown")
from pgx.poker_eval.tables import (
    HANDCLASS_HIGH_CARD, HANDCLASS_PAIR, HANDCLASS_TWO_PAIR,
    HANDCLASS_THREE_OF_A_KIND, HANDCLASS_STRAIGHT, HANDCLASS_FLUSH,
    HANDCLASS_FULL_HOUSE, HANDCLASS_FOUR_OF_A_KIND, HANDCLASS_STRAIGHT_FLUSH
)

def cards_from_string(cards_str: str) -> jnp.ndarray:
    """Convert card string like 'As Kh Qd Jc Ts' to card ID array."""
    card_strs = cards_str.split()
    card_ids = [parse_card(card) for card in card_strs]
    # Pad to 7 cards with -1 for JAX evaluator
    while len(card_ids) < 7:
        card_ids.append(-1)
    return jnp.array(card_ids[:7], dtype=jnp.int32)

class TestHandEvaluation:
    """Test basic hand evaluation functionality."""
    
    def test_royal_flush(self):
        """Test royal flush evaluation."""
        # Royal flush in spades
        hand = cards_from_string("As Ks Qs Js Ts")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 8  # Straight flush
        assert hand_description(strength) == "Straight Flush"
    
    def test_straight_flush(self):
        """Test straight flush evaluation."""
        # 5-high straight flush
        hand = cards_from_string("5h 4h 3h 2h Ah")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 8  # Straight flush
        
        # 9-high straight flush
        hand = cards_from_string("9s 8s 7s 6s 5s")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 8  # Straight flush
    
    def test_four_of_a_kind(self):
        """Test four of a kind evaluation."""
        hand = cards_from_string("As Ah Ad Ac Kh")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 7  # Four of a kind
        
        # Lower quads
        hand = cards_from_string("2s 2h 2d 2c 3h")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 7  # Four of a kind
    
    def test_full_house(self):
        """Test full house evaluation."""
        hand = cards_from_string("As Ah Ad Kc Kh")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 6  # Full house
        
        # Different full house
        hand = cards_from_string("3s 3h 3d 7c 7h")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 6  # Full house
    
    def test_flush(self):
        """Test flush evaluation."""
        hand = cards_from_string("As Ks 9s 7s 2s")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 5  # Flush
        
        # Different flush
        hand = cards_from_string("Qh Jh 8h 6h 4h")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 5  # Flush
    
    def test_straight(self):
        """Test straight evaluation."""
        # Broadway straight
        hand = cards_from_string("As Kh Qd Jc Ts")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 4  # Straight
        
        # Wheel (A-5 straight)
        hand = cards_from_string("5h 4d 3s 2c Ah")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 4  # Straight
        
        # Middle straight
        hand = cards_from_string("9h 8d 7s 6c 5h")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 4  # Straight
    
    def test_three_of_a_kind(self):
        """Test three of a kind evaluation."""
        hand = cards_from_string("As Ah Ad Kc Qh")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 3  # Three of a kind
        
        # Lower trips
        hand = cards_from_string("7s 7h 7d Ac Kh")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 3  # Three of a kind
    
    def test_two_pair(self):
        """Test two pair evaluation."""
        hand = cards_from_string("As Ah Kd Kc Qh")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 2  # Two pair
        
        # Lower two pair
        hand = cards_from_string("8s 8h 3d 3c 7h")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 2  # Two pair
    
    def test_one_pair(self):
        """Test one pair evaluation."""
        hand = cards_from_string("As Ah Kd Qc Jh")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 1  # One pair
        
        # Lower pair
        hand = cards_from_string("7s 7h Ad Qc Jh")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 1  # One pair
    
    def test_high_card(self):
        """Test high card evaluation."""
        hand = cards_from_string("As Kh Qd Jc 9h")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 0  # High card
        
        # Lower high card
        hand = cards_from_string("Qh Jd 9s 7c 5h")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 0  # High card

class TestHandComparison:
    """Test hand vs hand comparisons."""
    
    def test_same_class_comparison(self):
        """Test comparison within same hand class."""
        # Ace high flush vs King high flush
        hand1 = cards_from_string("As Ks 9s 7s 2s")
        hand2 = cards_from_string("Kh Qh 9h 7h 2h")
        
        result = hand_vs_hand(hand1, hand2)
        assert result > 0  # hand1 should win
        
        # Aces vs Kings
        hand1 = cards_from_string("As Ah Kd Qc Jh")
        hand2 = cards_from_string("Ks Kh Ad Qc Jh")
        
        result = hand_vs_hand(hand1, hand2)
        assert result > 0  # hand1 should win
    
    def test_different_class_comparison(self):
        """Test comparison between different hand classes."""
        # Pair vs high card
        hand1 = cards_from_string("2s 2h Ad Qc Jh")
        hand2 = cards_from_string("As Kh Qd Jc 9h")
        
        result = hand_vs_hand(hand1, hand2)
        assert result > 0  # pair should win
        
        # Full house vs flush
        hand1 = cards_from_string("As Ah Ad Kc Kh")
        hand2 = cards_from_string("As Ks 9s 7s 2s")
        
        result = hand_vs_hand(hand1, hand2)
        assert result > 0  # full house should win
    
    def test_tie_hands(self):
        """Test tied hands."""
        # Same exact hand
        hand1 = cards_from_string("As Kh Qd Jc Ts")
        hand2 = cards_from_string("As Kh Qd Jc Ts")
        
        result = hand_vs_hand(hand1, hand2)
        assert result == 0  # should tie

class TestSevenCardHands:
    """Test evaluation with 7-card hands (Texas Hold'em)."""
    
    def test_seven_card_flush(self):
        """Test flush from 7 cards."""
        # 7 cards with 5-card flush
        hand = cards_from_string("As Ks Qs Js 9s 2h 3d")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 5  # Should find flush
    
    def test_seven_card_straight(self):
        """Test straight from 7 cards."""
        # 7 cards with straight
        hand = cards_from_string("As Kh Qd Jc Ts 2h 3d")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 4  # Should find straight
    
    def test_seven_card_best_hand(self):
        """Test that best 5-card hand is found from 7."""
        # Should make full house, not just trips
        hand = cards_from_string("As Ah Ad Kc Kh 2s 3d")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 6  # Should find full house

class TestBatchEvaluation:
    """Test vectorized evaluation."""
    
    def test_batch_evaluation(self):
        """Test evaluating multiple hands at once."""
        hands = jnp.array([
            cards_from_string("As Ah Ad Kc Kh"),  # Full house
            cards_from_string("As Ks 9s 7s 2s"),  # Flush
            cards_from_string("As Ah Kd Qc Jh"),  # Pair
        ])
        
        strengths = batch_evaluate(hands)
        assert len(strengths) == 3
        
        # Full house should be strongest
        assert hand_class(strengths[0]) == 6
        assert hand_class(strengths[1]) == 5
        assert hand_class(strengths[2]) == 1
        
        # Order should be correct
        assert strengths[0] > strengths[1] > strengths[2]

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_wheel_straight(self):
        """Test A-5 straight (wheel)."""
        hand = cards_from_string("5h 4d 3s 2c Ah")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 4  # Should be straight
    
    def test_wheel_straight_flush(self):
        """Test A-5 straight flush."""
        hand = cards_from_string("5s 4s 3s 2s As")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 8  # Should be straight flush
    
    def test_minimum_cards(self):
        """Test with exactly 5 cards."""
        hand = cards_from_string("As Kh Qd Jc Ts")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 4  # Broadway straight
    
    def test_kicker_ordering(self):
        """Test that kickers are properly ordered."""
        # Same pair, different kickers
        hand1 = cards_from_string("As Ah Kd Qc Jh")
        hand2 = cards_from_string("As Ah Kd Qc 9h")
        
        strength1 = evaluate_hand(hand1)
        strength2 = evaluate_hand(hand2)
        
        assert strength1 > strength2  # Better kicker should win

if __name__ == "__main__":
    # Run basic smoke tests
    print("Running poker evaluator tests...")
    
    test_eval = TestHandEvaluation()
    test_eval.test_royal_flush()
    test_eval.test_four_of_a_kind()
    test_eval.test_full_house()
    test_eval.test_flush()
    test_eval.test_straight()
    print("✅ Basic hand evaluation tests passed")
    
    test_comp = TestHandComparison()
    test_comp.test_same_class_comparison()
    test_comp.test_different_class_comparison()
    print("✅ Hand comparison tests passed")
    
    test_batch = TestBatchEvaluation()
    test_batch.test_batch_evaluation()
    print("✅ Batch evaluation tests passed")
    
    print("All tests passed! 🎉")
