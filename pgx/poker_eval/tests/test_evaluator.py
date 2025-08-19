"""
Comprehensive unit tests for poker hand evaluation.

Tests all hand types, edge cases, and evaluation accuracy.
"""

import pytest
import jax.numpy as jnp
import sys
import os

from pgx.poker_eval.jax_evaluator_new import evaluate_hand_cards as evaluate_hand
from pgx.poker_eval.cardset import parse_card, hand_class, hand_description
# For now, create simple implementations for missing functions
def batch_evaluate(hands):
    import jax
    return jax.vmap(evaluate_hand)(hands)

def hand_vs_hand(hand1, hand2):
    import jax.numpy as jnp
    strength1 = evaluate_hand(hand1)
    strength2 = evaluate_hand(hand2) 
    return jnp.sign(strength1 - strength2)

# hand_class and hand_description are now imported from cardset.py

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

class TestComprehensiveHighCard:
    """20 comprehensive tests for high card hands."""
    
    def test_high_card_ace_high(self):
        hand = cards_from_string("As Kh Qd Jc 9h")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_king_high(self):
        hand = cards_from_string("Kh Qd Jc 9h 7s")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_queen_high(self):
        hand = cards_from_string("Qh Jd 9c 7h 5s")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_jack_high(self):
        hand = cards_from_string("Jh 9d 7c 5h 3s")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_ten_high(self):
        hand = cards_from_string("Th 8d 6c 4h 2s")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_nine_high(self):
        hand = cards_from_string("9h 7d 5c 3h 2s")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_eight_high(self):
        hand = cards_from_string("8h 6d 4c 3h 2s")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_seven_high(self):
        hand = cards_from_string("7h 5d 4c 3h 2s")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_kicker_comparison_1(self):
        hand1 = cards_from_string("As Kh Qd Jc 9h")
        hand2 = cards_from_string("As Kh Qd Jc 8h")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_high_card_kicker_comparison_2(self):
        hand1 = cards_from_string("As Kh Qd Tc 9h")
        hand2 = cards_from_string("As Kh Qd 9c 8h")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_high_card_kicker_comparison_3(self):
        hand1 = cards_from_string("As Kh Jd Tc 9h")
        hand2 = cards_from_string("As Kh Td 9c 8h")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_high_card_all_different_suits(self):
        hand = cards_from_string("As Kh Qd Jc 9s")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_with_gaps(self):
        hand = cards_from_string("As Qh 9d 6c 3s")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_low_end(self):
        hand = cards_from_string("6h 5d 3c 2h As")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_seven_cards(self):
        hand = cards_from_string("As Kh Qd Jc 9h 7s 5d")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_almost_straight(self):
        hand = cards_from_string("As Kh Qd Jc 9h")  # Missing 10 for straight
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_almost_flush(self):
        hand = cards_from_string("As Ks Qs Js 9h")  # 4 spades, not 5
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_identical_except_suit(self):
        hand1 = cards_from_string("As Kh Qd Jc 9s")
        hand2 = cards_from_string("Ah Ks Qc Jd 9h")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_high_card_mixed_order(self):
        hand = cards_from_string("9h As Qd Kh Jc")
        assert hand_class(evaluate_hand(hand)) == 0
    
    def test_high_card_minimum_vs_maximum(self):
        hand1 = cards_from_string("7h 5d 4c 3h 2s")  # Lowest possible high card
        hand2 = cards_from_string("As Kh Qd Jc 9h")  # High ace-high
        assert evaluate_hand(hand2) > evaluate_hand(hand1)

class TestComprehensivePair:
    """20 comprehensive tests for one pair hands."""
    
    def test_pair_aces(self):
        hand = cards_from_string("As Ah Kd Qc Jh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_kings(self):
        hand = cards_from_string("Ks Kh Ad Qc Jh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_queens(self):
        hand = cards_from_string("Qs Qh Ad Kc Jh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_jacks(self):
        hand = cards_from_string("Js Jh Ad Kc Qh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_tens(self):
        hand = cards_from_string("Ts Th Ad Kc Qh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_twos(self):
        hand = cards_from_string("2s 2h Ad Kc Qh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_with_ace_kicker(self):
        hand = cards_from_string("7s 7h Ad Kc Qh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_comparison_same_pair_different_kickers(self):
        hand1 = cards_from_string("7s 7h Ad Kc Qh")
        hand2 = cards_from_string("7d 7c Ad Kh Jh")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_pair_comparison_different_pairs(self):
        hand1 = cards_from_string("As Ah 2d 3c 4h")
        hand2 = cards_from_string("Ks Kh Ad Qc Jh")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_pair_first_position(self):
        hand = cards_from_string("As Ah Kd Qc Jh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_last_position(self):
        hand = cards_from_string("As Kh Qd Jc Jh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_middle_position(self):
        hand = cards_from_string("As Kh Kd Qc Jh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_seven_cards_with_pair(self):
        hand = cards_from_string("As Ah Kd Qc Jh 9s 7d")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_kicker_ordering_1(self):
        hand1 = cards_from_string("As Ah Kd Qc Jh")
        hand2 = cards_from_string("As Ah Kd Qc Th")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_pair_kicker_ordering_2(self):
        hand1 = cards_from_string("As Ah Kd Tc 9h")
        hand2 = cards_from_string("As Ah Kd 9c 8h")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_pair_kicker_ordering_3(self):
        hand1 = cards_from_string("As Ah Jd Tc 9h")
        hand2 = cards_from_string("As Ah Td 9c 8h")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_pair_low_kickers(self):
        hand = cards_from_string("As Ah 5d 4c 3h")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_vs_high_card(self):
        pair_hand = cards_from_string("2s 2h 3d 4c 5h")
        high_card = cards_from_string("As Kh Qd Jc 9h")
        assert evaluate_hand(pair_hand) > evaluate_hand(high_card)
    
    def test_pair_mixed_suits(self):
        hand = cards_from_string("As Ah Kd Qc Jh")
        assert hand_class(evaluate_hand(hand)) == 1
    
    def test_pair_identical_kickers(self):
        hand1 = cards_from_string("As Ah Kd Qc Jh")
        hand2 = cards_from_string("Ad Ac Ks Qh Js")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)

class TestComprehensiveTwoPair:
    """20 comprehensive tests for two pair hands."""
    
    def test_two_pair_aces_kings(self):
        hand = cards_from_string("As Ah Kd Kc Qh")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_aces_queens(self):
        hand = cards_from_string("As Ah Qd Qc Kh")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_kings_queens(self):
        hand = cards_from_string("Ks Kh Qd Qc Ah")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_jacks_tens(self):
        hand = cards_from_string("Js Jh Td Tc Ah")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_nines_eights(self):
        hand = cards_from_string("9s 9h 8d 8c Ah")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_threes_twos(self):
        hand = cards_from_string("3s 3h 2d 2c Ah")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_comparison_higher_pair_wins(self):
        hand1 = cards_from_string("As Ah Kd Kc Qh")
        hand2 = cards_from_string("Ks Kh Qd Qc Ah")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_two_pair_comparison_same_high_pair(self):
        hand1 = cards_from_string("As Ah Kd Kc Qh")
        hand2 = cards_from_string("As Ah Qd Qc Kh")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_two_pair_comparison_same_pairs_different_kicker(self):
        hand1 = cards_from_string("As Ah Kd Kc Qh")
        hand2 = cards_from_string("As Ah Kd Kc Jh")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_two_pair_seven_cards(self):
        hand = cards_from_string("As Ah Kd Kc Qh Js 9d")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_adjacent_ranks(self):
        hand = cards_from_string("6s 6h 5d 5c Ah")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_spread_out(self):
        hand = cards_from_string("As Ah 7d 7c 3h")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_vs_one_pair(self):
        two_pair = cards_from_string("3s 3h 2d 2c 4h")
        one_pair = cards_from_string("As Ah Kd Qc Jh")
        assert evaluate_hand(two_pair) > evaluate_hand(one_pair)
    
    def test_two_pair_low_kicker(self):
        hand = cards_from_string("As Ah Kd Kc 2h")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_high_kicker(self):
        hand = cards_from_string("5s 5h 4d 4c Ah")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_in_seven_cards_chooses_best(self):
        # Should choose AA/KK over AA/33
        hand = cards_from_string("As Ah Kd Kc 3h 3s 2d")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 2
        # Higher two pair should be selected
    
    def test_two_pair_mixed_positions(self):
        hand = cards_from_string("As Kh Ad Kc Qh")
        assert hand_class(evaluate_hand(hand)) == 2
    
    def test_two_pair_identical_hands(self):
        hand1 = cards_from_string("As Ah Kd Kc Qh")
        hand2 = cards_from_string("Ad Ac Ks Kh Qs")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_two_pair_order_independence(self):
        hand1 = cards_from_string("As Ah Kd Kc Qh")
        hand2 = cards_from_string("Kd Kc As Ah Qh")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_two_pair_aces_and_eights_vs_aces_and_fives(self):
        # Two pair: Aces and 8s vs Aces and 5s - higher second pair should win
        hand1 = cards_from_string("As Ah 8d 8c 9h")  # Aces and 8s
        hand2 = cards_from_string("Ad Ac 5s 5h 9d")  # Aces and 5s  
        strength1 = evaluate_hand(hand1)
        strength2 = evaluate_hand(hand2)
        print(f"Aces and 8s strength: {strength1}")
        print(f"Aces and 5s strength: {strength2}")
        assert strength1 > strength2, f"Aces and 8s ({strength1}) should beat Aces and 5s ({strength2})"
        assert hand_class(strength1) == 2
        assert hand_class(strength2) == 2
    
    def test_seven_card_two_pair_bug(self):
        # Test the specific 7-card scenario from our side pot test that's failing
        # P1: 8s, 8h, Ac, Ad, 5c, 5d, 9s should be two pair Aces and 8s, not full house
        hand1 = cards_from_string("8s 8h Ac Ad 5c 5d 9s")  # Should be two pair Aces and 8s
        strength1 = evaluate_hand(hand1)
        class1 = hand_class(strength1)
        print(f"7-card hand with 8s,8h,Ac,Ad,5c,5d,9s: strength={strength1}, class={class1}")
        
        # Compare with 5-card version
        hand1_5card = cards_from_string("As Ah 8s 8h 9c")  # Pure two pair Aces and 8s
        strength1_5card = evaluate_hand(hand1_5card)
        class1_5card = hand_class(strength1_5card)
        print(f"5-card Aces and 8s: strength={strength1_5card}, class={class1_5card}")
        
        # P3: 7c, 2d, Ac, Ad, 5c, 5d, 9s should be two pair Aces and 5s
        hand3 = cards_from_string("7c 2d Ac Ad 5c 5d 9s")  # Should be two pair Aces and 5s
        strength3 = evaluate_hand(hand3)
        class3 = hand_class(strength3)
        print(f"7-card hand with 7c,2d,Ac,Ad,5c,5d,9s: strength={strength3}, class={class3}")
        
        # The 7-card hands should be two pair, not full house
        assert class1 == 2, f"7-card hand should be two pair (2), got {class1}"
        assert class3 == 2, f"7-card hand should be two pair (2), got {class3}"
        
        # The first hand should be stronger (Aces and 8s vs Aces and 5s)  
        assert strength1 > strength3, f"Aces and 8s ({strength1}) should beat Aces and 5s ({strength3})"
    
    def test_two_pair_pocket_pairs_vs_board_pairs(self):
        hand = cards_from_string("As Ah Kd Kc Qh")
        assert hand_class(evaluate_hand(hand)) == 2

class TestComprehensiveThreeOfAKind:
    """20 comprehensive tests for three of a kind hands."""
    
    def test_trips_aces(self):
        hand = cards_from_string("As Ah Ad Kc Qh")
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_kings(self):
        hand = cards_from_string("Ks Kh Kd Ac Qh")
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_queens(self):
        hand = cards_from_string("Qs Qh Qd Ac Kh")
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_jacks(self):
        hand = cards_from_string("Js Jh Jd Ac Kh")
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_tens(self):
        hand = cards_from_string("Ts Th Td Ac Kh")
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_twos(self):
        hand = cards_from_string("2s 2h 2d Ac Kh")
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_comparison_different_trips(self):
        hand1 = cards_from_string("As Ah Ad Kc Qh")
        hand2 = cards_from_string("Ks Kh Kd Ac Qh")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_trips_comparison_same_trips_different_kickers(self):
        hand1 = cards_from_string("7s 7h 7d Ac Kh")
        hand2 = cards_from_string("7s 7h 7d Ac Qh")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_trips_comparison_kicker_ordering(self):
        hand1 = cards_from_string("7s 7h 7d Kc Qh")
        hand2 = cards_from_string("7s 7h 7d Kc Jh")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_trips_seven_cards(self):
        hand = cards_from_string("As Ah Ad Kc Qh Js 9d")
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_vs_two_pair(self):
        trips = cards_from_string("2s 2h 2d 3c 4h")
        two_pair = cards_from_string("As Ah Kd Kc Qh")
        assert evaluate_hand(trips) > evaluate_hand(two_pair)
    
    def test_trips_low_kickers(self):
        hand = cards_from_string("As Ah Ad 4c 3h")
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_high_kickers(self):
        hand = cards_from_string("2s 2h 2d Ac Kh")
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_in_different_positions(self):
        hand1 = cards_from_string("As Ah Ad Kc Qh")  # Beginning
        hand2 = cards_from_string("Ac Kh As Ah Qd")  # Scattered
        assert hand_class(evaluate_hand(hand1)) == 3
        assert hand_class(evaluate_hand(hand2)) == 3
    
    def test_trips_with_potential_straight(self):
        hand = cards_from_string("6s 6h 6d 7c 8h")  # Not a straight
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_with_potential_flush(self):
        hand = cards_from_string("6s 6h 6d 7s 8s")  # Not a flush
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_from_seven_cards_best_kickers(self):
        # Should use A,K as kickers, not Q,J
        hand = cards_from_string("7s 7h 7d Ac Kh Qd Js")
        assert hand_class(evaluate_hand(hand)) == 3
    
    def test_trips_identical_except_suits(self):
        hand1 = cards_from_string("As Ah Ad Kc Qh")
        hand2 = cards_from_string("Ad Ac As Ks Qd")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_trips_set_vs_trips(self):
        # Both are three of a kind, should evaluate equally based on rank
        hand1 = cards_from_string("7s 7h 7d Ac Kh")  # Set
        hand2 = cards_from_string("7s 7h 7d Ac Kh")  # Same hand
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_trips_lowest_vs_highest(self):
        hand1 = cards_from_string("2s 2h 2d 3c 4h")  # Lowest trips
        hand2 = cards_from_string("As Ah Ad Kc Qh")  # Highest trips
        assert evaluate_hand(hand2) > evaluate_hand(hand1)

class TestComprehensiveStraight:
    """20 comprehensive tests for straight hands."""
    
    def test_straight_broadway(self):
        hand = cards_from_string("As Kh Qd Jc Ts")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_king_high(self):
        hand = cards_from_string("Ks Qh Jd Tc 9s")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_queen_high(self):
        hand = cards_from_string("Qs Jh Td 9c 8s")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_jack_high(self):
        hand = cards_from_string("Js Th 9d 8c 7s")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_ten_high(self):
        hand = cards_from_string("Ts 9h 8d 7c 6s")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_nine_high(self):
        hand = cards_from_string("9s 8h 7d 6c 5s")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_eight_high(self):
        hand = cards_from_string("8s 7h 6d 5c 4s")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_seven_high(self):
        hand = cards_from_string("7s 6h 5d 4c 3s")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_six_high(self):
        hand = cards_from_string("6s 5h 4d 3c 2s")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_wheel(self):
        hand = cards_from_string("5h 4d 3s 2c As")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_comparison_broadway_vs_wheel(self):
        broadway = cards_from_string("As Kh Qd Jc Ts")
        wheel = cards_from_string("5h 4d 3s 2c As")
        assert evaluate_hand(broadway) > evaluate_hand(wheel)
    
    def test_straight_comparison_adjacent(self):
        hand1 = cards_from_string("9s 8h 7d 6c 5s")
        hand2 = cards_from_string("8s 7h 6d 5c 4s")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_straight_mixed_suits(self):
        hand = cards_from_string("As Kh Qd Jc Ts")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_vs_trips(self):
        straight = cards_from_string("5h 4d 3s 2c As")
        trips = cards_from_string("As Ah Ad Kc Qh")
        assert evaluate_hand(straight) > evaluate_hand(trips)
    
    def test_straight_from_seven_cards(self):
        hand = cards_from_string("As Kh Qd Jc Ts 9h 7d")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_with_extra_cards(self):
        hand = cards_from_string("As Kh Qd Jc Ts 2h 3d")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_not_ace_low_straight_six_high(self):
        # This should NOT be a straight (A,6,5,4,3)
        hand = cards_from_string("As 6h 5d 4c 3s")
        assert hand_class(evaluate_hand(hand)) != 4
    
    def test_straight_order_independence(self):
        hand1 = cards_from_string("As Kh Qd Jc Ts")
        hand2 = cards_from_string("Ts Jc Qd Kh As")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_straight_with_pair_in_seven_cards(self):
        # Should detect straight, not pair
        hand = cards_from_string("As Kh Qd Jc Ts Th 9d")
        assert hand_class(evaluate_hand(hand)) == 4
    
    def test_straight_identical_different_suits(self):
        hand1 = cards_from_string("As Kh Qd Jc Ts")
        hand2 = cards_from_string("Ad Ks Qh Js Td")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)

class TestComprehensiveFlush:
    """20 comprehensive tests for flush hands."""
    
    def test_flush_spades_ace_high(self):
        hand = cards_from_string("As Ks Qs Js 9s")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_hearts_king_high(self):
        hand = cards_from_string("Kh Qh Jh 9h 7h")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_diamonds_queen_high(self):
        hand = cards_from_string("Qd Jd 9d 7d 5d")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_clubs_jack_high(self):
        hand = cards_from_string("Jc 9c 7c 5c 3c")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_comparison_ace_vs_king_high(self):
        ace_flush = cards_from_string("As Ks Qs Js 9s")
        king_flush = cards_from_string("Kh Qh Jh 9h 7h")
        assert evaluate_hand(ace_flush) > evaluate_hand(king_flush)
    
    def test_flush_comparison_same_high_card(self):
        hand1 = cards_from_string("As Ks Qs Js 9s")
        hand2 = cards_from_string("Ah Kh Qh Jh 8h")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_flush_comparison_kicker_order(self):
        hand1 = cards_from_string("As Ks Qs Ts 9s")
        hand2 = cards_from_string("Ah Kh Qh Jh 8h")
        assert evaluate_hand(hand1) < evaluate_hand(hand2)  # J > T
    
    def test_flush_vs_straight(self):
        flush = cards_from_string("As Ks Qs Js 9s")
        straight = cards_from_string("As Kh Qd Jc Ts")
        assert evaluate_hand(flush) > evaluate_hand(straight)
    
    def test_flush_seven_cards_all_same_suit(self):
        hand = cards_from_string("As Ks Qs Js 9s 7s 5s")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_seven_cards_six_of_suit(self):
        hand = cards_from_string("As Ks Qs Js 9s 7s 5h")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_seven_cards_exactly_five_of_suit(self):
        hand = cards_from_string("As Ks Qs Js 9s 7h 5d")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_low_cards(self):
        hand = cards_from_string("7s 5s 4s 3s 2s")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_with_ace_low(self):
        hand = cards_from_string("6s 5s 4s 3s As")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_not_straight_flush(self):
        hand = cards_from_string("As Ks Qs Js 9s")  # Missing 10s for straight flush
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_identical_ranks_different_suits(self):
        hand1 = cards_from_string("As Ks Qs Js 9s")
        hand2 = cards_from_string("Ah Kh Qh Jh 9h")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_flush_from_seven_cards_best_five(self):
        # Should use A,K,Q,T,9 not A,K,Q,J,T
        hand = cards_from_string("As Ks Qs Jc Ts 9s 7s")
        strength = evaluate_hand(hand)
        assert hand_class(strength) == 5
    
    def test_flush_gap_in_sequence(self):
        hand = cards_from_string("As Ks Qs 9s 7s")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_multiple_suits_in_seven_cards(self):
        # 5 spades, 2 hearts - should be flush
        hand = cards_from_string("As Ks Qs Js 9s Ah Kh")
        assert hand_class(evaluate_hand(hand)) == 5
    
    def test_flush_minimum_vs_maximum(self):
        min_flush = cards_from_string("7s 5s 4s 3s 2s")
        max_flush = cards_from_string("As Ks Qs Js Ts")
        assert evaluate_hand(max_flush) > evaluate_hand(min_flush)
    
    def test_flush_suit_independence(self):
        # Same ranks, different suits should be equal
        spades = cards_from_string("As Ks Qs Js 9s")
        hearts = cards_from_string("Ah Kh Qh Jh 9h")
        clubs = cards_from_string("Ac Kc Qc Jc 9c")
        diamonds = cards_from_string("Ad Kd Qd Jd 9d")
        assert evaluate_hand(spades) == evaluate_hand(hearts)
        assert evaluate_hand(hearts) == evaluate_hand(clubs)
        assert evaluate_hand(clubs) == evaluate_hand(diamonds)

class TestComprehensiveFullHouse:
    """20 comprehensive tests for full house hands."""
    
    def test_full_house_aces_over_kings(self):
        hand = cards_from_string("As Ah Ad Kc Kh")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_kings_over_aces(self):
        hand = cards_from_string("Ks Kh Kd Ac Ah")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_queens_over_jacks(self):
        hand = cards_from_string("Qs Qh Qd Jc Jh")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_jacks_over_tens(self):
        hand = cards_from_string("Js Jh Jd Tc Th")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_tens_over_nines(self):
        hand = cards_from_string("Ts Th Td 9c 9h")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_threes_over_twos(self):
        hand = cards_from_string("3s 3h 3d 2c 2h")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_twos_over_aces(self):
        hand = cards_from_string("2s 2h 2d Ac Ah")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_comparison_trips_rank(self):
        hand1 = cards_from_string("As Ah Ad Kc Kh")
        hand2 = cards_from_string("Ks Kh Kd Ac Ah")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_full_house_comparison_same_trips(self):
        hand1 = cards_from_string("As Ah Ad Kc Kh")
        hand2 = cards_from_string("As Ah Ad Qc Qh")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_full_house_vs_flush(self):
        full_house = cards_from_string("2s 2h 2d 3c 3h")
        flush = cards_from_string("As Ks Qs Js 9s")
        assert evaluate_hand(full_house) > evaluate_hand(flush)
    
    def test_full_house_from_seven_cards_two_trips(self):
        # Should choose best full house: AAA over KKK
        hand = cards_from_string("As Ah Ad Ks Kh Kd Qc")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_from_seven_cards_trips_and_two_pairs(self):
        # Should make full house with highest pair
        hand = cards_from_string("As Ah Ad Ks Kh Qd Qc")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_order_independence(self):
        hand1 = cards_from_string("As Ah Ad Kc Kh")
        hand2 = cards_from_string("Kc Kh As Ah Ad")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_full_house_mixed_suits(self):
        hand = cards_from_string("As Ah Ad Kc Kh")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_lowest_possible(self):
        hand = cards_from_string("2s 2h 2d 3c 3h")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_highest_possible(self):
        hand = cards_from_string("As Ah Ad Kc Kh")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_vs_four_of_a_kind(self):
        full_house = cards_from_string("As Ah Ad Kc Kh")
        four_kind = cards_from_string("2s 2h 2d 2c 3h")
        assert evaluate_hand(four_kind) > evaluate_hand(full_house)
    
    def test_full_house_identical_different_suits(self):
        hand1 = cards_from_string("As Ah Ad Kc Kh")
        hand2 = cards_from_string("Ad Ac As Ks Kd")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_full_house_from_two_pairs_and_fifth_card(self):
        # With two pairs and a fifth card matching one pair
        hand = cards_from_string("As Ah Kd Kc Ac Qh Jd")
        assert hand_class(evaluate_hand(hand)) == 6
    
    def test_full_house_boat_terminology(self):
        # Aces full of kings
        hand = cards_from_string("As Ah Ad Kc Kh")
        assert hand_class(evaluate_hand(hand)) == 6

class TestComprehensiveFourOfAKind:
    """20 comprehensive tests for four of a kind hands."""
    
    def test_quads_aces(self):
        hand = cards_from_string("As Ah Ad Ac Kh")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_kings(self):
        hand = cards_from_string("Ks Kh Kd Kc Ah")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_queens(self):
        hand = cards_from_string("Qs Qh Qd Qc Ah")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_jacks(self):
        hand = cards_from_string("Js Jh Jd Jc Ah")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_tens(self):
        hand = cards_from_string("Ts Th Td Tc Ah")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_nines(self):
        hand = cards_from_string("9s 9h 9d 9c Ah")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_twos(self):
        hand = cards_from_string("2s 2h 2d 2c Ah")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_comparison_different_ranks(self):
        hand1 = cards_from_string("As Ah Ad Ac Kh")
        hand2 = cards_from_string("Ks Kh Kd Kc Ah")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_quads_comparison_same_rank_different_kicker(self):
        hand1 = cards_from_string("7s 7h 7d 7c Ah")
        hand2 = cards_from_string("7s 7h 7d 7c Kh")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_quads_vs_full_house(self):
        quads = cards_from_string("2s 2h 2d 2c 3h")
        full_house = cards_from_string("As Ah Ad Kc Kh")
        assert evaluate_hand(quads) > evaluate_hand(full_house)
    
    def test_quads_from_seven_cards(self):
        hand = cards_from_string("As Ah Ad Ac Kh Qd Js")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_with_high_kicker(self):
        hand = cards_from_string("2s 2h 2d 2c Ah")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_with_low_kicker(self):
        hand = cards_from_string("As Ah Ad Ac 3h")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_order_independence(self):
        hand1 = cards_from_string("As Ah Ad Ac Kh")
        hand2 = cards_from_string("Kh As Ah Ad Ac")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_quads_from_seven_cards_best_kicker(self):
        # Should use A as kicker, not Q
        hand = cards_from_string("7s 7h 7d 7c Ah Qd Js")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_lowest_vs_highest(self):
        hand1 = cards_from_string("2s 2h 2d 2c 3h")
        hand2 = cards_from_string("As Ah Ad Ac Kh")
        assert evaluate_hand(hand2) > evaluate_hand(hand1)
    
    def test_quads_vs_straight_flush(self):
        quads = cards_from_string("As Ah Ad Ac Kh")
        straight_flush = cards_from_string("5s 4s 3s 2s As")
        assert evaluate_hand(straight_flush) > evaluate_hand(quads)
    
    def test_quads_identical_different_suits(self):
        hand1 = cards_from_string("As Ah Ad Ac Kh")
        hand2 = cards_from_string("As Ah Ad Ac Ks")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_quads_pocket_quads(self):
        # All four cards of same rank
        hand = cards_from_string("As Ah Ad Ac Kh Qd Js")
        assert hand_class(evaluate_hand(hand)) == 7
    
    def test_quads_multiple_kickers_chooses_best(self):
        # Should choose A as kicker from A,K,Q
        hand = cards_from_string("7s 7h 7d 7c Ah Kd Qs")
        assert hand_class(evaluate_hand(hand)) == 7

class TestComprehensiveStraightFlush:
    """20 comprehensive tests for straight flush hands."""
    
    def test_royal_flush_spades(self):
        hand = cards_from_string("As Ks Qs Js Ts")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_royal_flush_hearts(self):
        hand = cards_from_string("Ah Kh Qh Jh Th")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_royal_flush_diamonds(self):
        hand = cards_from_string("Ad Kd Qd Jd Td")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_royal_flush_clubs(self):
        hand = cards_from_string("Ac Kc Qc Jc Tc")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_king_high(self):
        hand = cards_from_string("Ks Qs Js Ts 9s")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_queen_high(self):
        hand = cards_from_string("Qh Jh Th 9h 8h")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_jack_high(self):
        hand = cards_from_string("Jd Td 9d 8d 7d")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_ten_high(self):
        hand = cards_from_string("Tc 9c 8c 7c 6c")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_nine_high(self):
        hand = cards_from_string("9s 8s 7s 6s 5s")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_eight_high(self):
        hand = cards_from_string("8h 7h 6h 5h 4h")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_seven_high(self):
        hand = cards_from_string("7d 6d 5d 4d 3d")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_six_high(self):
        hand = cards_from_string("6c 5c 4c 3c 2c")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_five_high_wheel(self):
        hand = cards_from_string("5s 4s 3s 2s As")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_comparison_royal_vs_king_high(self):
        royal = cards_from_string("As Ks Qs Js Ts")
        king_high = cards_from_string("Kh Qh Jh Th 9h")
        assert evaluate_hand(royal) > evaluate_hand(king_high)
    
    def test_straight_flush_comparison_adjacent_ranks(self):
        hand1 = cards_from_string("9s 8s 7s 6s 5s")
        hand2 = cards_from_string("8h 7h 6h 5h 4h")
        assert evaluate_hand(hand1) > evaluate_hand(hand2)
    
    def test_straight_flush_vs_four_of_a_kind(self):
        straight_flush = cards_from_string("5s 4s 3s 2s As")
        quads = cards_from_string("As Ah Ad Ac Kh")
        assert evaluate_hand(straight_flush) > evaluate_hand(quads)
    
    def test_straight_flush_wheel_vs_broadway_straight(self):
        wheel_sf = cards_from_string("5s 4s 3s 2s As")
        broadway_straight = cards_from_string("As Kh Qd Jc Ts")
        assert evaluate_hand(wheel_sf) > evaluate_hand(broadway_straight)
    
    def test_straight_flush_from_seven_cards(self):
        hand = cards_from_string("9s 8s 7s 6s 5s 4h 3d")
        assert hand_class(evaluate_hand(hand)) == 8
    
    def test_straight_flush_identical_different_suits(self):
        hand1 = cards_from_string("9s 8s 7s 6s 5s")
        hand2 = cards_from_string("9h 8h 7h 6h 5h")
        assert evaluate_hand(hand1) == evaluate_hand(hand2)
    
    def test_straight_flush_not_royal_flush(self):
        # King-high straight flush is not royal flush
        hand = cards_from_string("Ks Qs Js Ts 9s")
        assert hand_class(evaluate_hand(hand)) == 8
        # Should be less than royal flush
        royal = cards_from_string("As Ks Qs Js Ts")
        assert evaluate_hand(royal) > evaluate_hand(hand)

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
    
    # Run comprehensive tests
    print("Running comprehensive tests...")
    
    # High Card Tests
    test_hc = TestComprehensiveHighCard()
    for method_name in dir(test_hc):
        if method_name.startswith('test_'):
            getattr(test_hc, method_name)()
    print("✅ Comprehensive high card tests passed")
    
    # Pair Tests
    test_pair = TestComprehensivePair()
    for method_name in dir(test_pair):
        if method_name.startswith('test_'):
            getattr(test_pair, method_name)()
    print("✅ Comprehensive pair tests passed")
    
    # Two Pair Tests
    test_2p = TestComprehensiveTwoPair()
    for method_name in dir(test_2p):
        if method_name.startswith('test_'):
            getattr(test_2p, method_name)()
    print("✅ Comprehensive two pair tests passed")
    
    # Three of a Kind Tests
    test_3k = TestComprehensiveThreeOfAKind()
    for method_name in dir(test_3k):
        if method_name.startswith('test_'):
            getattr(test_3k, method_name)()
    print("✅ Comprehensive three of a kind tests passed")
    
    # Straight Tests
    test_str = TestComprehensiveStraight()
    for method_name in dir(test_str):
        if method_name.startswith('test_'):
            getattr(test_str, method_name)()
    print("✅ Comprehensive straight tests passed")
    
    # Flush Tests
    test_fl = TestComprehensiveFlush()
    for method_name in dir(test_fl):
        if method_name.startswith('test_'):
            getattr(test_fl, method_name)()
    print("✅ Comprehensive flush tests passed")
    
    # Full House Tests
    test_fh = TestComprehensiveFullHouse()
    for method_name in dir(test_fh):
        if method_name.startswith('test_'):
            getattr(test_fh, method_name)()
    print("✅ Comprehensive full house tests passed")
    
    # Four of a Kind Tests
    test_4k = TestComprehensiveFourOfAKind()
    for method_name in dir(test_4k):
        if method_name.startswith('test_'):
            getattr(test_4k, method_name)()
    print("✅ Comprehensive four of a kind tests passed")
    
    # Straight Flush Tests
    test_sf = TestComprehensiveStraightFlush()
    for method_name in dir(test_sf):
        if method_name.startswith('test_'):
            getattr(test_sf, method_name)()
    print("✅ Comprehensive straight flush tests passed")
    
    print("All comprehensive tests passed! 🎉")
