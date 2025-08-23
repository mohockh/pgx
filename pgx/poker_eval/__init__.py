"""
Fast poker hand evaluation library using precomputed lookup tables.

Based on the ACPC (Annual Computer Poker Competition) hand evaluation system
for maximum speed and accuracy.
"""

from .evaluator import evaluate_hand
from .cardset import card_to_id, cards_to_cardset, hand_class, hand_description

__all__ = [
    'evaluate_hand',
    'card_to_id',
    'cards_to_cardset',
    'hand_class',
    'hand_description'
]