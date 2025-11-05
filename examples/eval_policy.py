#!/usr/bin/env python3
"""
Policy evaluation diagnostic tool for poker agents.

Visualizes starting hand strategy as a 13x13 heatmap showing fold/call/raise
probabilities for each starting hand in UTG position with 9 players.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import orbax.checkpoint
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path to import pgx
sys.path.insert(0, str(Path(__file__).parent.parent))

from pgx import universal_poker
from pgx.poker_eval.cardset import cards_to_cardset
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load a checkpoint saved with Orbax.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        Dictionary with "train_state", "chunk_idx", and "config"
    """
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(checkpoint_path)


@jax.jit
def cardset_to_onehot(cardset: jnp.ndarray) -> jnp.ndarray:
    """Convert uint32[2] cardset to one-hot (52,) encoding.

    Args:
        cardset: uint32[2] array representing a set of cards

    Returns:
        One-hot encoded array of shape (52,) with 1.0 at positions of cards in the set
    """
    onehot = jnp.zeros(52, dtype=jnp.float32)

    for card_id in range(52):
        # ACPC bit position: (suit << 4) + rank
        bit_pos = (card_id // 13) * 16 + (card_id % 13)

        word_idx = bit_pos // 32
        bit_in_word = bit_pos % 32

        is_set = (cardset[word_idx] >> bit_in_word) & jnp.uint32(1)
        onehot = onehot.at[card_id].set(is_set.astype(jnp.float32))

    return onehot


class ActorCritic(nn.Module):
    """Actor-Critic network for poker policy."""
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Handle both batched and unbatched inputs
        is_batched = x.ndim > 1

        if is_batched:
            hole_cardsets = x[:, :2].astype(jnp.uint32)
            visible_board_cardsets = x[:, 2:4].astype(jnp.uint32)
            other_features = x[:, 4:]

            hole_onehots = jax.vmap(cardset_to_onehot)(hole_cardsets)
            board_onehots = jax.vmap(cardset_to_onehot)(visible_board_cardsets)
        else:
            hole_cardset = x[:2].astype(jnp.uint32)
            visible_board_cardset = x[2:4].astype(jnp.uint32)
            other_features = x[4:]

            hole_onehots = cardset_to_onehot(hole_cardset)
            board_onehots = cardset_to_onehot(visible_board_cardset)

        # Embed the card representations
        hole_embed = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(hole_onehots)
        hole_embed = activation(hole_embed)

        board_embed = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(board_onehots)
        board_embed = activation(board_embed)

        # Concatenate embeddings with other features
        x = jnp.concatenate([hole_embed, board_embed, other_features], axis=-1)

        actor_mean = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean) + actor_mean
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            3, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic) + critic
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def card_id_to_name(card_id: int) -> str:
    """Convert card ID (0-51) to name (e.g., 'As', '2c')."""
    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suit_names = ['c', 'd', 'h', 's']
    rank = card_id % 13
    suit = card_id // 13
    return rank_names[rank] + suit_names[suit]


def make_starting_hand_obs(card1_id: int, card2_id: int, num_players: int = 8) -> jnp.ndarray:
    """Create observation vector for a starting hand preflop.

    Args:
        card1_id: First hole card ID (0-51)
        card2_id: Second hole card ID (0-51)
        num_players: Number of players in game

    Returns:
        Observation vector in format: [hole_cardset[2], board_cardset[2], pot, stack,
                                       bets[num_players], folded[num_players], round]
    """
    # Convert cards to cardset
    hole_cardset = cards_to_cardset(jnp.array([card1_id, card2_id], dtype=jnp.uint32))

    # Empty board cardset (preflop)
    board_cardset = jnp.zeros(2, dtype=jnp.uint32)

    # Game state: pot (two blinds = 3 chips), stack
    pot = jnp.uint32(3)
    stack = jnp.uint32(100)

    # Bets: first player has small blind (1), second has big blind (2)
    bets = jnp.zeros(num_players, dtype=jnp.uint32)
    bets = bets.at[0].set(jnp.uint32(1))  # Small blind
    bets = bets.at[1].set(jnp.uint32(2))  # Big blind

    # Folded status: all False
    folded = jnp.zeros(num_players, dtype=jnp.uint32)

    # Current round: 0 = preflop
    round_num = jnp.uint32(0)

    # Concatenate observation
    obs = jnp.concatenate([
        hole_cardset,           # [2]
        board_cardset,          # [2]
        jnp.array([pot, stack], dtype=jnp.uint32),  # [2]
        bets,                   # [num_players]
        folded,                 # [num_players]
        jnp.array([round_num], dtype=jnp.uint32),  # [1]
    ])

    return obs


def get_action_probabilities(network, params, obs: jnp.ndarray) -> Tuple[float, float, float]:
    """Get action probabilities from policy for an observation.

    Args:
        network: ActorCritic network instance
        params: Network parameters
        obs: Observation vector

    Returns:
        Tuple of (fold_prob, call_prob, raise_prob)
    """
    pi, _ = network.apply(params, obs)
    probs = pi.probs

    fold_prob = float(probs[0])
    call_prob = float(probs[1])
    raise_prob = float(jnp.sum(probs[2:]))  # Sum all raise actions

    return fold_prob, call_prob, raise_prob


def rank_to_name(rank: int) -> str:
    """Convert rank index (0-12) to name."""
    names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return names[rank]


def get_hand_name(rank1: int, rank2: int) -> str:
    """Get hand name for two ranks.

    Args:
        rank1: First card rank (0-12)
        rank2: Second card rank (0-12)

    Returns:
        Hand name like 'AA', 'AKs', 'AKo'
    """
    name1 = rank_to_name(rank1)
    name2 = rank_to_name(rank2)

    if rank1 == rank2:
        return name1 + name2  # Pocket pair
    elif rank1 > rank2:
        return name1 + name2 + 's'  # Suited (above diagonal)
    else:
        return name1 + name2 + 'o'  # Offsuit (below diagonal)


def draw_cell_with_actions(ax, x: float, y: float, width: float, height: float,
                            fold_prob: float, call_prob: float, raise_prob: float,
                            hand_name: str = ""):
    """Draw a cell with GTO Wizard-style action visualization.

    Uses horizontal bars and gradient coloring based on raise probability.

    Args:
        ax: Matplotlib axes
        x, y: Cell position
        width, height: Cell dimensions
        fold_prob, call_prob, raise_prob: Action probabilities (sum to 1)
        hand_name: Name of the hand (for text display)
    """
    # Normalize probabilities to ensure they sum to 1
    total = fold_prob + call_prob + raise_prob
    if total > 0:
        fold_prob /= total
        call_prob /= total
        raise_prob /= total

    # Determine background color based on raise probability (gradient)
    # Light gray for low raise%, dark red for high raise%
    if raise_prob > 0.8:
        bg_color = '#8B0000'  # Dark red
        text_color = 'white'
    elif raise_prob > 0.6:
        bg_color = '#B22222'  # Fire brick
        text_color = 'white'
    elif raise_prob > 0.4:
        bg_color = '#CD5C5C'  # Indian red
        text_color = 'white'
    elif raise_prob > 0.2:
        bg_color = '#F08080'  # Light coral
        text_color = 'black'
    else:
        bg_color = '#D3D3D3'  # Light gray
        text_color = 'black'

    # Draw background cell
    rect = patches.Rectangle((x, y), width, height,
                             linewidth=1, edgecolor='#333333', facecolor=bg_color)
    ax.add_patch(rect)

    # Draw horizontal action bars within cell
    bar_height = height / 3.2  # Leave some space between bars
    spacing = height / 3.5

    # FOLD bar (top) - Blue
    if fold_prob > 0:
        bar_y = y + height - bar_height - spacing * 0.5
        bar_width = width * fold_prob
        fold_rect = patches.Rectangle((x, bar_y), bar_width, bar_height,
                                      linewidth=0.5, edgecolor='#1a1a1a', facecolor='#4169E1')
        ax.add_patch(fold_rect)

    # CALL bar (middle) - Green
    if call_prob > 0:
        bar_y = y + height - bar_height * 2 - spacing * 1.5
        bar_width = width * call_prob
        call_rect = patches.Rectangle((x, bar_y), bar_width, bar_height,
                                      linewidth=0.5, edgecolor='#1a1a1a', facecolor='#32CD32')
        ax.add_patch(call_rect)

    # RAISE bar (bottom) - Red
    if raise_prob > 0:
        bar_y = y + height - bar_height * 3 - spacing * 2.5
        bar_width = width * raise_prob
        raise_rect = patches.Rectangle((x, bar_y), bar_width, bar_height,
                                       linewidth=0.5, edgecolor='#1a1a1a', facecolor='#FF6347')
        ax.add_patch(raise_rect)

    # Add hand name text in center
    if hand_name:
        text_x = x + width / 2
        text_y = y + height / 2 - spacing
        ax.text(text_x, text_y, hand_name, ha='center', va='center',
               fontsize=7, fontweight='bold', color=text_color, family='monospace')


def create_heatmap_visualization(hand_probs: Dict[Tuple[int, int], Tuple[float, float, float]],
                                 output_path: str = "examples/starting_hand_strategy.png"):
    """Create and save a 13x13 heatmap of starting hand strategy in GTO Wizard style.

    Args:
        hand_probs: Dictionary mapping (rank1, rank2) to (fold_prob, call_prob, raise_prob)
        output_path: Path to save PNG file
    """
    fig, ax = plt.subplots(figsize=(16, 14), facecolor='#1a1a1a')
    ax.set_facecolor('#2a2a2a')

    # Cell dimensions
    cell_width = 1.0
    cell_height = 1.0
    margin = 0.8

    # Draw the 13x13 grid
    for rank1 in range(13):
        for rank2 in range(13):
            if (rank1, rank2) in hand_probs:
                fold_prob, call_prob, raise_prob = hand_probs[(rank1, rank2)]
                hand_name = get_hand_name(rank1, rank2)

                # Position: x increases with rank2, y increases with rank1
                x = margin + rank2 * cell_width
                y = margin + (12 - rank1) * cell_height

                draw_cell_with_actions(ax, x, y, cell_width, cell_height,
                                      fold_prob, call_prob, raise_prob, hand_name)

    # Set axis properties
    ax.set_xlim(margin - 0.5, margin + 13 * cell_width + 0.5)
    ax.set_ylim(margin - 0.5, margin + 13 * cell_height + 0.5)
    ax.set_aspect('equal')

    # Set tick labels
    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    ax.set_xticks([margin + i * cell_width + cell_width / 2 for i in range(13)])
    ax.set_xticklabels(rank_names, fontsize=11, color='white', fontweight='bold')
    ax.set_yticks([margin + (12 - i) * cell_height + cell_height / 2 for i in range(13)])
    ax.set_yticklabels(rank_names, fontsize=11, color='white', fontweight='bold')

    # Style axis
    ax.spines['bottom'].set_color('#555555')
    ax.spines['left'].set_color('#555555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='white', length=5, width=1)
    ax.grid(True, alpha=0.1, color='white', linestyle='-', linewidth=0.5)

    # Labels and title
    ax.set_xlabel('Second Card Rank', fontsize=13, color='white', fontweight='bold')
    ax.set_ylabel('First Card Rank', fontsize=13, color='white', fontweight='bold')

    title_text = 'UTG Position - Starting Hand Strategy (8-player)\nHorizontal bars: FOLD (Blue) | CALL (Green) | RAISE (Red)\nBackground: Gradient by raise frequency'
    ax.set_title(title_text, fontsize=14, color='white', fontweight='bold', pad=20)

    # Add legend with GTO Wizard style
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4169E1', edgecolor='#333333', label='FOLD'),
        Patch(facecolor='#32CD32', edgecolor='#333333', label='CALL'),
        Patch(facecolor='#FF6347', edgecolor='#333333', label='RAISE'),
        Patch(facecolor='#D3D3D3', edgecolor='#333333', label='Cold hand (low raise%)'),
        Patch(facecolor='#F08080', edgecolor='#333333', label='Medium hand'),
        Patch(facecolor='#B22222', edgecolor='#333333', label='Hot hand (high raise%)'),
    ]
    legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                      fontsize=10, facecolor='#2a2a2a', edgecolor='#555555', framealpha=0.9)
    for text in legend.get_texts():
        text.set_color('white')

    # Save figure
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
    print(f"Heatmap saved to {output_path}")
    plt.close()


def main(checkpoint_path: str, output_path: str = "examples/starting_hand_strategy.png"):
    """Main evaluation function.

    Args:
        checkpoint_path: Path to trained model checkpoint
        output_path: Path to save output heatmap PNG
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = load_checkpoint(checkpoint_path)
    params = checkpoint['train_state']['params']
    config = checkpoint.get('config', {})

    print(f"Config: {config}")
    print(f"Parameters loaded successfully")

    # Initialize network
    num_actions = 3  # FOLD, CALL, RAISE
    network = ActorCritic(num_actions, activation=config.get('ACTIVATION', 'tanh'))

    # Create dummy observation to check shape
    dummy_obs = make_starting_hand_obs(0, 1, num_players=8)
    print(f"Observation shape: {dummy_obs.shape}")

    # Collect action probabilities for all starting hands
    print("Evaluating all starting hands...")
    hand_probs = {}

    fold_probs_all = []
    call_probs_all = []
    raise_probs_all = []

    # Iterate through all 13x13 combinations
    for rank1 in range(13):
        for rank2 in range(13):
            # Generate two unique card IDs for this hand
            # Use different suits to avoid duplicate cards
            card1_id = rank1  # Clubs
            card2_id = rank2 + 13  # Diamonds (different suit)

            # Create observation
            obs = make_starting_hand_obs(card1_id, card2_id, num_players=8)

            # Get probabilities
            fold_prob, call_prob, raise_prob = get_action_probabilities(network, params, obs)

            hand_probs[(rank1, rank2)] = (fold_prob, call_prob, raise_prob)
            fold_probs_all.append(fold_prob)
            call_probs_all.append(call_prob)
            raise_probs_all.append(raise_prob)

            hand_name = get_hand_name(rank1, rank2)
            print(f"  {hand_name:4s}: FOLD={fold_prob:.3f}, CALL={call_prob:.3f}, RAISE={raise_prob:.3f}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"FOLD:  mean={np.mean(fold_probs_all):.3f}, std={np.std(fold_probs_all):.3f}")
    print(f"CALL:  mean={np.mean(call_probs_all):.3f}, std={np.std(call_probs_all):.3f}")
    print(f"RAISE: mean={np.mean(raise_probs_all):.3f}, std={np.std(raise_probs_all):.3f}")

    # Create visualization
    print(f"\nCreating heatmap visualization...")
    create_heatmap_visualization(hand_probs, output_path)

    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_policy.py <checkpoint_path> [output_path]")
        print("Example: python eval_policy.py /tmp/poker/ppo_poker_20231015_120000/checkpoints/checkpoint_000050")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "examples/starting_hand_strategy.png"

    main(checkpoint_path, output_path)
