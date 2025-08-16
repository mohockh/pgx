"""
Performance tests for poker hand evaluation.

Compare the new JAX evaluator against the old placeholder implementation.
"""

import time
import jax
import jax.numpy as jnp
import sys
import os

from pgx.poker_eval.jax_evaluator import evaluate_hand_jax
from pgx.universal_poker import UniversalPoker

def old_evaluate_hand(hole_cards, board_cards, round_num):
    """Old placeholder implementation for comparison."""
    # Number of board cards visible in each round: preflop=0, flop=3, turn=1, river=1
    num_board_cards = jnp.array([0, 3, 4, 5], dtype=jnp.int32)  # Cumulative counts
    
    # Get number of visible board cards for this round
    num_visible = jax.lax.select(round_num >= 4, 5, num_board_cards[round_num])
    
    # Create a mask for visible board cards
    visible_mask = jnp.arange(5) < num_visible
    visible_board = jnp.where(visible_mask, board_cards[:5], -1)
    
    # Combine hole cards and visible board cards
    all_cards = jnp.concatenate([hole_cards[:2], visible_board])
    valid_cards = jnp.where(all_cards >= 0, all_cards, 0)
    
    # Return highest card as simple hand strength (old implementation)
    return jnp.max(valid_cards)

def new_evaluate_hand(hole_cards, board_cards, round_num):
    """New JAX implementation."""
    # Number of board cards visible in each round: preflop=0, flop=3, turn=4, river=5
    num_board_cards = jnp.array([0, 3, 4, 5], dtype=jnp.int32)  # Cumulative counts
    
    # Get number of visible board cards for this round
    num_visible = jax.lax.select(round_num >= 4, 5, num_board_cards[round_num])
    
    # Create a mask for visible board cards
    visible_mask = jnp.arange(5) < num_visible
    visible_board = jnp.where(visible_mask, board_cards[:5], -1)
    
    # Combine hole cards and visible board cards
    all_cards = jnp.concatenate([hole_cards[:2], visible_board])
    
    # For preflop, just return high card value
    def preflop_eval():
        return jnp.max(hole_cards[:2])
    
    def postflop_eval():
        return evaluate_hand_jax(all_cards)
    
    return jax.lax.cond(
        num_visible == 0,  # Preflop
        preflop_eval,
        postflop_eval
    )

def generate_test_hands(num_hands: int, key: jax.random.PRNGKey):
    """Generate random test hands."""
    # Generate random hole cards and board cards
    keys = jax.random.split(key, num_hands)
    
    hole_cards = []
    board_cards = []
    rounds = []
    
    for i in range(num_hands):
        # Random cards (simplified - just pick random card IDs)
        hand_cards = jax.random.choice(keys[i], 52, shape=(7,), replace=False)
        hole_cards.append(hand_cards[:2])
        board_cards.append(hand_cards[2:7])
        rounds.append(jax.random.randint(keys[i], (), 1, 4))  # Rounds 1-3 (postflop)
    
    return jnp.array(hole_cards), jnp.array(board_cards), jnp.array(rounds)

def benchmark_evaluators(num_hands: int = 1000):
    """Benchmark old vs new evaluators."""
    print(f"Benchmarking hand evaluation with {num_hands} hands...")
    
    # Generate test data
    key = jax.random.PRNGKey(42)
    hole_cards, board_cards, rounds = generate_test_hands(num_hands, key)
    
    # Compile both functions
    old_eval_jit = jax.jit(jax.vmap(old_evaluate_hand))
    new_eval_jit = jax.jit(jax.vmap(new_evaluate_hand))
    
    # Warm up
    print("Warming up...")
    old_results = old_eval_jit(hole_cards, board_cards, rounds)
    new_results = new_eval_jit(hole_cards, board_cards, rounds)
    
    # Benchmark old evaluator
    print("Benchmarking old evaluator...")
    start_time = time.time()
    for _ in range(10):
        old_results = old_eval_jit(hole_cards, board_cards, rounds)
    old_time = (time.time() - start_time) / 10
    
    # Benchmark new evaluator  
    print("Benchmarking new evaluator...")
    start_time = time.time()
    for _ in range(10):
        new_results = new_eval_jit(hole_cards, board_cards, rounds)
    new_time = (time.time() - start_time) / 10
    
    print(f"\nResults:")
    print(f"Old evaluator: {old_time:.4f}s ({num_hands/old_time:.0f} hands/sec)")
    print(f"New evaluator: {new_time:.4f}s ({num_hands/new_time:.0f} hands/sec)")
    print(f"Speedup: {old_time/new_time:.2f}x")
    
    # Compare some results
    print(f"\nSample old results: {old_results[:5]}")
    print(f"Sample new results: {new_results[:5]}")
    
    return old_time, new_time

def test_universal_poker_performance():
    """Test performance with the actual Universal Poker environment."""
    print("\nTesting Universal Poker performance with new evaluator...")
    
    env = UniversalPoker(num_players=2, stack_size=100)
    
    # Test vectorized game simulation
    batch_size = 100
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    
    # Vectorized functions
    init_fn = jax.jit(jax.vmap(env.init))
    
    print(f"Running {batch_size} games...")
    start_time = time.time()
    
    # Initialize games
    states = init_fn(keys)
    
    # Play some actions
    for round_num in range(4):  # Play through all rounds
        # Simple action: always call/check
        actions = jnp.ones(batch_size, dtype=jnp.int32)  # Call action
        
        # This will test the hand evaluator when games end
        step_fn = jax.jit(jax.vmap(env.step))
        states = step_fn(states, actions)
        
        terminated_count = jnp.sum(states.terminated)
        print(f"Round {round_num}: {terminated_count}/{batch_size} games terminated")
        
        if terminated_count == batch_size:
            break
    
    elapsed_time = time.time() - start_time
    print(f"Completed in {elapsed_time:.3f}s ({batch_size/elapsed_time:.1f} games/sec)")

if __name__ == "__main__":
    print("Poker Hand Evaluation Performance Tests")
    print("=" * 50)
    
    # Test hand evaluation performance
    benchmark_evaluators(1000)
    
    # Test with Universal Poker
    test_universal_poker_performance()
    
    print("\n✅ Performance tests completed!")
