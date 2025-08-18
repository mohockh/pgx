"""
Performance tests for poker hand evaluation.

Compare the new JAX evaluator against the old placeholder implementation.
"""

import time
import jax
import jax.numpy as jnp
import sys
import os

from pgx.poker_eval.jax_evaluator_new import evaluate_hand
from pgx.poker_eval.cardset import cards_to_cardset
from pgx.universal_poker import UniversalPoker


def generate_test_cardsets(num_hands: int, key: jax.random.PRNGKey):
    """Generate random 7-card hands as cardsets - vectorized version."""
    # Split keys for each hand
    keys = jax.random.split(key, num_hands)
    
    # Vectorized cardset generation
    def generate_single_cardset(single_key):
        # Generate 7 unique cards
        cards = jax.random.choice(single_key, 52, shape=(7,), replace=False)
        # Convert to cardset
        return cards_to_cardset(cards)
    
    # Use vmap to vectorize across all hands
    all_cardsets = jax.vmap(generate_single_cardset)(keys)
    
    return all_cardsets

def benchmark_evaluators(num_hands: int = 1000, chunk_size: int = 100000):
    """Benchmark cardset-based hand evaluator with chunking for memory efficiency."""
    chunk_size = min(num_hands, chunk_size)
    print(f"Benchmarking cardset hand evaluation with {num_hands} hands (chunk_size={chunk_size})...")
    
    # Compile function once
    eval_jit = jax.jit(jax.vmap(evaluate_hand))
    
    # Warm up with small batch
    print("Warming up...")
    key = jax.random.PRNGKey(42)
    warmup_cardsets = generate_test_cardsets(chunk_size, key)
    _ = eval_jit(warmup_cardsets)
    
    # Calculate number of chunks
    num_chunks = (num_hands + chunk_size - 1) // chunk_size
    print(f"Processing {num_chunks} chunks...")
    
    # Benchmark with chunking
    start_time = time.time()
    total_results = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_hands)
        current_chunk_size = end_idx - start_idx
        
        # Generate chunk data
        chunk_key = jax.random.split(key, num_chunks)[chunk_idx]
        chunk_cardsets = generate_test_cardsets(current_chunk_size, chunk_key)
        
        # Evaluate chunk
        chunk_results = eval_jit(chunk_cardsets)
        total_results.append(chunk_results)
        
        if chunk_idx % max(1, num_chunks // 10) == 0:
            print(f"  Processed chunk {chunk_idx + 1}/{num_chunks}")
    
    eval_time = time.time() - start_time
    
    # Combine results for sample output
    first_chunk_results = total_results[0] if total_results else jnp.array([])
    
    print(f"\nResults:")
    print(f"Cardset Evaluator: {eval_time:.4f}s ({num_hands/eval_time:.0f} hands/sec)")
    print(f"Sample results: {first_chunk_results[:5] if len(first_chunk_results) >= 5 else first_chunk_results}")
    
    return eval_time

def test_universal_poker_performance(batch_size: int = 100):
    """Test performance with the actual Universal Poker environment."""
    print("\nTesting Universal Poker performance with cardset evaluator...")
    
    env = UniversalPoker(num_players=2, stack_size=100)
    
    # Test vectorized game simulation
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
    benchmark_evaluators(1000000, chunk_size=1000000)
    
    # Test with Universal Poker
    test_universal_poker_performance(1000)
    
    print("\n✅ Performance tests completed!")
