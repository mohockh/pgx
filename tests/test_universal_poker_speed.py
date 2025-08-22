import time
import jax
import jax.numpy as jnp
import sys
import os

from pgx.universal_poker import UniversalPoker, FOLD, CALL, RAISE


def play_random_hands_vectorized(env, keys):
    """Play multiple hands simultaneously using vectorization."""
    batch_size = keys.shape[0]
    
    # Create vectorized functions
    init_fn = jax.vmap(env.init)
    step_fn = jax.vmap(env.step)
    
    # Initialize all states
    states = init_fn(keys)
    
    # Track steps for each hand
    steps = jnp.zeros(batch_size, dtype=jnp.int32)
    max_steps = 50
    
    def loop_body(loop_state):
        states, steps, step_num = loop_state
        
        # Check which hands are still active
        active = ~states.terminated
        
        # Simple random action selection without dynamic indexing
        # Just randomly pick from all possible actions and let the environment handle invalid ones
        action_keys = jax.random.split(jax.random.PRNGKey(step_num + 42), batch_size)
        random_values = jax.vmap(lambda k: jax.random.randint(k, (), 0, 3))(action_keys)
        
        # Mask actions to legal ones where possible, otherwise use action 0 (fold)
        def select_legal_action(state, random_val):
            # Try each action in order: random_val, fold(0), call(1), raise(2)
            action_candidates = jnp.array([random_val, 0, 1, 2])
            
            def try_action(i):
                action = action_candidates[i]
                return jnp.where(state.legal_action_mask[action], action, -1)
            
            # Find first legal action
            result = -1
            for i in range(4):
                candidate = try_action(i)
                result = jnp.where((result == -1) & (candidate != -1), candidate, result)
            
            # If no legal action found (shouldn't happen), default to 0
            return jnp.where(result == -1, 0, result)
        
        actions = jax.vmap(select_legal_action)(states, random_values)
        
        # Step all hands
        new_states = step_fn(states, actions)
        
        # For simplicity, just use new states - JAX will handle inactive hands properly
        states = new_states
        
        # Increment step count for active hands
        steps = jnp.where(active, steps + 1, steps)
        
        return states, steps, step_num + 1
    
    def loop_cond(loop_state):
        states, steps, step_num = loop_state
        # Continue if we haven't reached max steps and some hands are still active
        return (step_num < max_steps) & jnp.any(~states.terminated)
    
    # Run the loop
    final_states, final_steps, _ = jax.lax.while_loop(
        loop_cond, loop_body, (states, steps, 0)
    )
    
    return final_states.terminated, final_steps


def benchmark_speed_vectorized(num_hands_to_benchmark: list, max_batch_size: int):
    """Benchmark using vectorized approach for better performance."""
    print("Universal Poker Speed Benchmark (Vectorized)")
    print("=" * 60)
    
    config_str = """GAMEDEF
numplayers = 8
stack = 100 100 100 100 100 100 100 100
blind = 1 2 0 0 0 0 0 0
END GAMEDEF"""
    env = UniversalPoker(num_players=8, config_str=config_str)
    
    # JIT compile the vectorized function
    play_vectorized_jit = jax.jit(play_random_hands_vectorized, static_argnums=(0,))

    for num_hands in num_hands_to_benchmark:
        print(f"\nTesting {num_hands} hands:")
        print("-" * 25)
        
        # Split into batches to avoid memory issues
        batch_size = min(num_hands, max_batch_size)
        num_batches = (num_hands + batch_size - 1) // batch_size
        
        start_time = time.time()
        total_completed = 0
        total_steps = 0

        for batch_idx in range(num_batches):
            # Calculate actual batch size for this batch
            current_batch_size = min(batch_size, num_hands - batch_idx * batch_size)
            
            # Generate keys for this batch
            keys = jax.random.split(jax.random.PRNGKey(42 + batch_idx), current_batch_size)
            
            # Run vectorized simulation
            terminated, steps = play_vectorized_jit(env, keys)
            
            # Count completed hands and total steps
            completed_in_batch = jnp.sum(terminated)
            steps_in_batch = jnp.sum(steps)
            
            total_completed += int(completed_in_batch)
            total_steps += int(steps_in_batch)
            
        elapsed_time = time.time() - start_time
        
        print(f"  Completed: {total_completed}/{num_hands}")
        print(f"  Total steps: {total_steps}")
        if total_completed > 0:
            print(f"  Avg steps/hand: {total_steps/total_completed:.1f}")
        print(f"  Time: {elapsed_time:.3f}s")
        if elapsed_time > 0:
            print(f"  Hands/sec: {total_completed/elapsed_time:.1f}")
            print(f"  Steps/sec: {total_steps/elapsed_time:.0f}")
        


def benchmark_speed(num_hands_to_benchmark: list):
    """Benchmark with smaller batches to avoid memory issues."""
    print("Universal Poker Speed Benchmark (Serial)")
    print("=" * 60)
    
    config_str = """GAMEDEF
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF"""
    env = UniversalPoker(num_players=2, config_str=config_str)
    
    for num_hands in num_hands_to_benchmark:
        print(f"\nTesting {num_hands} hands:")
        print("-" * 25)
        
        key = jax.random.PRNGKey(42)
        start_time = time.time()
        
        completed_hands = 0
        total_steps = 0
        
        for i in range(num_hands):
            key, subkey = jax.random.split(key)
            state = env.init(subkey)
            steps = 0
            max_steps = 50
            while not state.terminated and steps < max_steps:
                key, subkey = jax.random.split(key)
                legal_actions = jnp.where(state.legal_action_mask)[0]
                action = jax.random.choice(subkey, legal_actions)
                steps += 1
                state = env.step(state, action)
            
            if state.terminated:
                completed_hands += 1
                total_steps += steps
                
        elapsed_time = time.time() - start_time
        
        print(f"  Completed: {completed_hands}/{num_hands}")
        print(f"  Total steps: {total_steps}")
        if completed_hands > 0:
            print(f"  Avg steps/hand: {total_steps/completed_hands:.1f}")
        print(f"  Time: {elapsed_time:.3f}s")
        if elapsed_time > 0:
            print(f"  Hands/sec: {completed_hands/elapsed_time:.1f}")
            print(f"  Steps/sec: {total_steps/elapsed_time:.0f}")


if __name__ == "__main__":
    try:
        print("Universal Poker Speed Test")
        print("=" * 60)
        
        # Run both serial and vectorized benchmarks
        benchmark_speed(num_hands_to_benchmark=[10,])
        print("\n" + "=" * 60)
        benchmark_speed_vectorized(num_hands_to_benchmark=[10, 1000000], max_batch_size=2**16)
        
        print(f"\n\nAll tests completed successfully! ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
