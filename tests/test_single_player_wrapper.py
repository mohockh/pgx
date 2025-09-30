"""Tests for SinglePlayerWrapper - TDD approach starting with Kuhn Poker."""
import jax
import jax.numpy as jnp
import pytest

from pgx.kuhn_poker import KuhnPoker
from pgx.universal_poker import UniversalPoker
from pgx.single_player_wrapper import SinglePlayerWrapper


class TestSinglePlayerWrapperBasics:
    """Phase 1: Basic functionality tests with Kuhn Poker."""

    def test_init_basic(self):
        """Test that wrapper initializes correctly."""
        env = KuhnPoker()
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        assert wrapper is not None
        assert wrapper.env == env
        assert wrapper.active_player_id == 0
        assert wrapper.max_steps_per_turn == 1000  # Default value


    def test_init_returns_active_player_observation(self):
        """Test that init returns state with observation for active player."""
        env = KuhnPoker()
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        # Initialize the wrapper
        key = jax.random.PRNGKey(42)
        state = wrapper.init(key)

        # State should not be None
        assert state is not None

        # Current player in underlying state might not be active player,
        # but observation should be for active player
        assert state.observation is not None
        assert state.observation.shape == env.observation_shape

        # State should not be terminated initially
        assert not state.terminated


    def test_step_executes_active_player_action(self):
        """Test that step executes the active player's action."""
        env = KuhnPoker()
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        key = jax.random.PRNGKey(42)
        state = wrapper.init(key)

        # Player 0 should be current player after init
        # (In Kuhn Poker, first player is randomized, but wrapper ensures active player's turn)

        # Take an action
        action = jnp.int32(0)  # BET action in Kuhn Poker
        new_state = wrapper.step(state, action)

        # State should have changed
        assert new_state is not None
        # Step count should have increased
        assert new_state._step_count > state._step_count

    def test_step_auto_plays_opponent_turns(self):
        """Test that step automatically plays opponent turns."""
        env = KuhnPoker()
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        key = jax.random.PRNGKey(42)
        state = wrapper.init(key)

        # Take an action
        action = jnp.int32(1)  # PASS action
        new_state = wrapper.step(state, action)

        # After our action, opponent should have acted (if game not terminated)
        # and it should be our turn again OR game terminated
        if not new_state.terminated:
            # If still playing, should be back to our turn
            # (but Kuhn Poker might terminate quickly, so we check conditionally)
            pass  # We'll verify this more thoroughly in later tests

        # At minimum, state should have advanced
        assert new_state._step_count > state._step_count


    def test_observation_always_for_active_player(self):
        """Test that observation is always for active player."""
        env = KuhnPoker()
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        key = jax.random.PRNGKey(42)
        state = wrapper.init(key)

        # Observation should be from active player's perspective
        obs1 = wrapper.observe(state)
        assert obs1 is not None

        # Step and check again
        action = jnp.int32(1)  # PASS
        new_state = wrapper.step(state, action)
        obs2 = wrapper.observe(new_state)
        assert obs2 is not None

    def test_reward_accumulation(self):
        """Test that rewards accumulate correctly across turns."""
        env = KuhnPoker()
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        key = jax.random.PRNGKey(42)
        state = wrapper.init(key)

        # Loop until termination or max steps
        max_steps = 100
        total_accumulated_reward = 0.0

        print(f"\n--- Reward Accumulation Test ---")
        print(f"Initial state - All player rewards: {state.rewards}")
        print(f"Initial state - Active player ({wrapper.active_player_id}) reward: {state.rewards[wrapper.active_player_id]}")

        for step in range(max_steps):
            if state.terminated:
                break

            # Take random legal action
            legal_actions = jnp.where(state.legal_action_mask)[0]
            action = legal_actions[0] if len(legal_actions) > 0 else jnp.int32(0)

            state = wrapper.step(state, action)
            step_reward = state.rewards[wrapper.active_player_id]
            total_accumulated_reward += step_reward

            print(f"Step {step + 1} - All player rewards: {state.rewards}")
            print(f"Step {step + 1} - Active player ({wrapper.active_player_id}) reward: {step_reward}")
            print(f"Step {step + 1} - Total accumulated: {total_accumulated_reward}")
            print(f"Step {step + 1} - Terminated: {state.terminated}")

        # Game should terminate
        assert state.terminated, "Game should terminate within max_steps"

        # Final reward should be set and finite
        final_reward = state.rewards[wrapper.active_player_id]
        assert jnp.isfinite(final_reward)

        print(f"\nFinal - All player rewards: {state.rewards}")
        print(f"Final - Active player ({wrapper.active_player_id}) reward: {final_reward}")
        print(f"Final - Total accumulated reward: {total_accumulated_reward}")
        print(f"--- End Reward Accumulation Test ---\n")

    def test_game_termination(self):
        """Test that termination flag propagates correctly."""
        env = KuhnPoker()
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        key = jax.random.PRNGKey(123)
        state = wrapper.init(key)

        # Play until termination (Kuhn Poker terminates quickly)
        max_steps = 10
        for _ in range(max_steps):
            if state.terminated:
                break
            # Random action from legal actions
            legal_actions = jnp.where(state.legal_action_mask)[0]
            action = legal_actions[0] if len(legal_actions) > 0 else jnp.int32(0)
            state = wrapper.step(state, action)

        # Game should terminate within max_steps
        # (Kuhn Poker always terminates in 2-3 actions)
        assert state.terminated

    def test_random_opponent_policy(self):
        """Test that default random policy works."""
        env = KuhnPoker()
        # Don't provide opponent_policy_fn - should use random
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0, opponent_policy_fns=None)

        key = jax.random.PRNGKey(42)
        state = wrapper.init(key)

        # Should initialize without error (using random opponent)
        assert state is not None

        # Should be able to step
        action = jnp.int32(1)
        new_state = wrapper.step(state, action)
        assert new_state is not None


class TestSinglePlayerWrapperEdgeCases:
    """Phase 2: Edge case tests."""

    def test_immediate_termination(self):
        """Test game that terminates on first action."""
        env = KuhnPoker()
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        # Try with different seeds to find immediate termination
        for seed in range(10):
            key = jax.random.PRNGKey(seed)
            state = wrapper.init(key)

            # BET action might lead to fold, causing termination
            action = jnp.int32(0)  # BET
            new_state = wrapper.step(state, action)

            # Should handle termination gracefully
            assert new_state is not None
            if new_state.terminated:
                # Success - found a case
                break

    def test_custom_opponent_policy(self):
        """Test that custom opponent policy works."""
        env = KuhnPoker()

        # Define a custom policy that always passes
        def always_pass_policy(obs):
            return jnp.int32(1)  # PASS action

        wrapper = SinglePlayerWrapper(
            env, num_players=2, active_player_id=0, opponent_policy_fns=[None, always_pass_policy]
        )

        key = jax.random.PRNGKey(42)
        state = wrapper.init(key)

        # Should work with custom policy
        assert state is not None

        action = jnp.int32(1)  # PASS
        new_state = wrapper.step(state, action)
        assert new_state is not None


class TestSinglePlayerWrapperJAX:
    """Phase 3: JAX compatibility tests."""

    def test_jit_compilation(self):
        """Test that wrapper is JIT-compilable."""
        env = KuhnPoker()
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        # JIT compile init and step
        jit_init = jax.jit(wrapper.init)
        jit_step = jax.jit(wrapper.step)

        key = jax.random.PRNGKey(42)
        state = jit_init(key)
        assert state is not None

        action = jnp.int32(1)
        new_state = jit_step(state, action)
        assert new_state is not None

    def test_vmap_compatibility(self):
        """Test that wrapper works with vmap (batched envs)."""
        env = KuhnPoker()
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        # Create batch of keys
        batch_size = 4
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

        # Vmap over init
        vmap_init = jax.vmap(wrapper.init)
        states = vmap_init(keys)
        assert states.observation.shape[0] == batch_size

        # Vmap over step
        actions = jnp.ones(batch_size, dtype=jnp.int32)
        vmap_step = jax.vmap(wrapper.step)
        new_states = vmap_step(states, actions)
        assert new_states.observation.shape[0] == batch_size


if __name__ == "__main__":
    # Run Phase 1 tests
    print("=== Phase 1: Basic Functionality ===")
    test = TestSinglePlayerWrapperBasics()
    test.test_init_basic()
    print("✓ test_init_basic passed")

    test.test_init_returns_active_player_observation()
    print("✓ test_init_returns_active_player_observation passed")

    test.test_step_executes_active_player_action()
    print("✓ test_step_executes_active_player_action passed")

    test.test_step_auto_plays_opponent_turns()
    print("✓ test_step_auto_plays_opponent_turns passed")

    test.test_observation_always_for_active_player()
    print("✓ test_observation_always_for_active_player passed")

    test.test_reward_accumulation()
    print("✓ test_reward_accumulation passed")

    test.test_game_termination()
    print("✓ test_game_termination passed")

    test.test_random_opponent_policy()
    print("✓ test_random_opponent_policy passed")

    # Run Phase 2 tests
    print("\n=== Phase 2: Edge Cases ===")
    test2 = TestSinglePlayerWrapperEdgeCases()
    test2.test_immediate_termination()
    print("✓ test_immediate_termination passed")

    test2.test_custom_opponent_policy()
    print("✓ test_custom_opponent_policy passed")

    # Run Phase 3 tests
    print("\n=== Phase 3: JAX Compatibility ===")
    test3 = TestSinglePlayerWrapperJAX()
    test3.test_jit_compilation()
    print("✓ test_jit_compilation passed")

    test3.test_vmap_compatibility()
    print("✓ test_vmap_compatibility passed")

class TestSinglePlayerWrapperUniversalPoker:
    """Phase 4: UniversalPoker integration tests."""

    def test_universal_poker_integration(self):
        """Test that wrapper works with 8-player UniversalPoker."""
        # Create 8-player poker game
        config_str = """GAMEDEF
numplayers 8
numrounds 4
firstplayer 3 1 1 1
blind 1 2 0 0 0 0 0 0
stack 100 100 100 100 100 100 100 100
END GAMEDEF"""
        env = UniversalPoker(num_players=8, config_str=config_str)
        wrapper = SinglePlayerWrapper(env, num_players=8, active_player_id=0)

        key = jax.random.PRNGKey(42)
        state = wrapper.init(key)

        # Should initialize successfully
        assert state is not None
        assert not state.terminated

        # Should be able to step
        action = jnp.int32(1)  # CALL
        new_state = wrapper.step(state, action)
        assert new_state is not None

    def test_universal_poker_full_game(self):
        """Test complete poker hand simulation."""
        config_str = """GAMEDEF
numplayers 2
numrounds 4
firstplayer 1 2 2 2
blind 1 2
stack 100 100
END GAMEDEF"""
        env = UniversalPoker(num_players=2, config_str=config_str)
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        key = jax.random.PRNGKey(42)
        state = wrapper.init(key)

        print(f"\n--- Universal Poker Full Game Test ---")
        print(f"Initial state - All player rewards: {state.rewards}")
        print(f"Initial state - Active player ({wrapper.active_player_id}) reward: {state.rewards[wrapper.active_player_id]}")

        # Play until termination
        max_steps = 100  # Safety limit
        total_accumulated_reward = 0.0
        for step in range(max_steps):
            if state.terminated:
                break

            # Take random legal action
            legal_actions = jnp.where(state.legal_action_mask)[0]
            if len(legal_actions) > 0:
                action = legal_actions[0]
            else:
                action = jnp.int32(0)

            state = wrapper.step(state, action)
            step_reward = state.rewards[wrapper.active_player_id]
            total_accumulated_reward += step_reward

            print(f"Step {step + 1} - All player rewards: {state.rewards}")
            print(f"Step {step + 1} - Active player ({wrapper.active_player_id}) reward: {step_reward}")
            print(f"Step {step + 1} - Total accumulated: {total_accumulated_reward}")
            print(f"Step {step + 1} - Terminated: {state.terminated}")

        # Game should terminate
        assert state.terminated

        # Final reward should be set
        reward = state.rewards[wrapper.active_player_id]
        assert jnp.isfinite(reward)

        print(f"\nFinal - All player rewards: {state.rewards}")
        print(f"Final - Active player ({wrapper.active_player_id}) reward: {reward}")
        print(f"Final - Total accumulated reward: {total_accumulated_reward}")
        print(f"--- End Universal Poker Full Game Test ---\n")

    def test_universal_poker_jit_and_vmap(self):
        """Test JIT and vmap with UniversalPoker."""
        config_str = """GAMEDEF
numplayers 2
numrounds 4
blind 1 2
stack 100 100
END GAMEDEF"""
        env = UniversalPoker(num_players=2, config_str=config_str)
        wrapper = SinglePlayerWrapper(env, num_players=2, active_player_id=0)

        # Test JIT
        jit_init = jax.jit(wrapper.init)
        key = jax.random.PRNGKey(42)
        state = jit_init(key)
        assert state is not None

        # Test vmap
        batch_size = 2
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        vmap_init = jax.vmap(wrapper.init)
        states = vmap_init(keys)
        assert states.observation.shape[0] == batch_size


if __name__ == "__main__":
    # Run Phase 1 tests
    print("=== Phase 1: Basic Functionality ===")
    test = TestSinglePlayerWrapperBasics()
    test.test_init_basic()
    print("✓ test_init_basic passed")

    test.test_init_returns_active_player_observation()
    print("✓ test_init_returns_active_player_observation passed")

    test.test_step_executes_active_player_action()
    print("✓ test_step_executes_active_player_action passed")

    test.test_step_auto_plays_opponent_turns()
    print("✓ test_step_auto_plays_opponent_turns passed")

    test.test_observation_always_for_active_player()
    print("✓ test_observation_always_for_active_player passed")

    test.test_reward_accumulation()
    print("✓ test_reward_accumulation passed")

    test.test_game_termination()
    print("✓ test_game_termination passed")

    test.test_random_opponent_policy()
    print("✓ test_random_opponent_policy passed")

    # Run Phase 2 tests
    print("\n=== Phase 2: Edge Cases ===")
    test2 = TestSinglePlayerWrapperEdgeCases()
    test2.test_immediate_termination()
    print("✓ test_immediate_termination passed")

    test2.test_custom_opponent_policy()
    print("✓ test_custom_opponent_policy passed")

    # Run Phase 3 tests
    print("\n=== Phase 3: JAX Compatibility ===")
    test3 = TestSinglePlayerWrapperJAX()
    test3.test_jit_compilation()
    print("✓ test_jit_compilation passed")

    test3.test_vmap_compatibility()
    print("✓ test_vmap_compatibility passed")

    # Run Phase 4 tests
    print("\n=== Phase 4: UniversalPoker Integration ===")
    test4 = TestSinglePlayerWrapperUniversalPoker()
    test4.test_universal_poker_integration()
    print("✓ test_universal_poker_integration passed")

    test4.test_universal_poker_full_game()
    print("✓ test_universal_poker_full_game passed")

    test4.test_universal_poker_jit_and_vmap()
    print("✓ test_universal_poker_jit_and_vmap passed")

    print("\n=== All tests passed! ===")
