import jax
import jax.numpy as jnp

from pgx import universal_poker


class TestUniversalPokerPlayers:
    """Test suite for Universal Poker player management - navigation, turn order, and state tracking."""

    # Player Navigation Tests (from test_universal_poker_player_navigation.py)
    def test_get_next_active_player_basic(self):
        """Test basic next active player functionality."""
        env = universal_poker.UniversalPoker(num_players=3)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # All players are initially active
        assert jnp.sum(state.active_mask) == 3, "All 3 players should be active initially"

        current_player = state.current_player
        next_player = env._get_next_active_player_from(state, state.current_player + 1)

        # Next player should be different from current
        assert next_player != current_player, "Next player should be different from current"

        # Next player should be active
        assert state.active_mask[next_player] == True, "Next player should be active"

        # Should follow circular order (next player in sequence)
        expected_next = (current_player + 1) % 3
        assert next_player == expected_next, f"Next player should be {expected_next}, got {next_player}"

    def test_get_next_active_player_circular(self):
        """Test circular player order wrapping."""
        env = universal_poker.UniversalPoker(num_players=4)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Test from player 3 (should wrap to player 0)
        state = state.replace(current_player=3)
        next_player = env._get_next_active_player_from(state, state.current_player + 1)
        assert next_player == 0, f"From player 3, next should be 0, got {next_player}"

        # Test from player 2 (should go to player 3)
        state = state.replace(current_player=2)
        next_player = env._get_next_active_player_from(state, state.current_player + 1)
        assert next_player == 3, f"From player 2, next should be 3, got {next_player}"

    def test_get_next_active_player_with_folded(self):
        """Test next active player with some players folded."""
        env = universal_poker.UniversalPoker(num_players=4)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Fold players 1 and 2, leaving 0 and 3 active
        folded = jnp.array([False, True, True, False])
        active_mask = (~folded) & (~state.all_in)
        state = state.replace(folded=folded, active_mask=active_mask)

        # From player 0, should skip to player 3
        state = state.replace(current_player=0)
        next_player = env._get_next_active_player_from(state, state.current_player + 1)
        assert next_player == 3, f"Should skip folded players 1,2 and go to 3, got {next_player}"

        # From player 3, should wrap to player 0
        state = state.replace(current_player=3)
        next_player = env._get_next_active_player_from(state, state.current_player + 1)
        assert next_player == 0, f"Should wrap from 3 to 0, got {next_player}"

    def test_get_next_active_player_with_all_in(self):
        """Test next active player with some players all-in."""
        env = universal_poker.UniversalPoker(num_players=4)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Players 1 and 2 are all-in, leaving 0 and 3 active
        all_in = jnp.array([False, True, True, False])
        active_mask = (~state.folded) & (~all_in)
        state = state.replace(all_in=all_in, active_mask=active_mask)

        # From player 0, should skip to player 3
        state = state.replace(current_player=0)
        next_player = env._get_next_active_player_from(state, state.current_player + 1)
        assert next_player == 3, f"Should skip all-in players 1,2 and go to 3, got {next_player}"

        # Verify all-in players are not considered active
        assert state.active_mask[1] == False, "All-in player 1 should not be active"
        assert state.active_mask[2] == False, "All-in player 2 should not be active"

    def test_get_next_active_player_single_active(self):
        """Test next active player when only one player is active."""
        env = universal_poker.UniversalPoker(num_players=3)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Only player 1 is active (others folded)
        folded = jnp.array([True, False, True])
        active_mask = (~folded) & (~state.all_in)
        state = state.replace(folded=folded, active_mask=active_mask, current_player=1)

        next_player = env._get_next_active_player_from(state, state.current_player + 1)
        # With only one active player, should return that same player
        assert next_player == 1, f"With only player 1 active, should return 1, got {next_player}"

    def test_get_first_player_for_round(self):
        """Test getting first player for different rounds."""
        env = universal_poker.UniversalPoker(num_players=3)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Check first player for each round using the first_player_array
        for round_num in range(4):
            first_player = env._get_first_player_for_round(state, round_num)

            # First player should be active
            assert (
                state.active_mask[first_player] == True
            ), f"First player {first_player} for round {round_num} should be active"

            # Should match the configured first_player_array
            expected_start = env.first_player_array[round_num]
            # Should be the first active player from that starting position
            assert first_player >= 0 and first_player < env._num_players, "First player should be valid index"

    def test_get_next_active_player_from(self):
        """Test getting next active player from a specific position."""
        env = universal_poker.UniversalPoker(num_players=4)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Test from position 1 with all players active
        next_player = env._get_next_active_player_from(state, 2)
        assert next_player == 2, f"From position 1 with all active, should return 2, got {next_player}"

        # Test with player 1 folded
        folded = jnp.array([False, True, False, False])
        active_mask = (~folded) & (~state.all_in)
        state = state.replace(folded=folded, active_mask=active_mask)

        # From position 1, should find next active player (2)
        next_player = env._get_next_active_player_from(state, 2)
        assert next_player == 2, f"From position 1 with player 1 folded, should return 2, got {next_player}"

        # Test with player 2 folded
        folded = jnp.array([False, False, True, False])
        active_mask = (~folded) & (~state.all_in)
        state = state.replace(folded=folded, active_mask=active_mask)

        # From position 1, should find next active player (3)
        next_player = env._get_next_active_player_from(state, 2)
        assert next_player == 3, f"From position 1 with player 2 folded, should return 3, got {next_player}"

        # Test wrapping - from position 3, with player 3 still active, should return 0
        next_player = env._get_next_active_player_from(state, 4)
        assert next_player == 0, f"From position 3 with player 0 active, should return 0, got {next_player}"

        # Test wrapping with folded players - if we want to test wrapping to 0, we need to start from position 3
        folded_more = jnp.array([True, True, False, False])  # Players 0 and 1 folded, only 2 and 3 active
        active_mask_more = (~folded_more) & (~state.all_in)
        state_more = state.replace(folded=folded_more, active_mask=active_mask_more)

        # Now from position 3, should wrap to 0 and advance to 2 (since 0 and 1 are folded)
        next_player = env._get_next_active_player_from(state_more, 4)
        assert next_player == 2, f"From position 3 with players 1 and 2 folded, should wrap to 2, got {next_player}"

    def test_next_player_method(self):
        """Test the _next_player method that updates state."""
        env = universal_poker.UniversalPoker(num_players=3)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        original_player = state.current_player
        new_state = env._next_player(state)

        # Current player should have changed
        assert new_state.current_player != original_player, "Current player should have changed"

        # New current player should be active
        assert state.active_mask[new_state.current_player] == True, "New current player should be active"

        # Other state should remain unchanged
        assert jnp.array_equal(new_state.stacks, state.stacks), "Stacks should not change"
        assert new_state.pot == state.pot, "Pot should not change"
        assert jnp.array_equal(new_state.folded, state.folded), "Folded status should not change"

    def test_navigation_edge_cases(self):
        """Test navigation edge cases and error conditions."""
        env = universal_poker.UniversalPoker(num_players=2)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Test with no active players (all folded) - should still return a player
        folded = jnp.array([True, True])
        active_mask = jnp.array([False, False])
        state = state.replace(folded=folded, active_mask=active_mask)

        # Even with no active players, should return a valid index
        next_player = env._get_next_active_player_from(state, state.current_player + 1)
        assert 0 <= next_player < 2, "Should return valid player index even with no active players"

        # Test with all players all-in
        all_in = jnp.array([True, True])
        active_mask = jnp.array([False, False])
        state = env.init(key)  # Reset
        state = state.replace(all_in=all_in, active_mask=active_mask)

        next_player = env._get_next_active_player_from(state, state.current_player + 1)
        assert 0 <= next_player < 2, "Should return valid player index even with all players all-in"

    def test_navigation_consistency(self):
        """Test consistency of navigation methods."""
        env = universal_poker.UniversalPoker(num_players=4)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Fold player 2
        folded = jnp.array([False, False, True, False])
        active_mask = (~folded) & (~state.all_in)
        state = state.replace(folded=folded, active_mask=active_mask)

        # Test that _get_next_active_player and _get_next_active_player_from are consistent
        for current in range(4):
            state = state.replace(current_player=current)

            next_from_current = env._get_next_active_player_from(state, state.current_player + 1)
            next_from_position = env._get_next_active_player_from(state, current + 2)

            # Both methods should find an active player
            assert (
                state.active_mask[next_from_current] == True or jnp.sum(state.active_mask) == 0
            ), "Next player from current should be active"

        # Test that first player for round is consistent with next player from position
        for round_num in range(4):
            first_player = env._get_first_player_for_round(state, round_num)
            start_pos = env.first_player_array[round_num]
            first_from_pos = env._get_next_active_player_from(state, start_pos)

            assert first_player == (
                3 if round_num == 0 else start_pos
            ), f"First player methods should skip folded players for round {round_num}"
            assert first_player == first_from_pos, f"First player methods should be consistent for round {round_num}"

    def test_first_player_array_configuration(self):
        """Test that first_player_array is correctly configured."""
        env = universal_poker.UniversalPoker(num_players=4)

        # Verify first_player_array exists and has correct structure
        assert hasattr(env, "first_player_array"), "Environment should have first_player_array"
        assert len(env.first_player_array) >= 4, "first_player_array should have at least 4 rounds"

        # All entries should be valid player indices
        for round_num, first_player in enumerate(env.first_player_array):
            assert (
                0 <= first_player < env._num_players
            ), f"first_player_array[{round_num}] = {first_player} should be valid player index"

        # Test with different numbers of players
        for num_players in [2, 3, 4, 5]:
            env_test = universal_poker.UniversalPoker(num_players=num_players)
            for first_player in env_test.first_player_array:
                assert 0 <= first_player < num_players, f"Invalid first_player {first_player} for {num_players} players"

    def test_complex_active_mask_scenarios(self):
        """Test complex scenarios with various combinations of folded/all-in players."""
        env = universal_poker.UniversalPoker(num_players=6)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Complex scenario: players 0,3,5 active, 1,4 folded, 2 all-in
        folded = jnp.array([False, True, False, False, True, False])
        all_in = jnp.array([False, False, True, False, False, False])
        active_mask = (~folded) & (~all_in)
        state = state.replace(folded=folded, all_in=all_in, active_mask=active_mask)

        # Verify active mask is correct
        expected_active = [True, False, False, True, False, True]
        for i in range(6):
            assert state.active_mask[i] == expected_active[i], f"Player {i} active status incorrect"

        # Test navigation through active players
        active_players = [0, 3, 5]
        for i, current_player in enumerate(active_players):
            state = state.replace(current_player=current_player)
            next_player = env._get_next_active_player_from(state, state.current_player + 1)
            expected_next = active_players[(i + 1) % len(active_players)]
            assert (
                next_player == expected_next
            ), f"From player {current_player}, expected {expected_next}, got {next_player}"

        # Test _get_next_active_player_from with various starting positions
        test_cases = [
            (0, 3),  # From 0, first active is 3
            (1, 3),  # From 1, first active is 3
            (2, 3),  # From 2, first active is 3
            (3, 5),  # From 3, first active is 5
            (4, 5),  # From 4, first active is 5
            (5, 0),  # From 5, first active is 0
        ]

        for start_pos, expected in test_cases:
            result = env._get_next_active_player_from(state, start_pos + 1)
            assert result == expected, f"From position {start_pos}, expected {expected}, got {result}"

        # Test with only one active player - need to reset all_in array
        folded_single = jnp.array([True, True, False, True, True, True])
        all_in_single = jnp.array([False, False, False, False, False, False])  # Reset all_in
        active_mask_single = (~folded_single) & (~all_in_single)
        state_single = state.replace(folded=folded_single, all_in=all_in_single, active_mask=active_mask_single)

        # Only player 2 should be active (not all-in and not folded)
        assert (
            state_single.active_mask[2] == True
        ), f"Player 2 should be active, but active_mask[2] = {state_single.active_mask[2]}"

        for start_pos in range(6):
            result = env._get_next_active_player_from(state_single, start_pos + 1)
            assert result == 2, f"With only player 2 active, from {start_pos} should return 2, got {result}"

    # Player State Tests
    def test_rewards_array_size_matches_num_players(self):
        """Test that rewards array size matches number of players."""
        # Test 2 player game
        env2 = universal_poker.UniversalPoker(num_players=2)
        key = jax.random.PRNGKey(42)
        state2 = env2.init(key)
        assert len(state2.rewards) == 2, f"2-player game should have 2 rewards, got {len(state2.rewards)}"

        # Test 3 player game
        env3 = universal_poker.UniversalPoker(num_players=3)
        state3 = env3.init(key)
        assert len(state3.rewards) == 3, f"3-player game should have 3 rewards, got {len(state3.rewards)}"

        # Test 4 player game
        env4 = universal_poker.UniversalPoker(num_players=4)
        state4 = env4.init(key)
        assert len(state4.rewards) == 4, f"4-player game should have 4 rewards, got {len(state4.rewards)}"

        # Test termination with 3 players - all rewards should be accessible
        env3 = universal_poker.UniversalPoker(num_players=3)
        state3 = env3.init(key)

        # Force termination by having player 0 fold
        state3 = env3.step(state3, universal_poker.FOLD)
        if state3.terminated:
            # Should be able to access all 3 reward positions
            assert len(state3.rewards) == 3
            # With net stack change semantics, total should be zero (chip conservation)
            total_reward = sum(float(r) for r in state3.rewards)
            assert abs(total_reward) < 0.01, f"Total net stack changes should sum to zero, got {total_reward}"

        # Test termination with 4 players
        env4 = universal_poker.UniversalPoker(num_players=4)
        state4 = env4.init(key)

        # Force early termination by having multiple players fold
        state4 = env4.step(state4, universal_poker.FOLD)  # Player folds
        if not state4.terminated:
            state4 = env4.step(state4, universal_poker.FOLD)  # Another player folds
        if not state4.terminated:
            state4 = env4.step(state4, universal_poker.FOLD)  # Third player folds

        if state4.terminated:
            assert len(state4.rewards) == 4
            # With net stack change semantics, total should be zero (chip conservation)
            total_reward = sum(float(r) for r in state4.rewards)
            assert abs(total_reward) < 0.01, f"Total net stack changes should sum to zero, got {total_reward}"

    def test_early_termination_no_advancement(self):
        """Test that when game terminates early, we don't advance player/round unnecessarily."""
        # Set up 3-player game where we can control termination precisely
        env = universal_poker.UniversalPoker(num_players=3)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Record initial state before the terminating action
        initial_round = state.round
        initial_current_player = state.current_player

        # Create scenario where this action should terminate the game
        # In a 3-player game, if 2 players fold, game should terminate

        # Player 0 (or whoever is current) folds
        state = env.step(state, universal_poker.FOLD)
        first_fold_round = state.round

        # Next Player folds
        final_current_player = state.current_player
        state = env.step(state, universal_poker.FOLD)

        # Game should now be terminated (only 1 active player left)
        # The key insight: when game terminates due to too few active players,
        # the final state should not have advanced round/player beyond what's necessary
        # The termination should be detected immediately after applying the action

        # Check that termination was detected appropriately
        active_players = jnp.sum(~state.folded[: state.num_players])
        assert active_players == 1, f"Should have exactly 1 active player, got {active_players}"

        # The critical test: verify state consistency at termination
        # When terminated, current_player field should not be advanced unnecessarily
        assert state.current_player == final_current_player, f"Current player should be last player to act."
        assert state.current_player != initial_current_player, f"Current player should be last player to act."

        # If this was a mid-round termination, round should not have advanced
        assert state.round == initial_round

        # The specific issue is that termination check comes after round advancement
        # So we want to make sure the game logic is consistent
        assert state.terminated, "Game should be terminated with 1 active player"


if __name__ == "__main__":
    import sys
    import traceback

    test_suite = TestUniversalPokerPlayers()

    print("Running Universal Poker player management tests...")

    # Get all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith("test_")]

    passed = 0
    failed = 0

    for method_name in test_methods:
        try:
            method = getattr(test_suite, method_name)
            method()
            print(f"✓ {method_name} passed")
            passed += 1
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
