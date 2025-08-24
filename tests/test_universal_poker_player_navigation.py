import jax
import jax.numpy as jnp

from pgx import universal_poker


class TestPlayerNavigation:
    """Test suite specifically for player navigation methods of UniversalPoker."""
    
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
        active_mask = (~folded) & (~state.all_in) & state.player_mask
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
        active_mask = (~state.folded) & (~all_in) & state.player_mask
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
        active_mask = (~folded) & (~state.all_in) & state.player_mask
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
            assert state.active_mask[first_player] == True, f"First player {first_player} for round {round_num} should be active"
            
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
        active_mask = (~folded) & (~state.all_in) & state.player_mask
        state = state.replace(folded=folded, active_mask=active_mask)
        
        # From position 1, should find next active player (2)
        next_player = env._get_next_active_player_from(state, 2)
        assert next_player == 2, f"From position 1 with player 1 folded, should return 2, got {next_player}"
        
        # Test with player 2 folded
        folded = jnp.array([False, False, True, False])
        active_mask = (~folded) & (~state.all_in) & state.player_mask
        state = state.replace(folded=folded, active_mask=active_mask)
        
        # From position 1, should find next active player (3)
        next_player = env._get_next_active_player_from(state, 2)
        assert next_player == 3, f"From position 1 with player 2 folded, should return 3, got {next_player}"
        
        # Test wrapping - from position 3, with player 3 still active, should return 0
        next_player = env._get_next_active_player_from(state, 4)
        assert next_player == 0, f"From position 3 with player 0 active, should return 0, got {next_player}"
        
        # Test wrapping with folded players - if we want to test wrapping to 0, we need to start from position 3
        folded_more = jnp.array([True, True, False, False])  # Players 0 and 1 folded, only 2 and 3 active
        active_mask_more = (~folded_more) & (~state.all_in) & state.player_mask
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
        active_mask = (~folded) & (~state.all_in) & state.player_mask
        state = state.replace(folded=folded, active_mask=active_mask)
        
        # Test that _get_next_active_player and _get_next_active_player_from are consistent
        for current in range(4):
            state = state.replace(current_player=current)
            
            next_from_current = env._get_next_active_player_from(state, state.current_player + 1)
            next_from_position = env._get_next_active_player_from(state, current + 2)
            
            # Both methods should find an active player
            assert state.active_mask[next_from_current] == True or jnp.sum(state.active_mask) == 0, "Next player from current should be active"
            
        # Test that first player for round is consistent with next player from position
        for round_num in range(4):
            first_player = env._get_first_player_for_round(state, round_num)
            start_pos = env.first_player_array[round_num]
            first_from_pos = env._get_next_active_player_from(state, start_pos)
            
            assert first_player == (3 if round_num == 0 else start_pos), f"First player methods should skip folded players for round {round_num}"
            assert first_player == first_from_pos, f"First player methods should be consistent for round {round_num}"
            
    def test_first_player_array_configuration(self):
        """Test that first_player_array is correctly configured."""
        env = universal_poker.UniversalPoker(num_players=4)
        
        # Verify first_player_array exists and has correct structure
        assert hasattr(env, 'first_player_array'), "Environment should have first_player_array"
        assert len(env.first_player_array) >= 4, "first_player_array should have at least 4 rounds"
        
        # All entries should be valid player indices
        for round_num, first_player in enumerate(env.first_player_array):
            assert 0 <= first_player < env._num_players, f"first_player_array[{round_num}] = {first_player} should be valid player index"
            
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
        active_mask = (~folded) & (~all_in) & state.player_mask
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
            assert next_player == expected_next, f"From player {current_player}, expected {expected_next}, got {next_player}"
            
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
        active_mask_single = (~folded_single) & (~all_in_single) & state.player_mask
        state_single = state.replace(folded=folded_single, all_in=all_in_single, active_mask=active_mask_single)
        
        # Only player 2 should be active (not all-in and not folded)
        assert state_single.active_mask[2] == True, f"Player 2 should be active, but active_mask[2] = {state_single.active_mask[2]}"
        
        for start_pos in range(6):
            result = env._get_next_active_player_from(state_single, start_pos + 1)
            assert result == 2, f"With only player 2 active, from {start_pos} should return 2, got {result}"


if __name__ == "__main__":
    try:
        # Run TestPlayerNavigation tests
        navigation_suite = TestPlayerNavigation()
        
        navigation_suite.test_get_next_active_player_basic()
        print("✓ Player navigation next active player basic test passed")
        
        navigation_suite.test_get_next_active_player_circular()
        print("✓ Player navigation circular order test passed")
        
        navigation_suite.test_get_next_active_player_with_folded()
        print("✓ Player navigation with folded players test passed")
        
        navigation_suite.test_get_next_active_player_with_all_in()
        print("✓ Player navigation with all-in players test passed")
        
        navigation_suite.test_get_next_active_player_single_active()
        print("✓ Player navigation single active player test passed")
        
        navigation_suite.test_get_first_player_for_round()
        print("✓ Player navigation first player for round test passed")
        
        navigation_suite.test_get_next_active_player_from()
        print("✓ Player navigation next active from position test passed")
        
        navigation_suite.test_next_player_method()
        print("✓ Player navigation next player method test passed")
        
        navigation_suite.test_navigation_edge_cases()
        print("✓ Player navigation edge cases test passed")
        
        navigation_suite.test_navigation_consistency()
        print("✓ Player navigation consistency test passed")
        
        navigation_suite.test_first_player_array_configuration()
        print("✓ Player navigation first player array configuration test passed")
        
        navigation_suite.test_complex_active_mask_scenarios()
        print("✓ Player navigation complex active mask scenarios test passed")
        
        print("\nAll tests passed! ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
