import jax
import jax.numpy as jnp

from pgx import universal_poker


class TestGetLegalActions:
    """Test suite specifically for the _get_legal_actions method of UniversalPoker."""
    
    def test_basic_legal_actions_preflop(self):
        """Test basic legal actions in preflop scenario."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # In a fresh game, current player should be able to fold, call, and potentially raise
        legal_actions = env._get_legal_actions(state)
        current_player = state.current_player
        
        # Player should not be all-in initially, so can fold
        assert legal_actions[universal_poker.FOLD] == True, "Should be able to fold in normal state"
        
        # Player should be able to call if they have chips and bet is less than max_bet
        expected_can_call = (state.bets[current_player] <= state.max_bet) and (state.stacks[current_player] > 0)
        assert legal_actions[universal_poker.CALL] == expected_can_call, "Call availability should match expected"
        
        # Player should be able to raise if total chips > max_bet
        total_chips = state.stacks[current_player] + state.bets[current_player]
        expected_can_raise = total_chips > state.max_bet
        assert legal_actions[universal_poker.RAISE] == expected_can_raise, "Raise availability should match expected"
    
    def test_legal_actions_all_in_player(self):
        """Test legal actions for a player who is already all-in."""
        config_str = """GAMEDEF
numplayers = 2
stack = 5 50
blind = 2 5
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Force player 0 to be all-in by setting stack to 0
        state = state.replace(
            stacks=state.stacks.at[0].set(0),
            all_in=state.all_in.at[0].set(True)
        )
        state = state.replace(current_player=0)
        
        legal_actions = env._get_legal_actions(state)
        
        # All-in player should not be able to fold (they're committed)
        assert legal_actions[universal_poker.FOLD] == False, "All-in player should not be able to fold"
        
        # All-in player should not be able to call (no chips left)
        assert legal_actions[universal_poker.CALL] == False, "All-in player should not be able to call"
        
        # All-in player should not be able to raise (no chips left)
        assert legal_actions[universal_poker.RAISE] == False, "All-in player should not be able to raise"
    
    def test_legal_actions_zero_stack_not_all_in(self):
        """Test legal actions for player with zero stack but not marked all-in."""
        config_str = """GAMEDEF
numplayers = 2
stack = 10 50
blind = 5 10
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Set player 0 stack to 0 but keep all_in as False
        state = state.replace(
            stacks=state.stacks.at[0].set(0),
            current_player=0
        )
        
        legal_actions = env._get_legal_actions(state)
        
        # Player can still fold (not marked all-in)
        assert legal_actions[universal_poker.FOLD] == True, "Player with 0 stack should be able to fold"
        
        # Player cannot call (no chips)
        assert legal_actions[universal_poker.CALL] == False, "Player with 0 stack should not be able to call"
        
        # Player cannot raise (no chips)
        assert legal_actions[universal_poker.RAISE] == False, "Player with 0 stack should not be able to raise"
    
    def test_legal_actions_exact_call_amount(self):
        """Test legal actions when player has exactly enough total chips to equal max_bet."""
        config_str = """GAMEDEF
numplayers = 2
stack = 8 50
blind = 2 8
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Player 0 has 6 chips (8-2), max_bet is 8, so total chips = max_bet
        current_player = 0
        state = state.replace(current_player=current_player)
        
        call_needed = state.max_bet - state.bets[current_player]  # Should be 8 - 2 = 6
        remaining_chips = state.stacks[current_player]  # Should be 6
        
        legal_actions = env._get_legal_actions(state)
        
        # Should be able to fold
        assert legal_actions[universal_poker.FOLD] == True, "Should be able to fold"
        
        # Should be able to call (has enough chips)
        assert legal_actions[universal_poker.CALL] == True, "Should be able to call with exact chips"
        
        # Should not be able to raise (total chips = max_bet, not > max_bet)
        total_chips = state.stacks[current_player] + state.bets[current_player]
        assert total_chips == state.max_bet, f"Total chips {total_chips} should equal max_bet {state.max_bet}"
        assert legal_actions[universal_poker.RAISE] == False, "Should not be able to raise with exact call amount"
    
    def test_legal_actions_insufficient_chips_to_call(self):
        """Test legal actions when player has insufficient chips to call."""
        config_str = """GAMEDEF
numplayers = 2
stack = 10 50
blind = 2 15
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Player 0 has 8 chips (10-2), max_bet is 15, so they can't call the full amount
        current_player = 0
        state = state.replace(current_player=current_player)
        
        legal_actions = env._get_legal_actions(state)
        
        # Should be able to fold
        assert legal_actions[universal_poker.FOLD] == True, "Should be able to fold"
        
        # Should still be able to call (partial call/all-in)
        has_chips = state.stacks[current_player] > 0
        bet_not_exceeding_max = state.bets[current_player] <= state.max_bet
        assert legal_actions[universal_poker.CALL] == (has_chips and bet_not_exceeding_max), "Should be able to call partially"
        
        # Should not be able to raise (insufficient total chips)
        total_chips = state.stacks[current_player] + state.bets[current_player]
        assert total_chips < state.max_bet, f"Total chips {total_chips} should be < max_bet {state.max_bet}"
        assert legal_actions[universal_poker.RAISE] == False, "Should not be able to raise with insufficient chips"
    
    def test_legal_actions_can_raise(self):
        """Test legal actions when player has enough chips to raise."""
        config_str = """GAMEDEF
numplayers = 2
stack = 50 50
blind = 2 5
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Player 0 has 48 chips (50-2), max_bet is 5, so they can raise
        current_player = 0
        state = state.replace(current_player=current_player)
        
        legal_actions = env._get_legal_actions(state)
        
        # Should be able to do all actions
        assert legal_actions[universal_poker.FOLD] == True, "Should be able to fold"
        assert legal_actions[universal_poker.CALL] == True, "Should be able to call"
        
        # Should be able to raise (total chips > max_bet)
        total_chips = state.stacks[current_player] + state.bets[current_player]
        assert total_chips > state.max_bet, f"Total chips {total_chips} should be > max_bet {state.max_bet}"
        assert legal_actions[universal_poker.RAISE] == True, "Should be able to raise with sufficient chips"
    
    def test_legal_actions_terminated_game(self):
        """Test legal actions when game is terminated."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Manually terminate the game
        state = state.replace(terminated=True)
        
        legal_actions = env._get_legal_actions(state)
        
        # All actions should be True when game is terminated per pgx step().
        assert legal_actions[universal_poker.FOLD] == True, "Should be able to fold in terminated game"
        assert legal_actions[universal_poker.CALL] == True, "Should be able to call in terminated game"
        assert legal_actions[universal_poker.RAISE] == True, "Should be able to raise in terminated game"
    
    def test_legal_actions_edge_case_bet_equals_max_bet(self):
        """Test legal actions when player's current bet equals max_bet."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        current_player = state.current_player
        
        # Set player's bet equal to max_bet (already called)
        state = state.replace(
            bets=state.bets.at[current_player].set(state.max_bet)
        )
        
        legal_actions = env._get_legal_actions(state)
        
        # Should be able to fold
        assert legal_actions[universal_poker.FOLD] == True, "Should be able to fold even when bet equals max_bet"
        
        # Should be able to call (no additional chips needed, but action is valid)
        assert legal_actions[universal_poker.CALL] == True, "Should be able to call when bet equals max_bet"
        
        # Raise depends on having more total chips than max_bet
        total_chips = state.stacks[current_player] + state.bets[current_player]
        expected_can_raise = total_chips > state.max_bet
        assert legal_actions[universal_poker.RAISE] == expected_can_raise, "Raise should depend on total chips vs max_bet"
    
    def test_legal_actions_edge_case_bet_exceeds_max_bet(self):
        """Test legal actions when player's bet somehow exceeds max_bet (edge case)."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        current_player = state.current_player
        
        # Set player's bet to exceed max_bet (shouldn't happen in normal gameplay)
        state = state.replace(
            bets=state.bets.at[current_player].set(state.max_bet + 5)
        )
        
        legal_actions = env._get_legal_actions(state)
        
        # Should still be able to fold
        assert legal_actions[universal_poker.FOLD] == True, "Should be able to fold even when bet exceeds max_bet"
        
        # Should not be able to call (bet already exceeds max_bet)
        bet_not_exceeding_max = state.bets[current_player] <= state.max_bet
        assert bet_not_exceeding_max == False, "Bet should exceed max_bet in this test"
        assert legal_actions[universal_poker.CALL] == False, "Should not be able to call when bet exceeds max_bet"
        
        # Raise depends on total chips vs max_bet
        total_chips = state.stacks[current_player] + state.bets[current_player]
        expected_can_raise = total_chips > state.max_bet
        assert legal_actions[universal_poker.RAISE] == expected_can_raise, "Raise should depend on total chips vs max_bet"
    
    def test_legal_actions_multiple_scenarios(self):
        """Test legal actions across multiple game scenarios."""
        scenarios = [
            # (stack_sizes, blinds, expected_description)
            ([100, 100], [1, 2], "normal game"),
            ([5, 100], [1, 2], "short stack"),
            ([2, 100], [1, 2], "very short stack"),
            ([50, 50], [10, 20], "high blinds"),
        ]
        
        for stacks, blinds, description in scenarios:
            config_str = f"""GAMEDEF
numplayers = 2
stack = {stacks[0]} {stacks[1]}
blind = {blinds[0]} {blinds[1]}
END GAMEDEF"""
            env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
            key = jax.random.PRNGKey(42)
            state = env.init(key)
            
            # Test for each player
            for player_id in range(2):
                state = state.replace(current_player=player_id)
                legal_actions = env._get_legal_actions(state)
                
                # Verify logic consistency
                current_player = state.current_player
                can_fold = ~state.all_in[current_player]
                can_call = (state.bets[current_player] <= state.max_bet) & (state.stacks[current_player] > 0)
                total_chips = state.stacks[current_player] + state.bets[current_player]
                can_raise = total_chips > state.max_bet
                
                assert legal_actions[universal_poker.FOLD] == can_fold, f"Fold logic failed for {description}, player {player_id}"
                assert legal_actions[universal_poker.CALL] == can_call, f"Call logic failed for {description}, player {player_id}"
                assert legal_actions[universal_poker.RAISE] == can_raise, f"Raise logic failed for {description}, player {player_id}"
    
    def test_legal_actions_return_type_and_shape(self):
        """Test that _get_legal_actions returns correct type and shape."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        legal_actions = env._get_legal_actions(state)
        
        # Should return JAX array
        assert isinstance(legal_actions, jnp.ndarray), "Should return JAX array"
        
        # Should have shape (3,) for [fold, call, raise]
        assert legal_actions.shape == (3,), f"Should have shape (3,), got {legal_actions.shape}"
        
        # Should have boolean dtype
        assert legal_actions.dtype == jnp.bool_, f"Should have boolean dtype, got {legal_actions.dtype}"
        
        # All values should be boolean
        for i, action_legal in enumerate(legal_actions):
            assert isinstance(bool(action_legal), bool), f"Action {i} should be boolean"


if __name__ == "__main__":
    # Run TestGetLegalActions tests
    legal_actions_suite = TestGetLegalActions()
    
    print("Running Get Legal Actions tests...")
    
    try:
        legal_actions_suite.test_basic_legal_actions_preflop()
        print("✓ Basic legal actions preflop test passed")
        
        legal_actions_suite.test_legal_actions_all_in_player()
        print("✓ Legal actions all-in player test passed")
        
        legal_actions_suite.test_legal_actions_zero_stack_not_all_in()
        print("✓ Legal actions zero stack not all-in test passed")
        
        legal_actions_suite.test_legal_actions_exact_call_amount()
        print("✓ Legal actions exact call amount test passed")
        
        legal_actions_suite.test_legal_actions_insufficient_chips_to_call()
        print("✓ Legal actions insufficient chips to call test passed")
        
        legal_actions_suite.test_legal_actions_can_raise()
        print("✓ Legal actions can raise test passed")
        
        legal_actions_suite.test_legal_actions_terminated_game()
        print("✓ Legal actions terminated game test passed")
        
        legal_actions_suite.test_legal_actions_edge_case_bet_equals_max_bet()
        print("✓ Legal actions bet equals max bet test passed")
        
        legal_actions_suite.test_legal_actions_edge_case_bet_exceeds_max_bet()
        print("✓ Legal actions bet exceeds max bet test passed")
        
        legal_actions_suite.test_legal_actions_multiple_scenarios()
        print("✓ Legal actions multiple scenarios test passed")
        
        legal_actions_suite.test_legal_actions_return_type_and_shape()
        print("✓ Legal actions return type and shape test passed")
        
        print("\nAll Get Legal Actions tests passed! ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
