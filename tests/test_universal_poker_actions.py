import jax
import jax.numpy as jnp

from pgx import universal_poker


class TestUniversalPokerActions:
    """Test suite for Universal Poker player actions - legal actions, action application, and validation."""
    
    # Legal Actions Tests (from test_universal_poker_get_legal_actions.py)
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
    
    def test_legal_actions_insufficient_chips_to_call_big_blind(self):
        """Test legal actions when player has insufficient chips to call the big blind."""
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
        
        # Should not be able to do anything, folded
        assert state.folded[current_player] == True, "Should be auto-folded at game start, not enough chips for big blind"
        assert legal_actions[universal_poker.FOLD] == False, "Should not be able to fold"
        assert legal_actions[universal_poker.CALL] == False, "Should not be able to fold"
        assert legal_actions[universal_poker.RAISE] == False, "Should not be able to fold"
        assert state.round == 0, "Game should be terminated in first round"
        assert state.terminated == True, "Only one player left, game is over"
    
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

    def test_legal_actions_preflop(self):
        """Test legal actions in preflop."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # First player to act should be able to fold, call, or raise
        legal_actions = state.legal_action_mask
        assert legal_actions[universal_poker.FOLD]   # Can fold
        assert legal_actions[universal_poker.CALL]   # Can call
        assert legal_actions[universal_poker.RAISE]  # Can raise

    # Action Application Tests (from test_universal_poker_apply_action.py)
    def test_fold_action_basic(self):
        """Test basic fold action updates state correctly."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        current_player = state.current_player
        initial_pot = state.pot
        initial_stack = state.stacks[current_player]
        initial_bet = state.bets[current_player]
        
        # Apply fold action directly
        new_state = env._apply_action(state, universal_poker.FOLD)
        
        # Verify fold updates
        assert new_state.folded[current_player] == True, "Player should be folded"
        assert new_state.stacks[current_player] == initial_stack, "Stack should not change on fold"
        assert new_state.bets[current_player] == initial_bet, "Bet should not change on fold"
        assert new_state.pot == initial_pot, "Pot should not change on fold"
        assert new_state.all_in[current_player] == False, "Player should not be all-in from fold"
        assert new_state.num_actions_this_round == state.num_actions_this_round + 1, "Action count should increment"
        
        # Verify active mask is updated
        assert new_state.active_mask[current_player] == False, "Folded player should not be active"
        
    def test_call_action_basic(self):
        """Test basic call action with sufficient chips."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        current_player = state.current_player
        initial_stack = state.stacks[current_player]
        initial_bet = state.bets[current_player]
        call_amount = state.max_bet - initial_bet
        
        # Apply call action directly
        new_state = env._apply_action(state, universal_poker.CALL)
        
        # Verify call updates
        assert new_state.folded[current_player] == False, "Player should not be folded"
        assert new_state.bets[current_player] == state.max_bet, "Bet should equal max_bet after call"
        assert new_state.stacks[current_player] == initial_stack - call_amount, "Stack should decrease by call amount"
        assert new_state.pot == state.pot + call_amount, "Pot should increase by call amount"
        assert new_state.max_bet == state.max_bet, "Max bet should not change on call"
        assert new_state.last_raiser == state.last_raiser, "Last raiser should not change on call"
        
    def test_call_action_insufficient_chips(self):
        """Test call action when player has insufficient chips (partial call/all-in)."""
        config_str = """GAMEDEF
numplayers = 2
stack = 10 50
blind = 1 5
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Player 0 has 9 chips (10-1), needs to call 4 more to match max_bet of 5
        # But then player 1 raises, making it impossible for player 0 to call fully
        
        # First, player 1 raises to make max_bet = 10
        state = state.replace(current_player=1)
        state = env._apply_action(state, universal_poker.RAISE)
        
        # Now player 0 tries to call but has insufficient chips
        state = state.replace(current_player=0)
        initial_stack = state.stacks[0]  # Should be 9
        initial_bet = state.bets[0]      # Should be 1
        max_bet = state.max_bet          # Should be 10
        
        new_state = env._apply_action(state, universal_poker.CALL)
        
        # Player should go all-in with remaining chips
        assert new_state.stacks[0] == 0, "Player should have 0 chips left"
        assert new_state.all_in[0] == True, "Player should be all-in"
        assert new_state.bets[0] == initial_bet + initial_stack, "Bet should include all remaining chips"
        assert new_state.pot == state.pot + initial_stack, "Pot should increase by player's remaining chips"
        
    def test_raise_action_basic(self):
        """Test basic raise action with sufficient chips."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        current_player = state.current_player
        initial_stack = state.stacks[current_player]
        initial_bet = state.bets[current_player]
        initial_max_bet = state.max_bet
        
        # Apply raise action directly
        new_state = env._apply_action(state, universal_poker.RAISE)
        
        # Verify raise updates
        assert new_state.folded[current_player] == False, "Player should not be folded"
        assert new_state.max_bet > initial_max_bet, "Max bet should increase after raise"
        assert new_state.bets[current_player] > initial_bet, "Player's bet should increase"
        assert new_state.stacks[current_player] < initial_stack, "Stack should decrease"
        assert new_state.pot > state.pot, "Pot should increase"
        assert new_state.last_raiser == current_player, "Last raiser should be current player"
        
        # Verify raise amount calculation
        expected_raise_target = initial_max_bet * 2
        assert new_state.bets[current_player] == expected_raise_target, f"Raise should be to {expected_raise_target}"
        
    def test_raise_action_zero_max_bet(self):
        """Test raise action when max_bet is 0 (uses min blind as reference)."""
        # Create a scenario where max_bet becomes 0 (post-flop)
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Manually set max_bet to 0 and see what happens
        state = state.replace(max_bet=jnp.int32(0))
        
        current_player = state.current_player
        initial_stack = state.stacks[current_player]
        max_blind = jnp.max(state.bets)  # Should be 2 (big blind)
        
        new_state = env._apply_action(state, universal_poker.RAISE)
        
        # When max_bet is 0, raise should use max blind as minimum
        expected_raise_target = max_blind
        assert new_state.bets[current_player] == expected_raise_target, f"Raise should be to {expected_raise_target} when max_bet is 0"
        assert new_state.max_bet == expected_raise_target, "Max bet should be updated to raise amount"
        
    def test_raise_action_insufficient_chips_all_in(self):
        """Test raise action when player has insufficient chips for full raise (goes all-in)."""
        config_str = """GAMEDEF
numplayers = 2
stack = 10 100
blind = 1 8
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Player 0 has 9 chips (10-1), max_bet is 8
        # Normal raise would be to 16, requiring 15 more chips 
        # Player only has 9 chips total, so they should go all-in
        
        current_player = 0  # Make sure we're testing player 0
        state = state.replace(current_player=current_player)
        initial_stack = state.stacks[current_player]
        
        new_state = env._apply_action(state, universal_poker.RAISE)
        
        # Player should go all-in since they can't make full raise
        assert new_state.stacks[current_player] == 0, "Player should have 0 chips left after all-in raise"
        assert new_state.all_in[current_player] == True, "Player should be all-in"
        assert new_state.bets[current_player] == 1 + initial_stack, "Bet should include all chips"
        assert new_state.max_bet >= new_state.bets[current_player], "Max bet should be at least player's total bet"
        
    def test_all_in_detection(self):
        """Test that all_in status is correctly set when stack reaches 0."""
        config_str = """GAMEDEF
numplayers = 2
stack = 3 50
blind = 1 2
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Player 0 has 2 chips left (3-1), max_bet is 2
        # Calling requires 1 more chip to reach max_bet of 2
        # Then let's have them try to call after a raise
        
        # First player 1 raises
        state = state.replace(current_player=1)
        state = env._apply_action(state, universal_poker.RAISE)
        
        # Now player 0 tries to call with only 2 chips remaining
        current_player = 0
        state = state.replace(current_player=current_player)
        
        new_state = env._apply_action(state, universal_poker.CALL)
        
        # Should be all-in (player called with all remaining chips)
        assert new_state.stacks[current_player] == 0, "Stack should be 0"
        assert new_state.all_in[current_player] == True, "Should be marked all-in"
        assert new_state.active_mask[current_player] == False, "All-in player should not be active"
        # Player had 2 chips, bet 1 initially, so final bet should be 3 (1 + 2)
        assert new_state.bets[current_player] == 3, "Player should have bet all their chips (1 initial + 2 remaining)"
        
    def test_active_mask_updates(self):
        """Test that active_mask is correctly updated for different actions."""
        env = universal_poker.UniversalPoker(num_players=3)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Initially all players should be active
        assert jnp.sum(state.active_mask) == 3, "All 3 players should be active initially"
        
        # Fold player 0
        state = state.replace(current_player=0)
        state = env._apply_action(state, universal_poker.FOLD)
        assert state.active_mask[0] == False, "Folded player should not be active"
        assert jnp.sum(state.active_mask) == 2, "Should have 2 active players after fold"
        
        # Test that calling doesn't affect active status
        state = state.replace(current_player=1)
        initial_active_count = jnp.sum(state.active_mask)
        state = env._apply_action(state, universal_poker.CALL)
        assert jnp.sum(state.active_mask) == initial_active_count, "Call should not change active count"
        assert state.active_mask[1] == True, "Calling player should remain active"

    def test_fold_action(self):
        """Test folding action."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Player folds
        current_player = state.current_player
        new_state = env.step(state, universal_poker.FOLD)
        
        assert new_state.folded[current_player]
        assert new_state.terminated  # Game should end with fold in 2-player game


if __name__ == "__main__":
    import sys
    import traceback
    
    test_suite = TestUniversalPokerActions()
    
    print("Running Universal Poker actions tests...")
    
    # Get all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
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