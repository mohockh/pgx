import jax
import jax.numpy as jnp

from pgx import universal_poker


class TestApplyAction:
    """Test suite specifically for the _apply_action method of UniversalPoker."""
    
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
        
    def test_bet_tracking_accuracy(self):
        """Test that bet amounts are tracked accurately across multiple actions."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Track initial bets (blinds)
        initial_bets = state.bets.copy()
        initial_pot = state.pot
        
        # Player calls
        current_player = state.current_player
        call_amount = state.max_bet - state.bets[current_player]
        state = env._apply_action(state, universal_poker.CALL)
        
        # Verify bet tracking
        assert state.bets[current_player] == state.max_bet, "Bet should equal max_bet after call"
        assert state.pot == initial_pot + call_amount, "Pot should increase by call amount"
        
        # Switch to other player and raise
        other_player = 1 - current_player
        state = state.replace(current_player=other_player)
        pre_raise_bet = state.bets[other_player]
        pre_raise_pot = state.pot
        
        state = env._apply_action(state, universal_poker.RAISE)
        
        # Verify raise tracking
        raise_amount = state.bets[other_player] - pre_raise_bet
        assert state.pot == pre_raise_pot + raise_amount, "Pot should increase by raise amount"
        assert state.max_bet == state.bets[other_player], "Max bet should equal raiser's total bet"
        
    def test_last_raiser_tracking(self):
        """Test that last_raiser is correctly tracked."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Initially no raiser
        initial_last_raiser = state.last_raiser
        
        # Player folds - should not change last_raiser
        current_player = state.current_player
        state = env._apply_action(state, universal_poker.FOLD)
        assert state.last_raiser == initial_last_raiser, "Fold should not change last_raiser"
        
        # Reset and test call - should not change last_raiser
        state = env.init(key)
        current_player = state.current_player
        state = env._apply_action(state, universal_poker.CALL)
        assert state.last_raiser == initial_last_raiser, "Call should not change last_raiser"
        
        # Reset and test raise - should update last_raiser
        state = env.init(key)
        current_player = state.current_player
        state = env._apply_action(state, universal_poker.RAISE)
        assert state.last_raiser == current_player, "Raise should set last_raiser to current player"
        
    def test_edge_case_minimum_stack(self):
        """Test behavior with extremely small stacks."""
        config_str = """GAMEDEF
numplayers = 2
stack = 2 3
blind = 1 2
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Player 0 has 1 chip left (2-1), Player 1 has 1 chip left (3-2)
        assert state.stacks[0] == 1, "Player 0 should have 1 chip"
        assert state.stacks[1] == 1, "Player 1 should have 1 chip"
        
        # Current player (should be player 0) needs to call 1 more to match max_bet of 2
        current_player = state.current_player
        
        # Test call - should go all-in
        new_state = env._apply_action(state, universal_poker.CALL)
        assert new_state.stacks[current_player] == 0, "Player should go all-in on call"
        assert new_state.all_in[current_player] == True, "Player should be all-in"
        
        # Test raise (should also be all-in) on fresh state
        new_state = env._apply_action(state, universal_poker.RAISE)
        assert new_state.stacks[current_player] == 0, "Player should go all-in on raise"
        assert new_state.all_in[current_player] == True, "Player should be all-in"
        
    def test_max_bet_calculation_edge_cases(self):
        """Test max_bet calculation in various edge cases."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Case 1: Normal raise doubles max_bet
        initial_max_bet = state.max_bet
        current_player = state.current_player
        new_state = env._apply_action(state, universal_poker.RAISE)
        expected_new_max = initial_max_bet * 2
        assert new_state.max_bet == expected_new_max, f"Max bet should be {expected_new_max}"
        
        # Case 2: Zero max_bet uses min blind
        state = state.replace(max_bet=jnp.int32(0))
        max_blind = jnp.max(state.bets)
        new_state = env._apply_action(state, universal_poker.RAISE)
        assert new_state.max_bet == max_blind, f"Max bet should be {max_blind} when starting from 0"
        
        # Case 3: All-in raise with insufficient chips for full double
        config_str = """GAMEDEF
numplayers = 2
stack = 10 50
blind = 1 2
END GAMEDEF"""
        env2 = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        state2 = env2.init(key)
        
        # Force a high max_bet that player 0 can't double
        state2 = state2.replace(max_bet=jnp.int32(8), current_player=0)
        
        new_state2 = env2._apply_action(state2, universal_poker.RAISE)
        # Player 0 should go all-in with their remaining 9 chips
        assert new_state2.stacks[0] == 0, "Player should go all-in"
        assert new_state2.all_in[0] == True, "Player should be all-in"
        
    def test_action_count_increment(self):
        """Test that num_actions_this_round increments correctly."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        initial_count = state.num_actions_this_round
        
        # Each action should increment the counter
        state = env._apply_action(state, universal_poker.CALL)
        assert state.num_actions_this_round == initial_count + 1, "Action count should increment"
        
        state = state.replace(current_player=1)
        state = env._apply_action(state, universal_poker.FOLD)
        assert state.num_actions_this_round == initial_count + 2, "Action count should increment again"
        
    def test_state_consistency_after_action(self):
        """Test that the state remains consistent after each action."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Check initial consistency
        initial_total_chips = jnp.sum(state.stacks) + state.pot
        
        # Apply various actions and check consistency
        actions_to_test = [universal_poker.CALL, universal_poker.RAISE, universal_poker.FOLD]
        
        for action in actions_to_test:
            test_state = env.init(key)  # Fresh state for each test
            current_player = test_state.current_player
            
            # Skip invalid actions
            if action == universal_poker.FOLD and test_state.all_in[current_player]:
                continue
                
            new_state = env._apply_action(test_state, action)
            
            # Check total chips conservation (except for fold, which doesn't move chips)
            new_total_chips = jnp.sum(new_state.stacks) + new_state.pot
            assert new_total_chips == initial_total_chips, f"Chip conservation failed for action {action}"
            
            # Check player count consistency
            assert new_state.num_players == test_state.num_players, "Player count should not change"
            
            # Check array sizes
            assert len(new_state.stacks) == len(test_state.stacks), "Stack array size should not change"
            assert len(new_state.bets) == len(test_state.bets), "Bet array size should not change"
            assert len(new_state.folded) == len(test_state.folded), "Folded array size should not change"


if __name__ == "__main__":
    # Run TestApplyAction tests
    apply_action_suite = TestApplyAction()
    
    print("Running Apply Action tests...")
    
    try:
        apply_action_suite.test_fold_action_basic()
        print("✓ Apply action fold basic test passed")
        
        apply_action_suite.test_call_action_basic()
        print("✓ Apply action call basic test passed")
        
        apply_action_suite.test_call_action_insufficient_chips()
        print("✓ Apply action call insufficient chips test passed")
        
        apply_action_suite.test_raise_action_basic()
        print("✓ Apply action raise basic test passed")
        
        apply_action_suite.test_raise_action_zero_max_bet()
        print("✓ Apply action raise zero max bet test passed")
        
        apply_action_suite.test_raise_action_insufficient_chips_all_in()
        print("✓ Apply action raise insufficient chips all-in test passed")
        
        apply_action_suite.test_all_in_detection()
        print("✓ Apply action all-in detection test passed")
        
        apply_action_suite.test_active_mask_updates()
        print("✓ Apply action active mask updates test passed")
        
        apply_action_suite.test_bet_tracking_accuracy()
        print("✓ Apply action bet tracking accuracy test passed")
        
        apply_action_suite.test_last_raiser_tracking()
        print("✓ Apply action last raiser tracking test passed")
        
        apply_action_suite.test_edge_case_minimum_stack()
        print("✓ Apply action edge case minimum stack test passed")
        
        apply_action_suite.test_max_bet_calculation_edge_cases()
        print("✓ Apply action max bet calculation edge cases test passed")
        
        apply_action_suite.test_action_count_increment()
        print("✓ Apply action count increment test passed")
        
        apply_action_suite.test_state_consistency_after_action()
        print("✓ Apply action state consistency test passed")
        
        print("\nAll Apply Action tests passed! ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise