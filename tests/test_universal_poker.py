import jax
import jax.numpy as jnp

from pgx import universal_poker


class TestUniversalPoker:
    """Test suite for Universal Poker implementation - unique tests only."""
    
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
            # Winner should be among the non-folded players
            assert sum(float(r) for r in state3.rewards) > 0, "Total rewards should be positive"
            
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
            assert sum(float(r) for r in state4.rewards) > 0, "Total rewards should be positive"

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
        active_players = jnp.sum(~state.folded[:state.num_players])
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

    def test_correct_raise_amounts(self):
        """Test that raise amounts are calculated correctly based on betting deltas, not total bets."""
        # Set up a clear scenario: Player 0 bets 1, Player 1 bets 2
        # Player 0's minimum raise should be 2 + (2-1) = 3, NOT 2*2 = 4
        config_str = """GAMEDEF
numplayers = 3
stack = 100 100 100
blind = 1 2 0
END GAMEDEF"""
        
        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Initial: P0=1 (SB), P1=2 (BB), P2=0, max_bet=2
        assert state.bets[0] == 1, "Player 0 should have small blind"
        assert state.bets[1] == 2, "Player 1 should have big blind" 
        assert state.bets[2] == 0, "Player 2 should have no blind"
        assert state.max_bet == 2, "Max bet should be big blind amount"
        
        # Player 2 calls (puts in 2)
        state = env.step(state, universal_poker.CALL)
        assert state.bets[2] == 2, "Player 2 should have called to 2"
        assert state.max_bet == 2, "Max bet should still be 2"
        
        # Player 0 raises - this is the key test
        # Current bet structure: P0=1, P1=2, P2=2, max_bet=2
        # The last raise was from 0->2, so raise amount was 2
        # Minimum next raise should be 2 + 2 = 4 (NOT 2*2=4 by coincidence, but 2+2 by correct logic)
        # So Player 0 needs to bet 4 total (was 1, needs to add 3)
        state = env.step(state, universal_poker.RAISE)
        
        # Test the correct behavior
        assert state.max_bet == 4, f"Max bet should be 4 (2 + minimum raise of 2), got {state.max_bet}"
        assert state.bets[0] == 4, f"Player 0 should have total bet of 4, got {state.bets[0]}"
        
        # Additional test: Next player raises again
        # Current: max_bet=4, last raise was from 2->4 (delta=2)  
        # Next minimum raise: 4 + 2 = 6
        current_player = state.current_player
        if current_player == 1:  # Player 1's turn
            initial_bet_p1 = state.bets[1] 
            state = env.step(state, universal_poker.RAISE)
            expected_min_raise = 4 + 2  # 6
            assert state.max_bet == expected_min_raise, f"After second raise, max_bet should be {expected_min_raise}, got {state.max_bet}"
            assert state.bets[1] == expected_min_raise, f"Player 1 should have total bet of {expected_min_raise}, got {state.bets[1]}"
            

if __name__ == "__main__":
    # Run unique tests only
    test_suite = TestUniversalPoker()
    
    print("Running Universal Poker unique tests...")
    
    try:
        test_suite.test_rewards_array_size_matches_num_players()
        print("✓ Rewards array size matches num players test passed")
        
        test_suite.test_early_termination_no_advancement()
        print("✓ Early termination no advancement test passed")
        
        test_suite.test_correct_raise_amounts()
        print("✓ Correct raise amounts test passed")
        
        print("\nAll unique tests passed! ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise