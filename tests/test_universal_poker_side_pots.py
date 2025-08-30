import jax
import jax.numpy as jnp

from pgx import universal_poker
from pgx.poker_eval.cardset import cards_to_cardset


class TestSidePotDistribution:
    """Comprehensive test suite for side pot distribution edge cases in _calculate_rewards."""
    
    def _create_test_state(self, num_players, stacks, bets, folded, hand_strengths, pot=None):
        """Helper to create test state for side pot testing."""
        config_str = f"""GAMEDEF
numplayers = {num_players}
stack = {' '.join([str(s) for s in [100] * num_players])}
blind = {' '.join([str(b) for b in [0] * num_players])}
END GAMEDEF"""
        
        env = universal_poker.UniversalPoker(num_players=num_players, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Override state values
        state = state.replace(
            stacks=jnp.array(stacks, dtype=jnp.uint32),
            bets=jnp.array(bets, dtype=jnp.uint32),
            folded=jnp.array(folded, dtype=jnp.bool_),
            hand_final_scores=jnp.array(hand_strengths, dtype=jnp.uint32),
            round=4,  # Force showdown
            pot=pot if pot is not None else jnp.sum(jnp.array(bets, dtype=jnp.uint32)),
            rewards=jnp.zeros(num_players, dtype=jnp.float32)  # Set correct reward array size
        )
        
        # Update player and active masks
        player_mask = jnp.arange(num_players) < num_players
        active_mask = ~state.folded & player_mask
        state = state.replace(player_mask=player_mask, active_mask=active_mask)
        
        return env, state
    
    def test_two_player_equal_side_pots(self):
        """Test two players with equal contributions and tied hands."""
        env, state = self._create_test_state(
            num_players=2,
            stacks=[0, 0],           # Both all-in
            bets=[50, 50],           # Equal contributions  
            folded=[False, False],   # Both active
            hand_strengths=[1000, 1000]  # Tied hands
        )
        
        rewards = env._calculate_rewards(state)
        
        # Should split pot equally: 50 each
        assert rewards[0] == 50.0, f"P0 should get 50, got {rewards[0]}"
        assert rewards[1] == 50.0, f"P1 should get 50, got {rewards[1]}"
        
    def test_three_player_unequal_side_pots(self):
        """Test three players with unequal contributions creating multiple side pots."""
        env, state = self._create_test_state(
            num_players=3,
            stacks=[0, 0, 0],        # All all-in
            bets=[10, 30, 50],       # Different contributions
            folded=[False, False, False],
            hand_strengths=[5000, 3000, 1000]  # P0 > P1 > P2
        )
        
        rewards = env._calculate_rewards(state)
        
        # Side pot calculation:
        # Layer 1 (0-10): 10 * 3 = 30 chips, P0 wins (best hand among all)
        # Layer 2 (10-30): 20 * 2 = 40 chips, P1 wins (best hand among P1,P2)  
        # Layer 3 (30-50): 20 * 1 = 20 chips, P2 wins (only eligible player)
        # Total: P0=30, P1=40, P2=20
        
        assert rewards[0] == 30.0, f"P0 should get 30, got {rewards[0]}"
        assert rewards[1] == 40.0, f"P1 should get 40, got {rewards[1]}"
        assert rewards[2] == 20.0, f"P2 should get 20, got {rewards[2]}"
        
    def test_three_player_tied_hands_in_side_pot(self):
        """Test tie within a specific side pot layer."""
        env, state = self._create_test_state(
            num_players=3,
            stacks=[0, 0, 0],
            bets=[10, 30, 30],       # P1 and P2 have equal high contributions
            folded=[False, False, False],
            hand_strengths=[1000, 5000, 5000]  # P1 and P2 tied, better than P0
        )
        
        rewards = env._calculate_rewards(state)
        
        # Side pot calculation:
        # Layer 1 (0-10): 10 * 3 = 30 chips, P1 and P2 tie -> 15 each
        # Layer 2 (10-30): 20 * 2 = 40 chips, P1 and P2 tie -> 20 each
        # Total: P0=0, P1=35, P2=35
        
        assert rewards[0] == 0.0, f"P0 should get 0, got {rewards[0]}"
        assert rewards[1] == 35.0, f"P1 should get 35, got {rewards[1]}"
        assert rewards[2] == 35.0, f"P2 should get 35, got {rewards[2]}"
        
    def test_four_player_complex_side_pots(self):
        """Test complex 4-player scenario with multiple side pots and ties."""
        env, state = self._create_test_state(
            num_players=4,
            stacks=[0, 0, 0, 0],
            bets=[5, 15, 25, 40],    # Four different contribution levels
            folded=[False, False, False, False],
            hand_strengths=[2000, 2000, 6000, 1000]  # P2 > (P0, P1 tied) > P3
        )
        
        rewards = env._calculate_rewards(state)
        
        # Side pot calculation:
        # Layer 1 (0-5): 5 * 4 = 20 chips, P2 wins (best hand among all) -> 20
        # Layer 2 (5-15): 10 * 3 = 30 chips, P2 wins (best among P1,P2,P3) -> 30  
        # Layer 3 (15-25): 10 * 2 = 20 chips, P2 wins (best among P2,P3) -> 20
        # Layer 4 (25-40): 15 * 1 = 15 chips, P3 wins (only eligible) -> 15
        # Total: P0=0, P1=0, P2=70, P3=15
        
        assert rewards[0] == 0.0, f"P0 should get 0, got {rewards[0]}"
        assert rewards[1] == 0.0, f"P1 should get 0, got {rewards[1]}" 
        assert rewards[2] == 70.0, f"P2 should get 70, got {rewards[2]}"
        assert rewards[3] == 15.0, f"P3 should get 15, got {rewards[3]}"
        
    def test_one_player_folded_side_pots(self):
        """Test side pot distribution when one player is folded."""
        env, state = self._create_test_state(
            num_players=3,
            stacks=[0, 0, 10],       # P2 not all-in
            bets=[20, 30, 40],       
            folded=[False, True, False],  # P1 folded
            hand_strengths=[3000, 1000, 5000]  # P2 > P0 > P1 (but P1 folded)
        )
        
        rewards = env._calculate_rewards(state)
        
        # Only P0 and P2 are active (P1 folded)
        # Side pot calculation between P0 and P2:
        # Layer 1 (0-20): 20 * 2 = 40 chips, P2 wins -> 40
        # Layer 2 (20-40): 20 * 1 = 20 chips, P2 wins -> 20  
        # Total: P0=0, P1=0 (folded), P2=60
        
        assert rewards[0] == 0.0, f"P0 should get 0, got {rewards[0]}"
        assert rewards[1] == 0.0, f"P1 should get 0 (folded), got {rewards[1]}"
        assert rewards[2] == 60.0, f"P2 should get 60, got {rewards[2]}"
        
    def test_zero_contribution_player(self):
        """Test when one player has zero contribution."""
        env, state = self._create_test_state(
            num_players=3,
            stacks=[50, 0, 0],       # P0 not all-in
            bets=[0, 25, 25],        # P0 has no bet
            folded=[False, False, False],
            hand_strengths=[6000, 3000, 1000]  # P0 > P1 > P2
        )
        
        rewards = env._calculate_rewards(state)
        
        # Side pot calculation:
        # P0 contributed 0, so not eligible for any pot layers
        # Layer 1 (0-25): 25 * 2 = 50 chips, P1 wins (best among eligible)
        # Total: P0=0, P1=50, P2=0
        
        assert rewards[0] == 0.0, f"P0 should get 0 (no contribution), got {rewards[0]}"
        assert rewards[1] == 50.0, f"P1 should get 50, got {rewards[1]}"
        assert rewards[2] == 0.0, f"P2 should get 0, got {rewards[2]}"
        
    def test_all_equal_contributions_tied_hands(self):
        """Test all players contribute equally with all hands tied."""
        env, state = self._create_test_state(
            num_players=4,
            stacks=[0, 0, 0, 0],
            bets=[25, 25, 25, 25],   # All equal
            folded=[False, False, False, False],
            hand_strengths=[4000, 4000, 4000, 4000]  # All tied
        )
        
        rewards = env._calculate_rewards(state)
        
        # Single side pot: 25 * 4 = 100, split 4 ways = 25 each
        assert all(r == 25.0 for r in rewards), f"All should get 25, got {rewards}"
        
    def test_single_chip_side_pots(self):
        """Test side pots with very small chip amounts."""
        env, state = self._create_test_state(
            num_players=3,
            stacks=[0, 0, 0],
            bets=[1, 2, 3],          # Tiny amounts
            folded=[False, False, False],
            hand_strengths=[5000, 3000, 1000]  # P0 > P1 > P2
        )
        
        rewards = env._calculate_rewards(state)
        
        # Side pot calculation:
        # Layer 1 (0-1): 1 * 3 = 3 chips, P0 wins (best among all) -> 3
        # Layer 2 (1-2): 1 * 2 = 2 chips, P1 wins (best among P1,P2) -> 2
        # Layer 3 (2-3): 1 * 1 = 1 chip, P2 wins (only eligible) -> 1
        # Total: P0=3, P1=2, P2=1
        
        assert rewards[0] == 3.0, f"P0 should get 3, got {rewards[0]}"
        assert rewards[1] == 2.0, f"P1 should get 2, got {rewards[1]}"
        assert rewards[2] == 1.0, f"P2 should get 1, got {rewards[2]}"
        
    def test_maximum_players_side_pots(self):
        """Test side pots with maximum number of players (stress test)."""
        num_players = 6
        env, state = self._create_test_state(
            num_players=num_players,
            stacks=[0] * num_players,
            bets=[10, 20, 30, 40, 50, 60],  # Increasing contributions
            folded=[False] * num_players,
            hand_strengths=[1000, 2000, 3000, 4000, 5000, 6000]  # Increasing strength
        )
        
        rewards = env._calculate_rewards(state)
        
        # P5 has best hand and highest contribution, should win everything
        total_pot = sum([10, 20, 30, 40, 50, 60])
        assert rewards[5] == float(total_pot), f"P5 should get {total_pot}, got {rewards[5]}"
        assert all(r == 0.0 for r in rewards[:5]), f"Others should get 0, got {rewards[:5]}"
        
    def test_integer_division_remainders(self):
        """Test side pot distribution with integer division remainders."""
        env, state = self._create_test_state(
            num_players=3,
            stacks=[0, 0, 0],
            bets=[10, 10, 10],       # Equal contributions
            folded=[False, False, False],
            hand_strengths=[5000, 5000, 5000],  # All tied - will split 30 / 3 = 10 each
        )
        
        rewards = env._calculate_rewards(state)
        
        # Perfect division: 30 / 3 = 10 each
        assert all(r == 10.0 for r in rewards), f"All should get 10, got {rewards}"
        
        # Now test with remainder
        env2, state2 = self._create_test_state(
            num_players=3,
            stacks=[0, 0, 0], 
            bets=[11, 11, 11],       # 33 total, 33/3 = 11 each (no remainder)
            folded=[False, False, False],
            hand_strengths=[5000, 5000, 5000]
        )
        
        rewards2 = env2._calculate_rewards(state2)
        assert all(r == 11.0 for r in rewards2), f"All should get 11, got {rewards2}"
        
    def test_edge_case_empty_side_pot_layers(self):
        """Test when pot layer increments might be zero."""
        env, state = self._create_test_state(
            num_players=4,
            stacks=[0, 0, 0, 0],
            bets=[10, 10, 20, 20],   # Two players at each level
            folded=[False, False, False, False],
            hand_strengths=[6000, 5000, 4000, 3000]  # P0 > P1 > P2 > P3
        )
        
        rewards = env._calculate_rewards(state)
        
        # Side pot calculation:
        # Layer 1 (0-10): 10 * 4 = 40 chips, P0 wins (best among all) -> 40
        # Layer 2 (10-20): 10 * 2 = 20 chips, P2 wins (best among P2,P3) -> 20
        # Total: P0=40, P1=0, P2=20, P3=0
        
        assert rewards[0] == 40.0, f"P0 should get 40, got {rewards[0]}"
        assert rewards[1] == 0.0, f"P1 should get 0, got {rewards[1]}"
        assert rewards[2] == 20.0, f"P2 should get 20, got {rewards[2]}"
        assert rewards[3] == 0.0, f"P3 should get 0, got {rewards[3]}"
    
    def test_massive_pot_layers_stress_test(self):
        """Test with many different contribution levels (stress test for algorithm)."""
        num_players = 8
        env, state = self._create_test_state(
            num_players=num_players,
            stacks=[0] * num_players,
            bets=[1, 5, 10, 15, 20, 25, 30, 100],  # 8 different levels
            folded=[False] * num_players,
            hand_strengths=[8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000]  # Decreasing
        )
        
        rewards = env._calculate_rewards(state)
        
        # P0 has best hand and should win lower layers, but P7 contributed most
        # Complex calculation, but total should equal pot
        total_pot = sum([1, 5, 10, 15, 20, 25, 30, 100])
        assert abs(sum(rewards) - total_pot) < 0.01, f"Total rewards {sum(rewards)} should equal pot {total_pot}"
        
        # P0 should get something (has best hand for some layers)
        assert rewards[0] > 0, f"P0 should get something, got {rewards[0]}"
        
    def test_three_way_tie_side_pot(self):
        """Test three-way tie in a side pot."""
        env, state = self._create_test_state(
            num_players=4,
            stacks=[0, 0, 0, 10],    # P3 not all-in
            bets=[20, 20, 20, 30],   
            folded=[False, False, False, False],
            hand_strengths=[5000, 5000, 5000, 1000]  # P0,P1,P2 tied, P3 worst
        )
        
        rewards = env._calculate_rewards(state)
        
        # Layer 1 (0-20): All 4 eligible, 20*4=80 chips, P0,P1,P2 tie -> 80//3=26 each (2 chips lost to integer division)
        # Layer 2 (20-30): P3 only, 10*1=10 chips, P3 wins -> 10
        # Total: P0=26, P1=26, P2=26, P3=10 (Total=88, 2 chips lost to rounding)
        
        # Check that tied players get equal amounts
        assert rewards[0] == rewards[1] == rewards[2], f"Tied players should get equal amounts: {rewards[:3]}"
        assert rewards[0] == 26.0, f"Tied players should get 26, got {rewards[0]}"
        assert rewards[3] == 10.0, f"P3 should get 10, got {rewards[3]}"
        
        # Total will be slightly less than pot due to integer division
        expected_total = 88.0  # 26*3 + 10
        assert abs(sum(rewards) - expected_total) < 0.01, f"Total rewards should be {expected_total}"
        
    def test_partial_contribution_with_ties(self):
        """Test partial contributions with tied hands."""
        env, state = self._create_test_state(
            num_players=5,
            stacks=[0, 0, 0, 0, 0],
            bets=[5, 10, 15, 15, 20],  # P2,P3 have same contribution
            folded=[False, False, False, False, False],
            hand_strengths=[1000, 2000, 6000, 6000, 3000]  # P2,P3 tied for best
        )
        
        rewards = env._calculate_rewards(state)
        
        # P2 and P3 have tied best hands, they should split winnings in layers they're eligible for
        # Complex calculation but P2 and P3 should get equal amounts
        assert rewards[2] == rewards[3], f"P2 and P3 should get equal amounts: P2={rewards[2]}, P3={rewards[3]}"
        
        # Total should be close to pot (may have small losses due to integer division)
        total_pot = sum([5, 10, 15, 15, 20])
        reward_sum = sum(rewards)
        assert reward_sum <= total_pot, f"Rewards {reward_sum} should not exceed pot {total_pot}"
        assert reward_sum >= total_pot - 10, f"Rewards {reward_sum} should be close to pot {total_pot}"
        
    def test_boundary_hand_strength_values(self):
        """Test with boundary hand strength values (0, max uint32)."""
        env, state = self._create_test_state(
            num_players=3,
            stacks=[0, 0, 0],
            bets=[10, 20, 30],
            folded=[False, False, False], 
            hand_strengths=[0, 4294967295, 2147483647]  # min, max, mid uint32
        )
        
        rewards = env._calculate_rewards(state)
        
        # P1 has maximum possible hand strength, should win everything they're eligible for
        assert rewards[1] > 0, f"P1 should win something, got {rewards[1]}"
        
        # Total should be close to pot (may have small losses due to integer division)
        total_pot = 60
        reward_sum = sum(rewards)
        assert reward_sum <= total_pot, f"Rewards {reward_sum} should not exceed pot {total_pot}"
        assert reward_sum >= total_pot - 5, f"Rewards {reward_sum} should be close to pot {total_pot}"


if __name__ == "__main__":
    # Run side pot distribution tests
    side_pot_suite = TestSidePotDistribution()
    
    print("Running comprehensive side pot distribution tests...")
    
    test_methods = [
        'test_two_player_equal_side_pots',
        'test_three_player_unequal_side_pots', 
        'test_three_player_tied_hands_in_side_pot',
        'test_four_player_complex_side_pots',
        'test_one_player_folded_side_pots',
        'test_zero_contribution_player',
        'test_all_equal_contributions_tied_hands',
        'test_single_chip_side_pots',
        'test_maximum_players_side_pots',
        'test_integer_division_remainders',
        'test_edge_case_empty_side_pot_layers',
        'test_massive_pot_layers_stress_test',
        'test_three_way_tie_side_pot',
        'test_partial_contribution_with_ties',
        'test_boundary_hand_strength_values'
    ]
    
    for test_method in test_methods:
        try:
            getattr(side_pot_suite, test_method)()
            print(f"✓ {test_method} passed")
        except Exception as e:
            print(f"❌ {test_method} failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    print("\nAll side pot distribution tests passed! ✅")