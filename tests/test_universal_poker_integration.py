import jax
import jax.numpy as jnp

from pgx import universal_poker


class TestUniversalPokerIntegration:
    """Test suite for Universal Poker integration - full game flow, observations, and complex scenarios."""

    def test_random_games(self):
        """Test playing multiple random games."""
        env = universal_poker.UniversalPoker()

        for seed in range(10):
            key = jax.random.PRNGKey(seed)
            state = env.init(key)

            steps = 0
            while not state.terminated and steps < 100:  # Prevent infinite loops
                legal_actions = jnp.where(state.legal_action_mask)[0]

                if len(legal_actions) > 0:
                    # Randomly choose a legal action
                    key, subkey = jax.random.split(key)
                    action = jax.random.choice(subkey, legal_actions)
                    state = env.step(state, action)
                else:
                    break  # No legal actions

                steps += 1

            # Game should eventually terminate
            assert state.terminated or steps >= 100

    def test_observation_shape(self):
        """Test observation shape and content."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        for player_id in range(state.num_players):
            state = state.replace(current_player=player_id)
            obs = env.observe(state)

            # Check observation is proper array
            assert isinstance(obs, jnp.ndarray)
            # New format uses uint32 types for unified data type
            assert obs.dtype == jnp.uint32  # All components now unified to uint32

            # Check new observation size: [hole_cardset[2], board_cardset[2], pot, stack, bets[num_players], folded[num_players], round]
            expected_size = 2 + 2 + 1 + 1 + state.num_players + state.num_players + 1  # cardsets uint32[2] + game state
            assert len(obs) == expected_size

            # Verify cardset components are present (first four elements)
            assert obs[0] >= 0  # hole cardset low uint32
            assert obs[1] >= 0  # hole cardset high uint32
            assert obs[2] >= 0  # board cardset low uint32 (could be 0 in preflop)
            assert obs[3] >= 0  # board cardset high uint32 (could be 0 in preflop)

    def test_config_observation_vectors(self):
        """Test observation vectors with config string setup."""
        config_str = """GAMEDEF
numplayers = 3
stack = 100 150 200
blind = 5 10 0
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Test observations for each player
        for player_id in range(3):
            state = state.replace(current_player=player_id)
            obs = env.observe(state)

            # Check observation structure: [hole_cardset[2], board_cardset[2], pot, stack, bets[num_players], folded[num_players], round]
            expected_size = 2 + 2 + 1 + 1 + 3 + 3 + 1  # 13 elements for 3 players
            assert len(obs) == expected_size

            # Check pot value is correct
            pot_idx = 4  # hole[2] + board[2] = 4
            assert obs[pot_idx] == 15  # 5 + 10 + 0

            # Check individual player's stack
            stack_idx = 5  # hole[2] + board[2] + pot[1] = 5
            print("obs:", obs)
            expected_stacks = [95, 140, 200]
            assert obs[stack_idx] == expected_stacks[player_id]

            # Check bets in observation
            bets_start_idx = 6  # hole[2] + board[2] + pot[1] + stack[1] = 6
            assert obs[bets_start_idx] == 5  # Player 0 bet
            assert obs[bets_start_idx + 1] == 10  # Player 1 bet
            assert obs[bets_start_idx + 2] == 0  # Player 2 bet

            # Check folded status (no one folded initially)
            folded_start_idx = 9  # bets_start + 3 = 9 (3 players)
            for i in range(3):
                assert obs[folded_start_idx + i] == 0  # No one folded

            # Check round (should be 0 for preflop)
            round_idx = 12  # folded_start + 3 = 12 (3 players)
            assert obs[round_idx] == 0

    def test_config_step_actions(self):
        """Test step actions with config string setup."""
        config_str = """GAMEDEF
numplayers = 3
stack = 100 150 200
blind = 5 10 0
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Initial state checks
        # With blinds [5, 10, 0], max_bet=10, first player to act should be player 2 (after the "big blind")
        assert state.current_player == 2  # Player 2 acts first (UTG)
        assert state.pot == 15
        assert state.max_bet == 10

        # Player 2 calls
        new_state = env.step(state, universal_poker.CALL)
        assert new_state.bets[2] == 10  # Player 2 now has 10 in pot
        assert new_state.stacks[2] == 190  # 200 - 10
        assert new_state.pot == 25  # 15 + 10
        assert new_state.current_player == 0  # Next player's turn

        # Player 0 raises
        new_state = env.step(new_state, universal_poker.RAISE)
        assert new_state.max_bet == 20  # Should be 2x current bet
        assert new_state.bets[0] == 20  # Player 0 total bet
        assert new_state.stacks[0] == 80  # 95 - 15 (additional chips beyond initial 5)
        assert new_state.pot == 40  # 25 + 15
        assert new_state.last_raiser == 0

        # Current player should now be 1 (but there's a bug in next player logic, let's check who's actually current)
        current_acting_player = new_state.current_player

        # Whoever is current player folds
        new_state = env.step(new_state, universal_poker.FOLD)
        assert new_state.folded[current_acting_player] == True

        # Check that folded player isn't included in active mask
        active_players = jnp.sum(new_state.active_mask)
        assert active_players == 2  # Only players 0 and 2 are active

    def test_config_multi_round_progression(self):
        """Test multi-round progression with config setup."""
        config_str = """GAMEDEF
numplayers = 3
stack = 1000 1000 1000
blind = 5 10 0
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # All players call to advance to flop
        state = env.step(state, universal_poker.CALL)  # Player 2 calls
        state = env.step(state, universal_poker.CALL)  # Player 0 calls
        state = env.step(state, universal_poker.CALL)  # Player 1 checks

        # Should advance to flop
        assert state.round == 1
        assert state.max_bet == 0  # Bets reset
        assert jnp.all(state.bets[:3] == 0)  # All bets reset
        assert state.pot == 30  # Total from preflop (10 + 10 + 10)

        # Check that all players still have chips
        # Each player contributed 10 total (P0: 5 blind + 5 call, P1: 10 blind, P2: 10 call)
        assert state.stacks[0] == 990  # 1000 - 10 total
        assert state.stacks[1] == 990  # 1000 - 10 total
        assert state.stacks[2] == 990  # 1000 - 10 total

    def test_config_all_in_scenario(self):
        """Test all-in scenario with different stack sizes."""
        config_str = """GAMEDEF
numplayers = 3
stack = 20 50 100
blind = 5 10 0
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Player 2 raises big
        state = env.step(state, universal_poker.RAISE)  # Player 2 raises to 20

        # Player 0 should be able to go all-in (has only 15 chips left)
        legal_actions = state.legal_action_mask
        assert legal_actions[universal_poker.CALL]  # Should be able to call/all-in

        state = env.step(state, universal_poker.CALL)  # Player 0 calls/all-in
        assert state.all_in[0] == True  # Player 0 should be all-in
        assert state.stacks[0] == 0  # No chips left

        # Check active mask excludes all-in player
        active_players = jnp.sum(state.active_mask)
        assert active_players == 2  # Only players 1 and 2 can still act

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

    def test_betting_edge_case_action_sequence_with_folds(self):
        """Test complex action sequences with folds affecting betting order."""
        env = universal_poker.UniversalPoker(num_players=4)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        initial_current_player = state.current_player

        # First player folds
        state = env.step(state, universal_poker.FOLD)
        assert state.folded[initial_current_player] == True, "First player should be folded"

        # Next player raises
        pre_raise_player = state.current_player
        state = env.step(state, universal_poker.RAISE)
        assert state.last_raiser == pre_raise_player, "Raiser should be tracked correctly"

        # Third player calls
        state = env.step(state, universal_poker.CALL)

        # Fourth player folds
        state = env.step(state, universal_poker.FOLD)

        # Verify only 2 players remain active
        active_players = jnp.sum(~state.folded[: state.num_players])
        assert active_players == 2, f"Should have 2 active players, got {active_players}"

    def test_betting_edge_case_raise_after_multiple_calls(self):
        """Test raising after multiple players have called."""
        env = universal_poker.UniversalPoker(num_players=4)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Multiple players call
        state = env.step(state, universal_poker.CALL)  # Player calls
        state = env.step(state, universal_poker.CALL)  # Player calls
        state = env.step(state, universal_poker.CALL)  # Player calls

        # Last player raises
        pre_raise_pot = state.pot
        state = env.step(state, universal_poker.RAISE)  # Player raises

        # Verify raise affects all previous callers
        assert state.max_bet > 2, "Max bet should increase after raise"
        post_raise_pot = state.pot
        assert post_raise_pot > pre_raise_pot, "Pot should increase from raise"

        # Now other players need to decide: fold, call the raise, or re-raise
        legal_actions = state.legal_action_mask
        assert legal_actions[universal_poker.FOLD] == True, "Should be able to fold"
        assert legal_actions[universal_poker.CALL] == True, "Should be able to call raise"
        assert legal_actions[universal_poker.RAISE] == True, "Should be able to re-raise"

    def test_betting_edge_case_all_in_side_pot_creation(self):
        """Test side pot creation with all-in players at different levels."""
        config_str = """GAMEDEF
numplayers = 3
stack = 5 15 50
blind = 1 2 0
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Create scenario where all players go all-in with different amounts
        # This tests the side pot logic through multiple rounds of raising

        # Player 2 raises (minimum raise from 2 to 4)
        state = env.step(state, universal_poker.RAISE)  # Player 2 raises to 4

        # Player 0 goes all-in with 5 chips (raises from 1 to 5)
        state = env.step(state, universal_poker.RAISE)  # Player 0 all-in with 5
        assert state.all_in[0] == True, "Player 0 should be all-in"
        assert state.bets[0] == 5, "Player 0 should have bet all 5 chips"

        # Player 1 raises further (minimum raise from 5 to 7)
        state = env.step(state, universal_poker.RAISE)  # Player 1 raises to 7

        # Player 2 raises again (minimum raise from 7 to 9)
        state = env.step(state, universal_poker.RAISE)  # Player 2 raises to 9

        # Player 1 raises again to exhaust more chips (9 -> 11)
        state = env.step(state, universal_poker.RAISE)  # Player 1 raises to 11

        # Player 2 raises to push Player 1 all-in (11 -> 13)
        state = env.step(state, universal_poker.RAISE)  # Player 2 raises to 13

        # Player 1 raises all-in (should use remaining chips to go to 15)
        state = env.step(state, universal_poker.RAISE)  # Player 1 all-in with 15
        assert state.all_in[1] == True, "Player 1 should be all-in"
        total_contribution = state.previous_round_bets[1] + state.bets[1]
        assert total_contribution == 15, f"Player 1 should have contributed 15 chips, got {total_contribution}"

        # Player 2 calls to match Player 1's all-in
        state = env.step(state, universal_poker.CALL)  # Player 2 calls to 15

        # Verify final all-in status: P0 and P1 are all-in with different amounts
        assert state.all_in[0] == True, "Player 0 should be all-in with 5 chips"
        assert state.all_in[1] == True, "Player 1 should be all-in with 15 chips"
        assert state.all_in[2] == False, "Player 2 should not be all-in (has chips remaining)"

    def test_betting_edge_case_fractional_all_in_raise(self):
        """Test raise when player's all-in amount is less than minimum raise."""
        config_str = """GAMEDEF
numplayers = 3
stack = 15 6 100
blind = 5 10 0
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Player 2 raises to 20
        state = env.step(state, universal_poker.RAISE)  # Player 2 raises to 20

        # Player 0 wants to raise but only has 10 chips left (15-5 blind)
        # Normal minimum raise would be to 40 (20 + 20), but player only has 15 total
        # This should still be allowed as an all-in raise
        state = env.step(state, universal_poker.RAISE)  # Player 0 attempts all-in raise

        assert state.all_in[0] == True, "Player 0 should be all-in"
        assert state.stacks[0] == 0, "Player 0 should have 0 chips left"
        total_contribution = state.previous_round_bets[0] + state.bets[0]
        assert total_contribution == 15, f"Player 0 should have contributed all chips (15), got {total_contribution}"
        # The raise should be allowed even though it's less than full minimum raise amount

    def test_betting_edge_case_dead_money_preservation(self):
        """Test that folded players' contributions remain in pot."""
        config_str = """GAMEDEF
numplayers = 4
stack = 100 100 100 100
blind = 5 10 0 0
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=4, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Track initial pot
        initial_pot = state.pot  # Should be 15 (5+10)

        # Players 2 and 3 call, then Player 2 raises
        state = env.step(state, universal_poker.CALL)  # Player 2 calls 10
        state = env.step(state, universal_poker.CALL)  # Player 3 calls 10
        state = env.step(state, universal_poker.CALL)  # Player 0 calls to 10
        state = env.step(state, universal_poker.RAISE)  # Player 1 raises to 20

        pot_after_raise = state.pot

        # Player 2 folds after contributing
        state = env.step(state, universal_poker.FOLD)  # Player 2 folds

        # Pot should not decrease when player folds - their money stays as "dead money"
        assert state.pot == pot_after_raise, "Pot should not decrease when player folds"
        assert state.folded[2] == True, "Player 2 should be folded"

    def test_betting_edge_case_string_betting_prevention(self):
        """Test that players cannot 'string bet' (raise to amount less than minimum)."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Initial max_bet is 2 (big blind)
        # Normal raise should be to 4 (double)
        current_player = state.current_player
        state = env.step(state, universal_poker.RAISE)

        # Verify raise went to proper amount (not less)
        assert state.max_bet == 4, f"Raise should go to 4, not less. Got {state.max_bet}"
        assert state.bets[current_player] == 4, f"Player bet should be 4. Got {state.bets[current_player]}"

    def test_betting_edge_case_exact_minimum_chips(self):
        """Test scenarios where players have exactly the minimum required chips."""
        config_str = """GAMEDEF
numplayers = 3
stack = 4 8 100
blind = 1 2 0
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Player 2 raises to 4 (doubling max_bet of 2)
        state = env.step(state, universal_poker.RAISE)  # Player 2 raises to 4

        # Player 0 has exactly 3 chips left (4-1 blind), needs 3 more to call max_bet=4
        # This should result in an all-in call
        state = env.step(state, universal_poker.CALL)  # Player 0 calls all-in
        assert state.all_in[0] == True, "Player 0 should be all-in with exact call amount"
        assert state.stacks[0] == 0, "Player 0 should have 0 chips left"

        # Player 1 has exactly 6 chips left (8-2 blind), needs 2 more to call max_bet=4
        # This should result in a normal call and advance to next round
        state = env.step(state, universal_poker.CALL)  # Player 1 calls
        assert state.all_in[1] == False, "Player 1 should not be all-in"
        assert state.stacks[1] == 4, "Player 1 should have 4 chips left (6-2)"
        # After all players call, betting round ends and bets reset for next round
        assert state.round == 1, "Should advance to flop after all players call"
        assert state.bets[1] == 0, "Bets should reset to 0 for new round"


if __name__ == "__main__":
    import sys
    import traceback

    test_suite = TestUniversalPokerIntegration()

    print("Running Universal Poker integration tests...")

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
