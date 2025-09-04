import jax
import jax.numpy as jnp

from pgx import universal_poker


class TestNoLimitBettingStructure:
    """Test suite for NoLimitBettingStructure class - testing no-limit poker specific logic."""

    def test_apply_action_basic_actions_fold_call_raise(self):
        """Test apply_action with basic actions 0-2 (fold, call, min_raise)."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        betting_structure = env.betting_structure
        
        # Test fold (action 0)
        fold_state = betting_structure.apply_action(env, state, universal_poker.FOLD)
        current_player = state.current_player
        assert fold_state.folded[current_player] == True, "Player should be folded after fold action"
        
        # Test call (action 1) 
        call_state = betting_structure.apply_action(env, state, universal_poker.CALL)
        original_bet = state.bets[current_player]
        call_amount = state.max_bet - original_bet
        expected_new_bet = original_bet + call_amount
        assert call_state.bets[current_player] == expected_new_bet, "Bet should increase by call amount"
        
        # Test raise (action 2)
        raise_state = betting_structure.apply_action(env, state, universal_poker.RAISE)
        assert raise_state.max_bet > state.max_bet, "Max bet should increase after raise"
        assert raise_state.last_raiser == current_player, "Last raiser should be updated"

    def test_apply_action_nolimit_actions_3_to_12(self):
        """Test apply_action with no-limit specific actions 3-12."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 1000 1000
blind = 10 20
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        betting_structure = env.betting_structure
        current_player = state.current_player
        original_max_bet = state.max_bet
        
        # Test a few no-limit actions (3-12)
        for action in [3, 5, 8, 12]:  # Sample of no-limit actions
            # Calculate expected bet amount
            expected_bet_amount = env._calculate_nolimit_bet_amount(state, action)
            if expected_bet_amount > 0 and expected_bet_amount <= state.stacks[current_player]:
                new_state = betting_structure.apply_action(env, state, action)
                
                # Verify it was processed as a no-limit action
                assert new_state.max_bet >= original_max_bet, f"Max bet should increase or stay same for action {action}"
                assert new_state.last_raiser == current_player, f"Last raiser should be updated for action {action}"
                
                # Verify bet was applied
                expected_new_bet = state.bets[current_player] + expected_bet_amount
                assert new_state.bets[current_player] == expected_new_bet, \
                    f"Bet should be updated correctly for action {action}"

    def test_apply_action_conditional_branching(self):
        """Test the conditional logic between basic and no-limit actions."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 500 500
blind = 5 10
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        betting_structure = env.betting_structure
        
        # Actions 0-2 should use basic action path
        for action in [0, 1, 2]:
            state_from_betting_structure = betting_structure.apply_action(env, state, action)
            state_from_basic_action = env._apply_basic_action(state, action)
            
            # Results should be identical since actions 0-2 delegate to _apply_basic_action
            assert jnp.array_equal(state_from_betting_structure.bets, state_from_basic_action.bets), \
                f"Basic action {action} should produce identical bets"
            assert jnp.array_equal(state_from_betting_structure.stacks, state_from_basic_action.stacks), \
                f"Basic action {action} should produce identical stacks"
            assert jnp.array_equal(state_from_betting_structure.folded, state_from_basic_action.folded), \
                f"Basic action {action} should produce identical folded status"
        
        # Actions 3+ should use no-limit action path (if affordable)
        current_player = state.current_player
        for action in [3, 6, 9]:
            bet_amount = env._calculate_nolimit_bet_amount(state, action)
            if bet_amount > 0 and bet_amount <= state.stacks[current_player]:
                state_from_betting_structure = betting_structure.apply_action(env, state, action)
                state_from_nolimit_action = env._apply_nolimit_action(state, action)
                
                # Results should be identical since actions 3+ delegate to _apply_nolimit_action
                assert jnp.array_equal(state_from_betting_structure.bets, state_from_nolimit_action.bets), \
                    f"No-limit action {action} should produce identical bets"
                assert jnp.array_equal(state_from_betting_structure.stacks, state_from_nolimit_action.stacks), \
                    f"No-limit action {action} should produce identical stacks"

    def test_get_legal_actions_shape_and_type(self):
        """Test get_legal_actions returns correct shape and type."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        
        # Verify shape and type
        assert legal_actions.shape == (13,), f"Expected shape (13,), got {legal_actions.shape}"
        assert legal_actions.dtype == jnp.bool_, f"Expected bool dtype, got {legal_actions.dtype}"

    def test_get_legal_actions_all_13_actions_considered(self):
        """Test that all 13 actions are properly evaluated."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 1000 1000
blind = 10 20
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        
        # In a normal no-limit game with large stacks, most actions should be available
        current_player = state.current_player
        
        # Actions 0-2 should typically be available
        if not (state.all_in[current_player] or state.folded[current_player]):
            assert legal_actions[0] == True, "Fold should be available in normal state"
        
        if state.stacks[current_player] > 0 and not state.folded[current_player]:
            assert legal_actions[1] == True, "Call should be available with sufficient chips"
        
        # At least some no-limit actions (3+) should be available with large stacks
        nolimit_actions_available = jnp.any(legal_actions[3:])
        assert nolimit_actions_available, "At least some no-limit actions should be available with large stacks"

    def test_get_legal_actions_basic_actions_logic(self):
        """Test the logic for basic actions 0-2."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        current_player = state.current_player
        
        # Test fold logic: can fold if not all-in and not folded
        expected_can_fold = ~(state.all_in[current_player] | state.folded[current_player])
        assert legal_actions[0] == expected_can_fold, "Fold availability should match expected"
        
        # Test call logic: can call if bet <= max_bet and has chips and not folded
        expected_can_call = (
            (state.bets[current_player] <= state.max_bet) &
            (state.stacks[current_player] > 0) &
            ~state.folded[current_player]
        )
        assert legal_actions[1] == expected_can_call, "Call availability should match expected"
        
        # Test raise logic: can raise if total chips > max_bet and not folded
        total_chips = state.stacks[current_player] + state.bets[current_player]
        expected_can_raise = (total_chips > state.max_bet) & ~state.folded[current_player]
        assert legal_actions[2] == expected_can_raise, "Raise availability should match expected"

    def test_get_legal_actions_affordability_checks(self):
        """Test affordability checks for actions 3-12."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 50 1000
blind = 1 2
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Find player with smaller stack
        small_stack_player = 0 if state.stacks[0] < state.stacks[1] else 1
        state = state.replace(current_player=small_stack_player)
        
        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        current_player = state.current_player
        
        # Check that affordability is correctly calculated for actions 3-12
        for action in range(3, 13):
            bet_amount = env._calculate_nolimit_bet_amount(state, action)
            expected_affordable = (
                (bet_amount > 0) & 
                (bet_amount <= state.stacks[current_player]) & 
                ~state.folded[current_player]
            )
            assert legal_actions[action] == expected_affordable, \
                f"Action {action} affordability should match calculation: expected {expected_affordable}, got {legal_actions[action]}, bet_amount={bet_amount}, stack={state.stacks[current_player]}"

    def test_get_legal_actions_vectorized_affordability_check(self):
        """Test the vectorized affordability check using jax.vmap."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 200 200
blind = 5 10
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        betting_structure = env.betting_structure
        current_player = state.current_player
        
        # Define the affordability check function (same as in the implementation)
        def check_affordability(action_idx):
            bet_amount = env._calculate_nolimit_bet_amount(state, action_idx)
            return (bet_amount > 0) & (bet_amount <= state.stacks[current_player]) & ~state.folded[current_player]
        
        # Test that vectorized version matches individual checks
        actions_to_check = jnp.arange(3, 13)
        vectorized_results = jax.vmap(check_affordability)(actions_to_check)
        
        # Get actual legal actions from betting structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        
        # Compare vectorized results with legal_actions[3:13]
        assert jnp.array_equal(vectorized_results, legal_actions[3:13]), \
            "Vectorized affordability check should match legal_actions[3:13]"

    def test_get_legal_actions_folded_player(self):
        """Test legal actions for folded player."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Force current player to be folded
        current_player = state.current_player
        state = state.replace(folded=state.folded.at[current_player].set(True))
        
        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        
        # Folded player should not be able to take any actions
        for i in range(13):
            assert legal_actions[i] == False, f"Folded player should not be able to take action {i}"

    def test_get_legal_actions_all_in_player(self):
        """Test legal actions for all-in player."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Force current player to be all-in
        current_player = state.current_player
        state = state.replace(
            stacks=state.stacks.at[current_player].set(0),
            all_in=state.all_in.at[current_player].set(True)
        )
        
        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        
        # All-in player should not be able to take most actions
        assert legal_actions[0] == False, "All-in player should not be able to fold"
        assert legal_actions[1] == False, "All-in player with 0 stack should not be able to call"
        assert legal_actions[2] == False, "All-in player should not be able to raise"
        
        # No-limit actions should also be unavailable (0 stack)
        for i in range(3, 13):
            assert legal_actions[i] == False, f"All-in player should not be able to take action {i}"

    def test_get_legal_actions_matches_env_delegation(self):
        """Test that get_legal_actions matches what env._get_legal_actions returns."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 300 300
blind = 5 10
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        betting_structure = env.betting_structure
        legal_actions_from_structure = betting_structure.get_legal_actions(env, state)
        
        # Get legal actions from the main env (which should delegate to betting structure)
        legal_actions_from_env = env._get_legal_actions(state)
        
        # They should be identical
        assert jnp.array_equal(legal_actions_from_structure, legal_actions_from_env), \
            "Legal actions from betting structure should match those from env"

    def test_initialize_game_params_sets_multipliers(self):
        """Test that initialize_game_params sets raise_multipliers correctly."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""")
        
        # Clear multipliers first
        env.raise_multipliers = jnp.zeros((4, 10), dtype=jnp.float32)
        
        # Call initialize_game_params
        betting_structure = env.betting_structure
        betting_structure.initialize_game_params(env)
        
        # Verify multipliers are set
        assert env.raise_multipliers.shape == (4, 10), "Multipliers should have shape (4, 10)"
        assert env.raise_multipliers.dtype == jnp.float32, "Multipliers should be float32"
        
        # Verify not all zeros anymore
        assert not jnp.all(env.raise_multipliers == 0), "Multipliers should not be all zeros after initialization"

    def test_initialize_game_params_preflop_defaults(self):
        """Test preflop (round 0) multiplier defaults."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
END GAMEDEF""")
        
        # Expected preflop defaults
        expected_preflop = jnp.array([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, -1.0], dtype=jnp.float32)
        
        # Verify preflop multipliers (round 0)
        assert jnp.array_equal(env.raise_multipliers[0], expected_preflop), \
            f"Preflop multipliers should match expected: expected {expected_preflop}, got {env.raise_multipliers[0]}"

    def test_initialize_game_params_postflop_defaults(self):
        """Test postflop (rounds 1-3) multiplier defaults."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
END GAMEDEF""")
        
        # Expected postflop defaults  
        expected_postflop = jnp.array([0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, -1.0], dtype=jnp.float32)
        
        # Verify postflop multipliers (rounds 1-3)
        for round_idx in [1, 2, 3]:
            assert jnp.array_equal(env.raise_multipliers[round_idx], expected_postflop), \
                f"Round {round_idx} multipliers should match expected: expected {expected_postflop}, got {env.raise_multipliers[round_idx]}"

    def test_initialize_game_params_array_structure(self):
        """Test the 4x10 array structure of raise_multipliers."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
END GAMEDEF""")
        
        # Verify structure
        assert env.raise_multipliers.shape == (4, 10), "Should have 4 rounds, 10 actions each"
        assert env.raise_multipliers.dtype == jnp.float32, "Should be float32 dtype"
        
        # Verify all rounds have data
        for round_idx in range(4):
            assert not jnp.all(env.raise_multipliers[round_idx] == 0), \
                f"Round {round_idx} should have non-zero multipliers"
        
        # Verify all rounds have the all-in marker (-1.0) in the last position
        for round_idx in range(4):
            assert env.raise_multipliers[round_idx, 9] == -1.0, \
                f"Round {round_idx} should have all-in marker (-1.0) at position 9"

    def test_initialize_game_params_env_assignment(self):
        """Test that initialize_game_params properly assigns to env.raise_multipliers."""
        # Create env but don't let constructor initialize multipliers
        env = universal_poker.UniversalPoker.__new__(universal_poker.UniversalPoker)
        env._num_players = 2
        env.game_type = "nolimit"
        env.raise_multipliers = jnp.zeros((4, 10), dtype=jnp.float32)
        
        # Create betting structure and initialize
        betting_structure = universal_poker.NoLimitBettingStructure()
        betting_structure.initialize_game_params(env)
        
        # Verify env.raise_multipliers was updated
        assert not jnp.all(env.raise_multipliers == 0), "env.raise_multipliers should be updated"
        assert env.raise_multipliers[0, 0] == 2.0, "First preflop multiplier should be 2.0"
        assert env.raise_multipliers[1, 0] == 0.25, "First postflop multiplier should be 0.25"

    def test_betting_structure_type_verification(self):
        """Test that no-limit games use NoLimitBettingStructure."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
END GAMEDEF""")
        
        # Should have NoLimitBettingStructure
        assert isinstance(env.betting_structure, universal_poker.NoLimitBettingStructure), \
            "No-limit games should use NoLimitBettingStructure"
        
        # Should have nolimit game type
        assert env.game_type == "nolimit", "Game type should be nolimit"

    def test_large_stack_scenarios(self):
        """Test legal actions with very large stacks."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 10000 10000
blind = 50 100
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        
        # With large stacks, most actions should be available
        available_count = jnp.sum(legal_actions)
        assert available_count >= 10, f"With large stacks, should have many available actions, got {available_count}"
        
        # All basic actions should be available
        assert legal_actions[0] == True, "Fold should be available"
        assert legal_actions[1] == True, "Call should be available" 
        assert legal_actions[2] == True, "Raise should be available"
        
        # Most no-limit actions should be available
        nolimit_available = jnp.sum(legal_actions[3:])
        assert nolimit_available >= 5, f"Should have several no-limit actions available, got {nolimit_available}"

    def test_small_stack_scenarios(self):
        """Test legal actions with very small stacks."""
        env = universal_poker.UniversalPoker(num_players=2, config_str="""GAMEDEF
nolimit
numplayers = 2
stack = 5 100
blind = 2 5
END GAMEDEF""")
        
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Make sure we're testing the small stack player
        small_stack_player = 0 if state.stacks[0] < state.stacks[1] else 1
        state = state.replace(current_player=small_stack_player)
        
        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        
        # With small stack, fewer actions should be available
        available_count = jnp.sum(legal_actions)
        
        # Should have some basic actions available if not all-in/folded
        current_player = state.current_player
        if not (state.all_in[current_player] or state.folded[current_player]):
            assert legal_actions[0] == True, "Fold should be available for small stack"
        
        # Most no-limit actions might not be affordable
        nolimit_available = jnp.sum(legal_actions[3:])
        # Note: Some might still be available depending on the multipliers and pot size

    def test_multiple_rounds_and_configurations(self):
        """Test betting structure across multiple game configurations."""
        configs = [
            ("""GAMEDEF
nolimit
numplayers = 2
stack = 100 200
blind = 1 2
END GAMEDEF""", 2),
            ("""GAMEDEF
nolimit
numplayers = 3
stack = 150 150 150
blind = 2 5 0
END GAMEDEF""", 3),
            ("""GAMEDEF
nolimit
numplayers = 4
stack = 500 500 500 500
blind = 10 20 0 0
END GAMEDEF""", 4)
        ]
        
        for i, (config, num_players) in enumerate(configs):
            env = universal_poker.UniversalPoker(num_players=num_players, config_str=config)
            key = jax.random.PRNGKey(42 + i)
            state = env.init(key)
            
            betting_structure = env.betting_structure
            legal_actions = betting_structure.get_legal_actions(env, state)
            
            # Basic validations for all configurations
            assert legal_actions.shape == (13,), f"Config {i}: Wrong shape"
            assert legal_actions.dtype == jnp.bool_, f"Config {i}: Wrong dtype"
            assert isinstance(env.betting_structure, universal_poker.NoLimitBettingStructure), \
                f"Config {i}: Should use NoLimitBettingStructure"
            
            # Should have proper raise multipliers initialized
            assert env.raise_multipliers.shape == (4, 10), f"Config {i}: Wrong multipliers shape"
            assert not jnp.all(env.raise_multipliers == 0), f"Config {i}: Multipliers should not be all zeros"

    def test_raise_multipliers_configuration_compatibility(self):
        """Test that betting structure works with custom raise multipliers from config."""
        config_str = """GAMEDEF
nolimit
raise_multipliers 0 3.0 4.0 5.0 all_in
raise_multipliers 1 0.5 1.0 1.5 all_in
numplayers = 2
stack = 200 200
blind = 5 10
END GAMEDEF"""
        
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Verify custom multipliers were set
        assert env.raise_multipliers[0, 0] == 3.0, "Custom preflop multiplier should be set"
        assert env.raise_multipliers[0, 1] == 4.0, "Custom preflop multiplier should be set"
        assert env.raise_multipliers[1, 0] == 0.5, "Custom postflop multiplier should be set"
        
        # Verify betting structure still works
        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        
        assert legal_actions.shape == (13,), "Should still return 13 actions"
        assert jnp.any(legal_actions), "Should have some legal actions available"