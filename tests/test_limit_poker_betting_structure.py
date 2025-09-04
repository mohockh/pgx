import jax
import jax.numpy as jnp

from pgx import universal_poker


class TestLimitPokerBettingStructure:
    """Test suite for LimitPokerBettingStructure class - testing limit poker specific logic."""

    def test_apply_action_fold(self):
        """Test apply_action with fold action (0)."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Get the betting structure and call apply_action directly
        betting_structure = env.betting_structure
        new_state = betting_structure.apply_action(env, state, universal_poker.FOLD)

        # Verify fold action was applied
        current_player = state.current_player
        assert new_state.folded[current_player] == True, "Player should be folded after fold action"
        assert isinstance(new_state, universal_poker.State), "Should return State object"

    def test_apply_action_call(self):
        """Test apply_action with call action (1)."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        original_current_player = state.current_player
        original_bet = state.bets[original_current_player]
        original_stack = state.stacks[original_current_player]
        call_amount = state.max_bet - original_bet

        # Get the betting structure and call apply_action directly
        betting_structure = env.betting_structure
        new_state = betting_structure.apply_action(env, state, universal_poker.CALL)

        # Verify call action was applied
        expected_new_bet = original_bet + call_amount
        expected_new_stack = original_stack - call_amount

        assert new_state.bets[original_current_player] == expected_new_bet, f"Bet should be {expected_new_bet}"
        assert new_state.stacks[original_current_player] == expected_new_stack, f"Stack should be {expected_new_stack}"
        assert not new_state.folded[original_current_player], "Player should not be folded after call"

    def test_apply_action_raise(self):
        """Test apply_action with raise action (2)."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        original_current_player = state.current_player
        original_max_bet = state.max_bet

        # Get the betting structure and call apply_action directly
        betting_structure = env.betting_structure
        new_state = betting_structure.apply_action(env, state, universal_poker.RAISE)

        # Verify raise action was applied
        assert new_state.max_bet > original_max_bet, "Max bet should increase after raise"
        assert new_state.last_raiser == original_current_player, "Last raiser should be updated"
        assert not new_state.folded[original_current_player], "Player should not be folded after raise"

    def test_apply_action_delegates_to_basic_action(self):
        """Test that apply_action always delegates to env._apply_basic_action."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        betting_structure = env.betting_structure

        # Test all three actions
        for action in [universal_poker.FOLD, universal_poker.CALL, universal_poker.RAISE]:
            # Apply action through betting structure
            state_via_betting_structure = betting_structure.apply_action(env, state, action)

            # Apply action directly through env (should be equivalent)
            state_via_env = env._apply_basic_action(state, action)

            # Results should be identical since LimitPokerBettingStructure delegates to _apply_basic_action
            assert jnp.array_equal(
                state_via_betting_structure.bets, state_via_env.bets
            ), f"Bets should match for action {action}"
            assert jnp.array_equal(
                state_via_betting_structure.stacks, state_via_env.stacks
            ), f"Stacks should match for action {action}"
            assert jnp.array_equal(
                state_via_betting_structure.folded, state_via_env.folded
            ), f"Folded should match for action {action}"

    def test_get_legal_actions_shape_and_type(self):
        """Test get_legal_actions returns correct shape and type."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)

        # Verify shape and type
        assert legal_actions.shape == (3,), f"Expected shape (3,) for limit games, got {legal_actions.shape}"
        assert legal_actions.dtype == jnp.bool_, f"Expected bool dtype, got {legal_actions.dtype}"

    def test_get_legal_actions_size_three_for_limit(self):
        """Test that legal actions array has exactly 3 elements for limit poker."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)

        # Legal actions should have exactly 3 elements for limit poker
        assert legal_actions.shape == (3,), f"Expected 3 actions for limit poker, got {legal_actions.shape}"

        # At least one action should be available in a normal game
        assert jnp.any(legal_actions), "At least one action should be available"

    def test_get_legal_actions_normal_state(self):
        """Test legal actions in normal game state."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)
        current_player = state.current_player

        # In initial state, player should typically be able to fold and call
        # Fold should be available (not all-in, not folded)
        expected_can_fold = ~(state.all_in[current_player] | state.folded[current_player])
        assert legal_actions[universal_poker.FOLD] == expected_can_fold, "Fold availability should match expected"

        # Call should be available if bet <= max_bet and has chips
        expected_can_call = (
            (state.bets[current_player] <= state.max_bet)
            and (state.stacks[current_player] > 0)
            and ~state.folded[current_player]
        )
        assert legal_actions[universal_poker.CALL] == expected_can_call, "Call availability should match expected"

        # Raise should be available if total chips > max_bet
        total_chips = state.stacks[current_player] + state.bets[current_player]
        expected_can_raise = (total_chips > state.max_bet) and ~state.folded[current_player]
        assert legal_actions[universal_poker.RAISE] == expected_can_raise, "Raise availability should match expected"

    def test_get_legal_actions_folded_player(self):
        """Test legal actions for folded player."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Force current player to be folded
        current_player = state.current_player
        state = state.replace(folded=state.folded.at[current_player].set(True))

        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)

        # Folded player should not be able to take any actions
        assert legal_actions[universal_poker.FOLD] == False, "Folded player should not be able to fold again"
        assert legal_actions[universal_poker.CALL] == False, "Folded player should not be able to call"
        assert legal_actions[universal_poker.RAISE] == False, "Folded player should not be able to raise"

    def test_get_legal_actions_all_in_player(self):
        """Test legal actions for all-in player."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Force current player to be all-in
        current_player = state.current_player
        state = state.replace(
            stacks=state.stacks.at[current_player].set(0), all_in=state.all_in.at[current_player].set(True)
        )

        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)

        # All-in player should not be able to fold (committed)
        assert legal_actions[universal_poker.FOLD] == False, "All-in player should not be able to fold"
        # Call and raise depend on other conditions, but generally shouldn't be available
        assert legal_actions[universal_poker.CALL] == False, "All-in player with 0 stack should not be able to call"
        assert legal_actions[universal_poker.RAISE] == False, "All-in player should not be able to raise"

    def test_get_legal_actions_insufficient_chips(self):
        """Test legal actions when player has insufficient chips."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 1 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Find player with very small stack
        small_stack_player = 0 if state.stacks[0] < state.stacks[1] else 1
        if state.current_player != small_stack_player:
            state = state.replace(current_player=small_stack_player)

        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)

        current_player = state.current_player

        # Should be able to fold if not all-in and not folded
        if not (state.all_in[current_player] or state.folded[current_player]):
            assert legal_actions[universal_poker.FOLD] == True, "Should be able to fold with small stack"

        # Call ability depends on having any chips
        if state.stacks[current_player] == 0:
            assert legal_actions[universal_poker.CALL] == False, "Cannot call with 0 stack"

    def test_get_legal_actions_matches_original_logic(self):
        """Test that get_legal_actions matches the original limit logic."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        betting_structure = env.betting_structure
        legal_actions_from_structure = betting_structure.get_legal_actions(env, state)

        # Get legal actions from the main env (which should delegate to betting structure)
        legal_actions_from_env = env._get_legal_actions(state)

        # They should be identical
        assert jnp.array_equal(
            legal_actions_from_structure, legal_actions_from_env
        ), "Legal actions from betting structure should match those from env"

    def test_initialize_game_params_no_op(self):
        """Test that initialize_game_params is a no-op for limit games."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        # Get initial state of raise_multipliers
        original_multipliers = env.raise_multipliers.copy()

        # Call initialize_game_params
        betting_structure = env.betting_structure
        betting_structure.initialize_game_params(env)

        # Multipliers should be unchanged (no-op for limit games)
        assert jnp.array_equal(
            env.raise_multipliers, original_multipliers
        ), "initialize_game_params should not modify raise_multipliers for limit games"

    def test_initialize_game_params_does_not_break_functionality(self):
        """Test that initialize_game_params doesn't break game functionality."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        # Initialize game and verify it works
        key = jax.random.PRNGKey(42)
        state_before = env.init(key)

        # Call initialize_game_params again
        betting_structure = env.betting_structure
        betting_structure.initialize_game_params(env)

        # Initialize again and verify identical results
        state_after = env.init(key)

        # States should be identical
        assert jnp.array_equal(state_before.stacks, state_after.stacks), "Stacks should be identical"
        assert jnp.array_equal(state_before.bets, state_after.bets), "Bets should be identical"
        assert jnp.array_equal(state_before.folded, state_after.folded), "Folded should be identical"
        assert state_before.pot == state_after.pot, "Pot should be identical"

    def test_betting_structure_type_verification(self):
        """Test that limit games use LimitPokerBettingStructure."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
END GAMEDEF""",
        )

        # Should have LimitPokerBettingStructure
        assert isinstance(
            env.betting_structure, universal_poker.LimitPokerBettingStructure
        ), "Limit games should use LimitPokerBettingStructure"

        # Should have limit game type
        assert env.game_type == "limit", "Game type should be limit"

    def test_edge_case_exact_call_amount(self):
        """Test legal actions when player bet exactly equals max bet."""
        env = universal_poker.UniversalPoker(
            num_players=2,
            config_str="""GAMEDEF
limit
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF""",
        )

        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Force current player's bet to equal max_bet
        current_player = state.current_player
        state = state.replace(bets=state.bets.at[current_player].set(state.max_bet))

        betting_structure = env.betting_structure
        legal_actions = betting_structure.get_legal_actions(env, state)

        # Should be able to fold if not all-in and not folded
        if not (state.all_in[current_player] or state.folded[current_player]):
            assert legal_actions[universal_poker.FOLD] == True, "Should be able to fold when bet equals max bet"

        # Call should be available (bet <= max_bet condition)
        if state.stacks[current_player] > 0 and not state.folded[current_player]:
            assert legal_actions[universal_poker.CALL] == True, "Should be able to call when bet equals max bet"

    def test_multiple_game_states(self):
        """Test legal actions across multiple game states and configurations."""
        configs = [
            """GAMEDEF
limit
numplayers = 2
stack = 50 200
blind = 1 2
END GAMEDEF""",
            """GAMEDEF
limit
numplayers = 3
stack = 100 100 100
blind = 1 2 0
END GAMEDEF""",
            """GAMEDEF
limit
numplayers = 4
stack = 200 200 200 200
blind = 1 2 0 0
END GAMEDEF""",
        ]

        for i, config in enumerate(configs):
            num_players = i + 2
            env = universal_poker.UniversalPoker(num_players=num_players, config_str=config)
            key = jax.random.PRNGKey(42 + i)
            state = env.init(key)

            betting_structure = env.betting_structure
            legal_actions = betting_structure.get_legal_actions(env, state)

            # Basic validations for all configurations
            assert legal_actions.shape == (3,), f"Config {i}: Wrong shape for limit games"
            assert legal_actions.dtype == jnp.bool_, f"Config {i}: Wrong dtype"
            assert isinstance(
                env.betting_structure, universal_poker.LimitPokerBettingStructure
            ), f"Config {i}: Should use LimitPokerBettingStructure"
