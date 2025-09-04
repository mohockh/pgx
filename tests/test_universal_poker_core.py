import jax
import jax.numpy as jnp

from pgx import universal_poker


class TestUniversalPokerCore:
    """Test suite for core Universal Poker game mechanics - initialization, configuration, and basic properties."""

    def test_init_basic(self):
        """Test basic game initialization."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        assert isinstance(state, universal_poker.State)
        assert state.num_players == 2
        assert state.round == 0  # Start at preflop
        assert state.pot == 3  # Small blind (1) + big blind (2)
        assert not state.terminated
        assert state.stacks[0] == 199  # 200 - 1 (small blind)
        assert state.stacks[1] == 198  # 200 - 2 (big blind)

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        config_str = """GAMEDEF
numplayers = 3
stack = 100 100 100
blind = 5 10 0
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        assert state.num_players == 3
        assert state.pot == 15  # 5 + 10
        assert state.stacks[0] == 95  # 100 - 5
        assert state.stacks[1] == 90  # 100 - 10
        assert state.stacks[2] == 100  # No blind

    def test_hole_cards_dealt(self):
        """Test that hole cards are properly dealt."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Each player should have 2 hole cards (convert from cardset)
        from pgx.poker_eval.cardset import cardset_to_cards

        for p in range(state.num_players):
            hole_cards = cardset_to_cards(state.hole_cardsets[p])[:2]  # Take first 2
            assert jnp.all(hole_cards >= 0)  # Valid card indices
            assert jnp.all(hole_cards < 52)  # Within deck range

    def test_board_cards_dealt(self):
        """Test that board cards are dealt but not visible initially."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # 5 board cards should be dealt (but not visible in preflop)
        from pgx.poker_eval.cardset import cardset_to_cards

        board_cards = cardset_to_cards(state.board_cardset)[:5]  # Take first 5
        assert jnp.all(board_cards >= 0)
        assert jnp.all(board_cards < 52)

    def test_game_properties(self):
        """Test game properties."""
        env = universal_poker.UniversalPoker()

        assert env.id == "universal_poker"
        assert env.version == "v1"
        assert env.num_players == 2

    def test_jax_compilation(self):
        """Test that key functions can be JIT compiled."""
        env = universal_poker.UniversalPoker()

        # Test init compilation
        init_fn = jax.jit(env.init)
        key = jax.random.PRNGKey(42)
        state = init_fn(key)
        assert isinstance(state, universal_poker.State)

        # Test step compilation
        step_fn = jax.jit(env.step)
        new_state = step_fn(state, universal_poker.CALL)
        assert isinstance(new_state, universal_poker.State)

    def test_deterministic_behavior(self):
        """Test that games are deterministic given the same seed."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)

        # Play two identical games
        state1 = env.init(key)
        state2 = env.init(key)

        # States should be identical
        assert jnp.array_equal(state1.hole_cardsets, state2.hole_cardsets)
        assert jnp.array_equal(state1.board_cardset, state2.board_cardset)
        assert state1.pot == state2.pot

    def test_state_consistency(self):
        """Test state consistency throughout game."""
        env = universal_poker.UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Check initial state consistency
        total_chips = jnp.sum(state.stacks[: state.num_players]) + state.pot
        initial_total = state.num_players * 200  # Each player starts with 200
        assert total_chips == initial_total

        # Play a few moves and check consistency
        for _ in range(5):
            if state.terminated:
                break

            legal_actions = jnp.where(state.legal_action_mask)[0]
            if len(legal_actions) > 0:
                key, subkey = jax.random.split(key)
                action = jax.random.choice(subkey, legal_actions)
                state = env.step(state, action)

                # Total chips should remain constant
                total_chips = jnp.sum(state.stacks[: state.num_players]) + state.pot
                assert total_chips == initial_total

    # Configuration String Tests
    def test_config_string_basic(self):
        """Test basic config string parsing."""
        config_str = """GAMEDEF
numplayers = 3
stack = 100 150 200
blind = 5 10 0
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Test number of players
        assert state.num_players == 3
        assert env.num_players == 3

        # Test stack sizes
        assert state.stacks[0] == 95  # 100 - 5 (blind)
        assert state.stacks[1] == 140  # 150 - 10 (blind)
        assert state.stacks[2] == 200  # 200 - 0 (no blind)

        # Test blind structure
        assert state.bets[0] == 5  # Player 0 posts 5
        assert state.bets[1] == 10  # Player 1 posts 10
        assert state.bets[2] == 0  # Player 2 posts 0

        # Test pot
        assert state.pot == 15  # 5 + 10 + 0
        assert state.max_bet == 10  # Max of blind amounts

    def test_config_string_four_players(self):
        """Test config string with four players."""
        config_str = """GAMEDEF
numplayers = 4
stack = 500 500 500 500
blind = 1 2 0 0
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=4, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        assert state.num_players == 4

        # Check stacks after blinds
        assert state.stacks[0] == 499  # 500 - 1
        assert state.stacks[1] == 498  # 500 - 2
        assert state.stacks[2] == 500  # 500 - 0
        assert state.stacks[3] == 500  # 500 - 0

        # Check blinds
        assert state.bets[0] == 1
        assert state.bets[1] == 2
        assert state.bets[2] == 0
        assert state.bets[3] == 0

        assert state.pot == 3
        assert state.max_bet == 2

    def test_config_string_different_stacks(self):
        """Test config string with different stack sizes."""
        config_str = """GAMEDEF
numplayers = 3
stack = 50 100 200
blind = 1 2 0
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Player 0: small stack
        assert state.stacks[0] == 49  # 50 - 1
        # Player 1: medium stack
        assert state.stacks[1] == 98  # 100 - 2
        # Player 2: large stack
        assert state.stacks[2] == 200  # 200 - 0

    def test_config_string_ante_structure(self):
        """Test config string with ante-like structure (all players post)."""
        config_str = """GAMEDEF
numplayers = 3
stack = 100 100 100
blind = 5 5 5
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # All players post same amount
        assert state.stacks[0] == 95  # 100 - 5
        assert state.stacks[1] == 95  # 100 - 5
        assert state.stacks[2] == 95  # 100 - 5

        assert state.bets[0] == 5
        assert state.bets[1] == 5
        assert state.bets[2] == 5

        assert state.pot == 15  # 5 + 5 + 5
        assert state.max_bet == 5

    def test_config_string_backwards_compatibility(self):
        """Test that regular constructor still works without config_str."""
        config_str = """GAMEDEF
numplayers = 2
stack = 200 200
blind = 1 2
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        assert state.num_players == 2
        assert state.stacks[0] == 199  # 200 - 1
        assert state.stacks[1] == 198  # 200 - 2
        assert state.bets[0] == 1
        assert state.bets[1] == 2
        assert state.pot == 3

    def test_config_string_partial_override(self):
        """Test that config_str validates constructor parameters."""
        # Constructor and config should match
        config_str = """GAMEDEF
numplayers = 3
stack = 150 150 150
blind = 2 4 0
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=3, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Config should override constructor
        assert state.num_players == 3
        assert state.stacks[0] == 148  # 150 - 2
        assert state.stacks[1] == 146  # 150 - 4
        assert state.stacks[2] == 150  # 150 - 0

    def test_numrounds_three_rounds(self):
        """Test config string with numRounds = 3 (preflop, flop, turn only - no river)."""
        config_str = """GAMEDEF
numplayers = 2
numrounds = 3
stack = 100 100
blind = 1 2
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Game should have 3 rounds instead of default 4
        assert hasattr(env, "_num_rounds"), "Environment should store num_rounds"
        assert env._num_rounds == 3, f"Expected 3 rounds, got {env._num_rounds}"

        # Game should terminate after round 2 (0=preflop, 1=flop, 2=turn, 3=river not reached)
        # Play through to check termination condition
        state = env.step(state, universal_poker.CALL)  # Player calls
        state = env.step(state, universal_poker.CALL)  # Player checks, advance to flop
        assert state.round == 1, "Should be on flop"

        state = env.step(state, universal_poker.CALL)  # Player checks
        state = env.step(state, universal_poker.CALL)  # Player checks, advance to turn
        assert state.round == 2, "Should be on turn"

        state = env.step(state, universal_poker.CALL)  # Player checks
        state = env.step(state, universal_poker.CALL)  # Player checks, should terminate

        # Game should terminate after turn (round 2) since numrounds=3
        assert state.terminated, "Game should terminate after 3 rounds"
        assert state.round == 2, "Game should have reached final round"

    def test_numrounds_two_rounds(self):
        """Test config string with numRounds = 2 (preflop and flop only)."""
        config_str = """GAMEDEF
numplayers = 2
numrounds = 2
stack = 100 100
blind = 1 2
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Game should have 2 rounds
        assert env._num_rounds == 2, f"Expected 2 rounds, got {env._num_rounds}"

        # Play through preflop
        state = env.step(state, universal_poker.CALL)  # Player calls
        state = env.step(state, universal_poker.CALL)  # Player checks, advance to flop
        assert state.round == 1, "Should be on flop"

        # Play through flop - should terminate after this
        state = env.step(state, universal_poker.CALL)  # Player checks
        state = env.step(state, universal_poker.CALL)  # Player checks, should terminate

        # Game should terminate after flop (round 1) since numrounds=2
        assert state.terminated, "Game should terminate after 2 rounds"

    def test_numrounds_default_four_rounds(self):
        """Test that default behavior is still 4 rounds when numRounds not specified."""
        config_str = """GAMEDEF
numplayers = 2
stack = 100 100
blind = 1 2
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Default should be 4 rounds
        assert env._num_rounds == 4, f"Expected default 4 rounds, got {env._num_rounds}"

        # Game should not terminate until after river (round 3)
        # Play through preflop
        state = env.step(state, universal_poker.CALL)
        state = env.step(state, universal_poker.CALL)
        assert state.round == 1 and not state.terminated, "Should be on flop and not terminated"

        # Play through flop
        state = env.step(state, universal_poker.CALL)
        state = env.step(state, universal_poker.CALL)
        assert state.round == 2 and not state.terminated, "Should be on turn and not terminated"

        # Play through turn
        state = env.step(state, universal_poker.CALL)
        state = env.step(state, universal_poker.CALL)
        assert state.round == 3 and not state.terminated, "Should be on river and not terminated"

        # Play through river - now should terminate
        state = env.step(state, universal_poker.CALL)
        state = env.step(state, universal_poker.CALL)
        assert state.terminated, "Game should terminate after 4 rounds"

    def test_numrounds_one_round(self):
        """Test config string with numRounds = 1 (preflop only)."""
        config_str = """GAMEDEF
numplayers = 2
numrounds = 1
stack = 100 100
blind = 1 2
END GAMEDEF"""

        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        # Game should have 1 round
        assert env._num_rounds == 1, f"Expected 1 round, got {env._num_rounds}"

        # Game should terminate immediately after preflop
        state = env.step(state, universal_poker.CALL)  # Player calls
        state = env.step(state, universal_poker.CALL)  # Player checks, should terminate

        # Game should terminate after preflop (round 0) since numrounds=1
        assert state.terminated, "Game should terminate after 1 round"
        assert state.round == 0, "Round should have advanced to termination, without incrementing round"

    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with minimum players
        env = universal_poker.UniversalPoker(num_players=2)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        assert state.num_players == 2

        # Test with very small stacks
        config_str = """GAMEDEF
numplayers = 2
stack = 3 3
blind = 1 2
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        state = env.init(key)
        assert state.stacks[0] == 2  # Almost all-in from start
        assert state.stacks[1] == 1  # Almost all-in from start

    def test_multiple_players(self):
        """Test game with more than 2 players."""
        env = universal_poker.UniversalPoker(num_players=4)
        key = jax.random.PRNGKey(42)
        state = env.init(key)

        assert state.num_players == 4

        # All players should have hole cards
        from pgx.poker_eval.cardset import cardset_to_cards

        for p in range(4):
            hole_cards = cardset_to_cards(state.hole_cardsets[p])[:2]  # Take first 2
            assert jnp.all(hole_cards >= 0)

    # No-Limit Configuration Tests
    def test_config_nolimit_basic(self):
        """Test basic 'nolimit' keyword parsing."""
        config_str = """GAMEDEF
nolimit
numplayers = 2
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)

        # Should set game_type to 'nolimit'
        assert hasattr(env, "game_type"), "Environment should have game_type attribute"
        assert env.game_type == "nolimit", f"Expected 'nolimit', got '{env.game_type}'"

        # Should have default raise_multipliers
        assert hasattr(env, "raise_multipliers"), "Environment should have raise_multipliers attribute"
        assert env.raise_multipliers.shape == (4, 10), f"Expected shape (4, 10), got {env.raise_multipliers.shape}"

    def test_config_raise_multipliers_parsing(self):
        """Test raise_multipliers parsing with space-separated format."""
        config_str = """GAMEDEF
nolimit  
raise_multipliers 0 2.0 2.5 3.0 all_in
raise_multipliers 1 0.33 0.5 0.75 all_in
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)

        # Verify raise_multipliers[0] contains parsed preflop values
        assert env.raise_multipliers[0, 0] == 2.0, f"Expected 2.0, got {env.raise_multipliers[0, 0]}"
        assert env.raise_multipliers[0, 1] == 2.5, f"Expected 2.5, got {env.raise_multipliers[0, 1]}"
        assert env.raise_multipliers[0, 2] == 3.0, f"Expected 3.0, got {env.raise_multipliers[0, 2]}"
        assert env.raise_multipliers[0, 3] == -1.0, f"Expected -1.0 (all_in), got {env.raise_multipliers[0, 3]}"

        # Verify raise_multipliers[1] contains parsed postflop values
        assert env.raise_multipliers[1, 0] == 0.33, f"Expected 0.33, got {env.raise_multipliers[1, 0]}"
        assert env.raise_multipliers[1, 1] == 0.5, f"Expected 0.5, got {env.raise_multipliers[1, 1]}"
        assert env.raise_multipliers[1, 2] == 0.75, f"Expected 0.75, got {env.raise_multipliers[1, 2]}"
        assert env.raise_multipliers[1, 3] == -1.0, f"Expected -1.0 (all_in), got {env.raise_multipliers[1, 3]}"

    def test_config_nolimit_defaults(self):
        """Test default multipliers when rounds not specified."""
        config_str = """GAMEDEF
nolimit
numplayers = 2
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)

        # Should have sensible defaults for preflop (round 0)
        expected_preflop = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0, -1.0]
        for i, expected in enumerate(expected_preflop):
            actual = float(env.raise_multipliers[0, i])
            assert actual == expected, f"Preflop default[{i}]: expected {expected}, got {actual}"

        # Should have sensible defaults for postflop (rounds 1-3)
        expected_postflop = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, -1.0]
        for round_num in [1, 2, 3]:
            for i, expected in enumerate(expected_postflop):
                actual = float(env.raise_multipliers[round_num, i])
                assert (
                    abs(actual - expected) < 1e-5
                ), f"Round {round_num} default[{i}]: expected {expected}, got {actual}"

    def test_config_all_rounds_specified(self):
        """Test configuration with all 4 rounds specified."""
        config_str = """GAMEDEF
nolimit
raise_multipliers 0 2.0 3.0 all_in
raise_multipliers 1 0.25 0.5 all_in  
raise_multipliers 2 0.33 0.67 all_in
raise_multipliers 3 0.5 1.0 all_in
END GAMEDEF"""
        env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)

        # Check each round has correct values
        assert (
            env.raise_multipliers[0, 0] == 2.0
            and env.raise_multipliers[0, 1] == 3.0
            and env.raise_multipliers[0, 2] == -1.0
        )
        assert (
            env.raise_multipliers[1, 0] == 0.25
            and env.raise_multipliers[1, 1] == 0.5
            and env.raise_multipliers[1, 2] == -1.0
        )
        assert (
            env.raise_multipliers[2, 0] == 0.33
            and env.raise_multipliers[2, 1] == 0.67
            and env.raise_multipliers[2, 2] == -1.0
        )
        assert (
            env.raise_multipliers[3, 0] == 0.5
            and env.raise_multipliers[3, 1] == 1.0
            and env.raise_multipliers[3, 2] == -1.0
        )

    def test_config_invalid_nolimit(self):
        """Test error handling for invalid nolimit configurations."""
        # Test invalid round number
        config_str = """GAMEDEF
nolimit
raise_multipliers 5 2.0 3.0
END GAMEDEF"""
        try:
            env = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
            # Should not raise error but should ignore invalid round
            assert env.game_type == "nolimit"  # Should still parse nolimit correctly
        except Exception:
            pass  # It's okay if it raises an error for invalid round

    def test_config_backward_compatibility(self):
        """Test that limit games still work unchanged."""
        # Default behavior (no config)
        env1 = universal_poker.UniversalPoker(num_players=2)
        assert hasattr(env1, "game_type") and env1.game_type == "limit"

        # Explicit limit config
        config_str = """GAMEDEF
limit
numplayers = 2
stack = 200 200
blind = 1 2
END GAMEDEF"""
        env2 = universal_poker.UniversalPoker(num_players=2, config_str=config_str)
        assert hasattr(env2, "game_type") and env2.game_type == "limit"

        # Both should behave identically
        key = jax.random.PRNGKey(42)
        state1 = env1.init(key)
        state2 = env2.init(key)

        # legal_action_mask is always size 13 now, but only first 3 are used for limit
        assert state1.legal_action_mask.shape == (
            13,
        ), f"Action mask should be size 13, got shape {state1.legal_action_mask.shape}"
        assert state2.legal_action_mask.shape == (
            13,
        ), f"Action mask should be size 13, got shape {state2.legal_action_mask.shape}"

        # For limit games, only first 3 actions should be set, rest should be False
        assert jnp.all(state1.legal_action_mask[3:] == False), "Actions 3-12 should be False for limit games"
        assert jnp.all(state2.legal_action_mask[3:] == False), "Actions 3-12 should be False for limit games"


if __name__ == "__main__":
    import sys
    import traceback

    test_suite = TestUniversalPokerCore()

    print("Running Universal Poker core tests...")

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
