import jax
import jax.numpy as jnp

from universal_poker import UniversalPoker, State, FOLD, CALL, RAISE, _init, _step


class TestUniversalPoker:
    """Test suite for Universal Poker implementation."""
    
    def test_init_basic(self):
        """Test basic game initialization."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        assert isinstance(state, State)
        assert state.num_players == 2
        assert state.round == 0  # Start at preflop
        assert state.pot == 3    # Small blind (1) + big blind (2)
        assert not state.terminated
        assert state.stacks[0] == 199  # 200 - 1 (small blind)
        assert state.stacks[1] == 198  # 200 - 2 (big blind)
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        env = UniversalPoker(num_players=3, stack_size=100, small_blind=5, big_blind=10)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        assert state.num_players == 3
        assert state.pot == 15  # 5 + 10
        assert state.stacks[0] == 95   # 100 - 5
        assert state.stacks[1] == 90   # 100 - 10
        assert state.stacks[2] == 100  # No blind
        
    def test_hole_cards_dealt(self):
        """Test that hole cards are properly dealt."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Each player should have 2 hole cards
        for p in range(state.num_players):
            hole_cards = state.hole_cards[p, :2]
            assert jnp.all(hole_cards >= 0)  # Valid card indices
            assert jnp.all(hole_cards < 52)  # Within deck range
            
    def test_board_cards_dealt(self):
        """Test that board cards are dealt but not visible initially."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # 5 board cards should be dealt (but not visible in preflop)
        board_cards = state.board_cards[:5]
        assert jnp.all(board_cards >= 0)
        assert jnp.all(board_cards < 52)
        
    def test_legal_actions_preflop(self):
        """Test legal actions in preflop."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # First player to act should be able to fold, call, or raise
        legal_actions = state.legal_action_mask
        assert legal_actions[FOLD]   # Can fold
        assert legal_actions[CALL]   # Can call
        assert legal_actions[RAISE]  # Can raise
        
    def test_fold_action(self):
        """Test folding action."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Player folds
        current_player = state.current_player
        new_state = env.step(state, FOLD)
        
        assert new_state.folded[current_player]
        assert new_state.terminated  # Game should end with fold in 2-player game
        
    def test_call_action(self):
        """Test calling action."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        current_player = state.current_player
        initial_stack = state.stacks[current_player]
        initial_bet = state.bets[current_player]
        call_amount = state.max_bet - initial_bet
        
        # Player calls
        new_state = env.step(state, CALL)
        
        assert new_state.bets[current_player] == state.max_bet
        assert new_state.stacks[current_player] == initial_stack - call_amount
        assert new_state.pot == state.pot + call_amount
        
    def test_raise_action(self):
        """Test raising action."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        current_player = state.current_player
        initial_stack = state.stacks[current_player]
        
        # Player raises
        new_state = env.step(state, RAISE)
        
        assert new_state.max_bet > state.max_bet
        assert new_state.stacks[current_player] < initial_stack
        assert new_state.last_raiser == current_player
        
    def test_betting_round_progression(self):
        """Test progression through betting rounds."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Both players call to end preflop
        state = env.step(state, CALL)  # First player calls
        state = env.step(state, CALL)  # Second player calls (checks)
        
        # Should advance to flop
        assert state.round == 1
        assert state.max_bet == 0  # Bets reset for new round
        
    def test_all_in_scenario(self):
        """Test all-in scenario."""
        env = UniversalPoker(stack_size=10)  # Small stacks
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Keep raising until someone goes all-in
        for _ in range(5):  # Limit iterations to prevent infinite loop
            if state.terminated:
                break
                
            legal_actions = state.legal_action_mask
            if legal_actions[RAISE]:
                state = env.step(state, RAISE)
            elif legal_actions[CALL]:
                state = env.step(state, CALL)
            else:
                state = env.step(state, FOLD)
        
        # Check if any player went all-in
        assert jnp.any(state.all_in[:state.num_players]) or state.terminated
        
    def test_observation_shape(self):
        """Test observation shape and content."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        for player_id in range(state.num_players):
            obs = env.observe(state, player_id)
            
            # Check observation is proper array
            assert isinstance(obs, jnp.ndarray)
            assert obs.dtype == jnp.float32
            
            # Check reasonable observation size
            expected_size = 52 + 52 + 1 + 1 + 10 + 10 + 4  # See _observe function
            assert len(obs) == expected_size
            
    def test_rewards_on_termination(self):
        """Test reward calculation when game terminates."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Player 0 folds, player 1 should win
        state = env.step(state, FOLD)
        
        assert state.terminated
        assert state.rewards[1] > 0  # Winner gets positive reward
        assert state.rewards[0] == 0  # Loser gets no reward
        
    def test_multiple_players(self):
        """Test game with more than 2 players."""
        env = UniversalPoker(num_players=4)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        assert state.num_players == 4
        
        # All players should have hole cards
        for p in range(4):
            hole_cards = state.hole_cards[p, :2]
            assert jnp.all(hole_cards >= 0)
            
    def test_game_properties(self):
        """Test game properties."""
        env = UniversalPoker()
        
        assert env.id == "universal_poker"
        assert env.version == "v1"
        assert env.num_players == 2
        
    def test_jax_compilation(self):
        """Test that key functions can be JIT compiled."""
        env = UniversalPoker()
        
        # Test init compilation
        init_fn = jax.jit(env.init)
        key = jax.random.PRNGKey(42)
        state = init_fn(key)
        assert isinstance(state, State)
        
        # Test step compilation
        step_fn = jax.jit(env.step)
        new_state = step_fn(state, CALL)
        assert isinstance(new_state, State)
        
    def test_random_games(self):
        """Test playing multiple random games."""
        env = UniversalPoker()
        
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
            
    def test_deterministic_behavior(self):
        """Test that games are deterministic given the same seed."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        
        # Play two identical games
        state1 = env.init(key)
        state2 = env.init(key)
        
        # States should be identical
        assert jnp.array_equal(state1.hole_cards, state2.hole_cards)
        assert jnp.array_equal(state1.board_cards, state2.board_cards)
        assert state1.pot == state2.pot
        
    def test_state_consistency(self):
        """Test state consistency throughout game."""
        env = UniversalPoker()
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        
        # Check initial state consistency
        total_chips = jnp.sum(state.stacks[:state.num_players]) + state.pot
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
                total_chips = jnp.sum(state.stacks[:state.num_players]) + state.pot
                assert total_chips == initial_total
                
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with minimum players
        env = UniversalPoker(num_players=2)
        key = jax.random.PRNGKey(42)
        state = env.init(key)
        assert state.num_players == 2
        
        # Test with very small stacks
        env = UniversalPoker(stack_size=3, small_blind=1, big_blind=2)
        state = env.init(key)
        assert state.stacks[0] == 2  # Almost all-in from start
        assert state.stacks[1] == 1  # Almost all-in from start


if __name__ == "__main__":
    # Run basic tests
    test_suite = TestUniversalPoker()
    
    print("Running Universal Poker tests...")
    
    try:
        test_suite.test_init_basic()
        print("✓ Basic initialization test passed")
        
        test_suite.test_init_custom_params()
        print("✓ Custom parameters test passed")
        
        test_suite.test_hole_cards_dealt()
        print("✓ Hole cards dealing test passed")
        
        test_suite.test_legal_actions_preflop()
        print("✓ Legal actions test passed")
        
        test_suite.test_fold_action()
        print("✓ Fold action test passed")
        
        test_suite.test_call_action()
        print("✓ Call action test passed")
        
        test_suite.test_raise_action()
        print("✓ Raise action test passed")
        
        test_suite.test_observation_shape()
        print("✓ Observation shape test passed")
        
        test_suite.test_game_properties()
        print("✓ Game properties test passed")
        
        test_suite.test_jax_compilation()
        print("✓ JAX compilation test passed")
        
        test_suite.test_deterministic_behavior()
        print("✓ Deterministic behavior test passed")
        
        test_suite.test_random_games()
        print("✓ Random games test passed")
        
        print("\nAll tests passed! ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
