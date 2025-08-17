import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

# Action types
FOLD = jnp.int32(0)
CALL = jnp.int32(1)
RAISE = jnp.int32(2)

# Maximum number of players supported
MAX_PLAYERS = 10

# Hand strength rankings (from weakest to strongest)
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(0, dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(3, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    
    # Universal Poker specific fields
    num_players: Array = jnp.int32(2)
    num_hole_cards: Array = jnp.int32(2)
    num_board_cards: Array = jnp.array([0, 3, 1, 1], dtype=jnp.int32)  # preflop, flop, turn, river
    num_ranks: Array = jnp.int32(13)
    num_suits: Array = jnp.int32(4)
    
    # Game state
    round: Array = jnp.int32(0)  # 0=preflop, 1=flop, 2=turn, 3=river
    pot: Array = jnp.int32(0)
    stacks: Array = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)
    bets: Array = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)
    folded: Array = jnp.zeros(MAX_PLAYERS, dtype=jnp.bool_)
    all_in: Array = jnp.zeros(MAX_PLAYERS, dtype=jnp.bool_)
    
    # Cards
    hole_cards: Array = jnp.full((MAX_PLAYERS, 3), -1, dtype=jnp.int32)  # Allow up to 3 hole cards
    board_cards: Array = jnp.full(7, -1, dtype=jnp.int32)  # Max 7 board cards
    
    # Betting info
    first_player: Array = jnp.array([1, 0, 0, 0], dtype=jnp.int32)  # first to act each round
    small_blind: Array = jnp.int32(1)
    big_blind: Array = jnp.int32(2)
    max_bet: Array = jnp.int32(0)
    
    # Action tracking
    num_actions_this_round: Array = jnp.int32(0)
    last_raiser: Array = jnp.int32(-1)
    
    @property
    def env_id(self) -> core.EnvId:
        return "universal_poker"


class UniversalPoker(core.Env):
    def __init__(self, num_players: int = 2, stack_size: int = 200, small_blind: int = 1, big_blind: int = 2):
        super().__init__()
        self._num_players = num_players
        self.stack_size = stack_size
        self.small_blind = small_blind
        self.big_blind = big_blind
        
    def _init(self, key: PRNGKey) -> State:
        return _init(key, self._num_players, self.stack_size, self.small_blind, self.big_blind)
    
    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        return _step(state, action)
    
    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)
    
    @property
    def id(self) -> core.EnvId:
        return "universal_poker"
    
    @property
    def version(self) -> str:
        return "v1"
    
    @property
    def num_players(self) -> int:
        return self._num_players


def _init(rng: PRNGKey, num_players: int, stack_size: int, small_blind: int, big_blind: int) -> State:
    """Initialize a new poker hand."""
    # Initialize stacks
    stacks = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)
    stacks = stacks.at[:num_players].set(stack_size)
    
    # Post blinds
    bets = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)
    bets = bets.at[0].set(small_blind)  # Small blind
    bets = bets.at[1].set(big_blind)    # Big blind
    
    # Update stacks after posting blinds
    stacks = stacks.at[0].subtract(small_blind)
    stacks = stacks.at[1].subtract(big_blind)
    
    pot = small_blind + big_blind
    max_bet = big_blind
    
    # Deal hole cards
    rng, subkey = jax.random.split(rng)
    deck = jax.random.permutation(subkey, jnp.arange(52))
    
    hole_cards = jnp.full((MAX_PLAYERS, 3), -1, dtype=jnp.int32)
    card_idx = 0
    for p in range(num_players):
        for i in range(2):  # Deal 2 hole cards per player
            hole_cards = hole_cards.at[p, i].set(deck[card_idx])
            card_idx += 1
    
    # Deal board cards (will be revealed in later rounds)
    board_cards = jnp.full(7, -1, dtype=jnp.int32)
    for i in range(5):  # Deal 5 board cards total
        board_cards = board_cards.at[i].set(deck[card_idx])
        card_idx += 1
    
    # Determine first player to act (after big blind in preflop)
    current_player = (2 % num_players) if num_players > 2 else 0
    
    state = State(
        num_players=jnp.int32(num_players),
        stacks=stacks,
        bets=bets,
        pot=pot,
        max_bet=max_bet,
        hole_cards=hole_cards,
        board_cards=board_cards,
        current_player=jnp.int32(current_player),
        small_blind=jnp.int32(small_blind),
        big_blind=jnp.int32(big_blind),
        rewards=jnp.zeros(2, dtype=jnp.float32),  # Fixed size for compatibility
    )
    
    # Set legal actions
    legal_action_mask = _get_legal_actions(state)
    state = state.replace(legal_action_mask=legal_action_mask)
    
    return state


def _step(state: State, action: int) -> State:
    """Execute one step of the game."""
    action = jnp.int32(action)
    current_player = state.current_player
    
    # Apply action
    new_state = _apply_action(state, action)
    
    # Check if betting round is over
    betting_round_over = _is_betting_round_over(new_state)
    
    # If betting round is over, advance to next round or end game
    new_state = jax.lax.cond(
        betting_round_over,
        lambda s: _advance_round(s),
        lambda s: _next_player(s),
        new_state
    )
    
    # Check if game is terminated
    player_mask = jnp.arange(MAX_PLAYERS) < new_state.num_players
    active_mask = (~new_state.folded & player_mask)[:MAX_PLAYERS]
    num_active = jnp.sum(active_mask)
    terminated = (num_active <= 1) | (new_state.round >= 4)
    
    # Calculate rewards and legal actions in single conditional (Phase 1.1: Merge termination checks)
    def terminated_updates():
        final_rewards = _calculate_rewards(new_state)
        final_legal_actions = jnp.zeros(3, dtype=jnp.bool_)
        return final_rewards, final_legal_actions
    
    def active_updates():
        active_rewards = new_state.rewards
        active_legal_actions = _get_legal_actions(new_state)
        return active_rewards, active_legal_actions
    
    rewards, legal_action_mask = jax.lax.cond(
        terminated,
        terminated_updates,
        active_updates
    )
    
    return new_state.replace(
        terminated=terminated,
        rewards=rewards,
        legal_action_mask=legal_action_mask
    )


def _apply_action(state: State, action: int) -> State:
    """Apply the given action to the current state. (Phase 1.2: Vectorized action application)"""
    current_player = state.current_player
    
    # Create action masks
    is_fold = action == FOLD
    is_call = action == CALL
    is_raise = action == RAISE
    
    # Calculate amounts for call/raise actions
    call_amount = state.max_bet - state.bets[current_player]
    actual_call = jnp.minimum(call_amount, state.stacks[current_player])
    
    min_raise = jnp.where(state.max_bet == 0, state.big_blind, state.max_bet * 2)
    raise_amount = min_raise - state.bets[current_player]
    actual_raise = jnp.minimum(raise_amount, state.stacks[current_player])
    
    # Determine final amounts based on action
    chips_to_add = jnp.where(is_call, actual_call, jnp.where(is_raise, actual_raise, 0))
    
    # Update state arrays
    new_folded = state.folded.at[current_player].set(state.folded[current_player] | is_fold)
    new_bets = state.bets.at[current_player].add(chips_to_add)
    new_stacks = state.stacks.at[current_player].subtract(chips_to_add)
    new_pot = state.pot + chips_to_add
    new_max_bet = jnp.where(is_raise, jnp.maximum(state.max_bet, new_bets[current_player]), state.max_bet)
    new_all_in = state.all_in.at[current_player].set(new_stacks[current_player] == 0)
    new_last_raiser = jnp.where(is_raise, current_player, state.last_raiser)
    
    return state.replace(
        folded=new_folded,
        bets=new_bets,
        stacks=new_stacks,
        pot=new_pot,
        max_bet=new_max_bet,
        all_in=new_all_in,
        last_raiser=new_last_raiser
    )


def _is_betting_round_over(state: State) -> bool:
    """Check if the current betting round is over. (Phase 3.1: Optimized betting logic)"""
    # Precompute masks once
    player_mask = jnp.arange(MAX_PLAYERS) < state.num_players
    active_mask = (~state.folded & ~state.all_in & player_mask)[:MAX_PLAYERS]
    
    # If less than 2 active players, round is over
    active_count = jnp.sum(active_mask)
    
    # For betting check, only compare bets of active players directly with max_bet
    # This avoids creating intermediate arrays
    active_bets_match_max = jnp.where(
        active_mask, 
        state.bets[:MAX_PLAYERS] == state.max_bet, 
        True  # Inactive players don't affect the condition
    )
    all_active_called = jnp.all(active_bets_match_max)
    
    # Combine conditions efficiently
    return (active_count < 2) | (all_active_called & (state.num_actions_this_round > 0))


def _advance_round(state: State) -> State:
    """Advance to the next betting round."""
    new_round = state.round + 1
    
    # Reset betting for new round
    bets = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)
    max_bet = jnp.int32(0)
    num_actions_this_round = jnp.int32(0)
    last_raiser = jnp.int32(-1)
    
    # Determine first player for new round
    current_player = _get_first_player_for_round(state, new_round)
    
    return state.replace(
        round=new_round,
        bets=bets,
        max_bet=max_bet,
        current_player=current_player,
        num_actions_this_round=num_actions_this_round,
        last_raiser=last_raiser
    )


def _next_player(state: State) -> State:
    """Move to the next player."""
    next_player = _get_next_active_player(state)
    num_actions_this_round = state.num_actions_this_round + 1
    
    return state.replace(
        current_player=next_player,
        num_actions_this_round=num_actions_this_round
    )


def _get_next_active_player(state: State) -> int:
    """Get the next active player (not folded or all-in). (Phase 2.1: Vectorized player finding)"""
    current = state.current_player
    
    # Create a mask for active players
    player_mask = jnp.arange(MAX_PLAYERS) < state.num_players
    active_mask = (~state.folded & ~state.all_in & player_mask)[:MAX_PLAYERS]
    
    # Create priority array: higher values for players that come after current player
    # Players after current get priority (MAX_PLAYERS - distance), others get (2*MAX_PLAYERS - distance)
    distances = (jnp.arange(MAX_PLAYERS) - current - 1) % MAX_PLAYERS
    priorities = jnp.where(
        jnp.arange(MAX_PLAYERS) > current,
        MAX_PLAYERS - distances,
        2 * MAX_PLAYERS - distances
    )
    
    # Set inactive players to have very low priority
    priorities = jnp.where(active_mask, priorities, -1)
    
    # Find player with highest priority (closest active player after current)
    next_player = jnp.argmax(priorities)
    
    return next_player


def _get_first_player_for_round(state: State, round: int) -> int:
    """Get the first player to act in a given round."""
    # In post-flop rounds, small blind acts first
    # Preflop: player after big blind acts first
    start_pos = jnp.where(round > 0, 0, 2 % state.num_players)
    return _get_next_active_player_from(state, start_pos)


def _get_next_active_player_from(state: State, start_pos: int) -> int:
    """Get next active player starting from a position. (Phase 2.1: Vectorized player finding)"""
    # Create a mask for active players
    player_mask = jnp.arange(MAX_PLAYERS) < state.num_players
    active_mask = (~state.folded & ~state.all_in & player_mask)[:MAX_PLAYERS]
    
    # Create priority array: higher values for players that come after start_pos
    distances = (jnp.arange(MAX_PLAYERS) - start_pos) % MAX_PLAYERS
    priorities = MAX_PLAYERS - distances
    
    # Set inactive players to have very low priority
    priorities = jnp.where(active_mask, priorities, -1)
    
    # Find player with highest priority (closest active player from start_pos)
    next_player = jnp.argmax(priorities)
    
    return next_player


def _get_legal_actions(state: State) -> Array:
    """Get legal actions for current player. (Phase 4: Batch legal action computation)"""
    current_player = state.current_player
    
    # Calculate legal actions regardless of termination state
    can_fold = ~state.all_in[current_player]
    can_call = (state.bets[current_player] < state.max_bet) & (state.stacks[current_player] > 0)
    total_chips = state.stacks[current_player] + state.bets[current_player]
    can_raise = total_chips > state.max_bet
    
    active_actions = jnp.array([can_fold, can_call, can_raise], dtype=jnp.bool_)
    
    # If terminated, return all False; otherwise return computed actions
    return jnp.where(state.terminated, False, active_actions)


def _calculate_rewards(state: State) -> Array:
    """Calculate final rewards for all players."""
    # Use the same shape as the input rewards
    rewards_shape = state.rewards.shape[0]
    rewards = jnp.zeros(rewards_shape, dtype=jnp.float32)
    
    # Get active players (not folded)
    player_mask = jnp.arange(rewards_shape) < state.num_players
    active_mask = (~state.folded[:rewards_shape] & player_mask)
    num_active = jnp.sum(active_mask)
    
    # Vectorized reward calculation (Phase 2.2: Eliminate nested conditionals)
    is_single_winner = num_active == 1
    reached_showdown = state.round >= 4
    is_showdown = (~is_single_winner) & reached_showdown
    is_equal_split = (~is_single_winner) & (~reached_showdown)
    
    # Single winner case: winner gets all the pot
    single_winner_idx = jnp.argmax(active_mask)
    single_winner_rewards = jnp.zeros(rewards_shape, dtype=jnp.float32).at[single_winner_idx].set(state.pot)
    
    # Showdown case: evaluate hands and award based on strength
    def eval_player_hand(p):
        in_range = p < state.num_players
        return jnp.where(
            active_mask[p] & in_range,
            _evaluate_hand(state.hole_cards[p], state.board_cards, state.round),
            jnp.int32(-1)  # Inactive players get -1
        )
    
    hand_strengths = jax.vmap(eval_player_hand)(jnp.arange(rewards_shape))
    best_strength = jnp.max(jnp.where(active_mask, hand_strengths, -1))
    showdown_winners_mask = (hand_strengths == best_strength) & active_mask
    showdown_num_winners = jnp.sum(showdown_winners_mask)
    showdown_pot_share = state.pot // jnp.maximum(showdown_num_winners, 1)
    showdown_rewards = jnp.where(showdown_winners_mask, showdown_pot_share, 0.0).astype(jnp.float32)
    
    # Equal split case: split pot among all active players
    equal_num_winners = jnp.sum(active_mask)
    equal_pot_share = state.pot // jnp.maximum(equal_num_winners, 1)
    equal_split_rewards = jnp.where(active_mask, equal_pot_share, 0.0).astype(jnp.float32)
    
    # Select final rewards based on case
    final_rewards = jnp.where(
        is_single_winner,
        single_winner_rewards,
        jnp.where(is_showdown, showdown_rewards, equal_split_rewards)
    )
    
    return final_rewards


def _evaluate_hand(hole_cards: Array, board_cards: Array, round: int) -> int:
    """Evaluate hand strength using proper poker hand evaluation."""
    # Import the evaluator (done inside function to avoid circular imports)
    try:
        from .poker_eval.jax_evaluator import evaluate_hand_jax
    except ImportError:
        # Fallback for relative import issues
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from poker_eval.jax_evaluator import evaluate_hand_jax
    
    # Number of board cards visible in each round: preflop=0, flop=3, turn=4, river=5
    num_board_cards = jnp.array([0, 3, 4, 5], dtype=jnp.int32)  # Cumulative counts
    
    # Get number of visible board cards for this round
    num_visible = jnp.where(round >= 4, 5, num_board_cards[round])
    
    # Create a mask for visible board cards
    visible_mask = jnp.arange(5) < num_visible
    visible_board = jnp.where(visible_mask, board_cards[:5], -1)
    
    # Combine hole cards and visible board cards
    all_cards = jnp.concatenate([hole_cards[:2], visible_board])
    
    # Unified hand evaluation (Phase 4: Batch hand evaluation)
    # For preflop, use high card value; otherwise use full evaluation
    preflop_value = jnp.max(hole_cards[:2])
    postflop_value = evaluate_hand_jax(all_cards)
    
    return jnp.where(num_visible == 0, preflop_value, postflop_value)


def _observe(state: State, player_id: int) -> Array:
    """Generate observation for a specific player."""
    # Observation includes:
    # - Own hole cards (one-hot encoded)
    # - Visible board cards (one-hot encoded) 
    # - Pot size (normalized)
    # - Own stack (normalized)
    # - Current bets for all players (normalized)
    # - Folded status for all players
    # - Current round
    
    obs_size = (
        52 +  # Own hole cards (2 cards, one-hot)
        52 +  # Board cards (up to 5, one-hot)
        1 +   # Pot size (normalized)
        1 +   # Own stack (normalized)
        MAX_PLAYERS +  # Current bets (normalized)
        MAX_PLAYERS +  # Folded status
        4     # Round (one-hot)
    )
    
    # Vectorized observation encoding (Phase 3.2: Eliminate loops and conditionals)
    obs = jnp.zeros(obs_size, dtype=jnp.float32)
    
    # Precompute common values
    total_chips = jnp.sum(state.stacks) + state.pot
    safe_total_chips = jnp.maximum(total_chips, 1)
    safe_pot = jnp.maximum(state.pot, 1)
    
    # Own hole cards - vectorized one-hot encoding
    hole_cards = state.hole_cards[player_id, :2]
    valid_holes = hole_cards >= 0
    safe_holes = jnp.clip(hole_cards, 0, 51)
    hole_indices = jnp.where(valid_holes, safe_holes, 52)  # Use out-of-bounds for invalid
    obs = obs.at[hole_indices].add(valid_holes.astype(jnp.float32))
    
    # Visible board cards - vectorized
    num_board_cards = jnp.array([0, 3, 4, 5], dtype=jnp.int32)
    num_visible = jnp.where(state.round >= 4, 5, num_board_cards[state.round])
    board_cards = state.board_cards[:5]
    visible_mask = jnp.arange(5) < num_visible
    valid_board = (board_cards >= 0) & visible_mask
    safe_board = jnp.clip(board_cards, 0, 51)
    board_indices = jnp.where(valid_board, safe_board + 52, obs_size)  # Offset by 52, use out-of-bounds for invalid
    obs = obs.at[board_indices].add(valid_board.astype(jnp.float32))
    
    # Build observation array in segments
    segments = [
        # Pot and stack (normalized)
        jnp.array([state.pot / safe_total_chips, state.stacks[player_id] / safe_total_chips]),
        # Current bets (normalized)
        state.bets[:MAX_PLAYERS] / safe_pot,
        # Folded status
        state.folded[:MAX_PLAYERS].astype(jnp.float32),
        # Current round (one-hot)
        jnp.eye(4)[jnp.clip(state.round, 0, 3)] * (state.round < 4)
    ]
    
    # Concatenate all segments efficiently
    numeric_obs = jnp.concatenate(segments)
    obs = obs.at[104:].set(numeric_obs)  # Start after the 2x52 card sections
    
    return obs
