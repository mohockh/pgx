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
    
    # Calculate rewards if terminated
    final_rewards = _calculate_rewards(new_state)
    rewards = jax.lax.select(terminated, final_rewards, new_state.rewards)
    
    # Update legal actions
    legal_action_mask = jax.lax.cond(
        terminated,
        lambda: jnp.zeros(3, dtype=jnp.bool_),
        lambda: _get_legal_actions(new_state)
    )
    
    return new_state.replace(
        terminated=terminated,
        rewards=rewards,
        legal_action_mask=legal_action_mask
    )


def _apply_action(state: State, action: int) -> State:
    """Apply the given action to the current state."""
    current_player = state.current_player
    
    def fold_action():
        folded = state.folded.at[current_player].set(TRUE)
        return state.replace(folded=folded)
    
    def call_action():
        call_amount = state.max_bet - state.bets[current_player]
        actual_call = jnp.minimum(call_amount, state.stacks[current_player])
        
        bets = state.bets.at[current_player].add(actual_call)
        stacks = state.stacks.at[current_player].subtract(actual_call)
        pot = state.pot + actual_call
        all_in = state.all_in.at[current_player].set(stacks[current_player] == 0)
        
        return state.replace(bets=bets, stacks=stacks, pot=pot, all_in=all_in)
    
    def raise_action():
        raise_amount = state.max_bet * 2 - state.bets[current_player]
        actual_raise = jnp.minimum(raise_amount, state.stacks[current_player])
        
        bets = state.bets.at[current_player].add(actual_raise)
        stacks = state.stacks.at[current_player].subtract(actual_raise)
        pot = state.pot + actual_raise
        max_bet = jnp.maximum(state.max_bet, bets[current_player])
        all_in = state.all_in.at[current_player].set(stacks[current_player] == 0)
        last_raiser = current_player
        
        return state.replace(
            bets=bets, stacks=stacks, pot=pot, max_bet=max_bet, 
            all_in=all_in, last_raiser=last_raiser
        )
    
    return jax.lax.switch(
        action,
        [fold_action, call_action, raise_action]
    )


def _is_betting_round_over(state: State) -> bool:
    """Check if the current betting round is over."""
    # Round is over if all active players have either folded, called, or are all-in
    player_mask = jnp.arange(MAX_PLAYERS) < state.num_players
    active_mask = (~state.folded & ~state.all_in & player_mask)[:MAX_PLAYERS]
    active_players = jnp.sum(active_mask)
    
    # If less than 2 active players, round is over
    round_over_few_players = active_players < 2
    
    # Check if all active players have bet the same amount
    player_bets = jnp.where(player_mask, state.bets, state.max_bet)[:MAX_PLAYERS]
    active_bets = jnp.where(active_mask, player_bets, state.max_bet)
    all_bets_equal = jnp.all(active_bets == state.max_bet)
    
    # Also need to check that at least one action has been taken this round
    # (to prevent immediate end at start of round)
    round_over_all_called = all_bets_equal & (state.num_actions_this_round > 0)
    
    return round_over_few_players | round_over_all_called


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
    """Get the next active player (not folded or all-in)."""
    current = state.current_player
    
    # Create a mask for active players using static slicing
    player_mask = jnp.arange(MAX_PLAYERS) < state.num_players
    active_mask = (~state.folded & ~state.all_in & player_mask)[:MAX_PLAYERS]
    
    # Create an array of potential next players
    candidates = (current + 1 + jnp.arange(MAX_PLAYERS)) % MAX_PLAYERS
    
    # Use scan to find first active player
    def find_active(carry, i):
        found, player = carry
        candidate = candidates[i]
        is_candidate_active = active_mask[candidate] & (candidate < state.num_players)
        new_found = found | is_candidate_active
        new_player = jax.lax.select(found, player, candidate)
        return (new_found, new_player), None
    
    (found, next_player), _ = jax.lax.scan(find_active, (FALSE, current), jnp.arange(MAX_PLAYERS))
    
    return next_player


def _get_first_player_for_round(state: State, round: int) -> int:
    """Get the first player to act in a given round."""
    # In post-flop rounds, small blind acts first
    # Preflop: player after big blind acts first
    start_pos = jax.lax.select(round > 0, 0, 2 % state.num_players)
    return _get_next_active_player_from(state, start_pos)


def _get_next_active_player_from(state: State, start_pos: int) -> int:
    """Get next active player starting from a position."""
    # Use static slicing with MAX_PLAYERS and then mask by num_players
    player_mask = jnp.arange(MAX_PLAYERS) < state.num_players
    active_mask = (~state.folded & ~state.all_in & player_mask)[:MAX_PLAYERS]
    
    # Create an array of potential players
    candidates = (start_pos + jnp.arange(MAX_PLAYERS)) % MAX_PLAYERS
    
    # Use scan to find first active player
    def find_active(carry, i):
        found, player = carry
        candidate = candidates[i]
        is_candidate_active = active_mask[candidate] & (candidate < state.num_players)
        new_found = found | is_candidate_active
        new_player = jax.lax.select(found, player, candidate)
        return (new_found, new_player), None
    
    (found, next_player), _ = jax.lax.scan(find_active, (FALSE, start_pos), jnp.arange(MAX_PLAYERS))
    
    return next_player


def _get_legal_actions(state: State) -> Array:
    """Get legal actions for current player."""
    terminated_actions = jnp.zeros(3, dtype=jnp.bool_)
    
    def get_active_actions():
        current_player = state.current_player
        
        # Can always fold (unless already all-in)
        can_fold = ~state.all_in[current_player]
        
        # Can call if there's a bet to call and player has chips
        can_call = (state.bets[current_player] < state.max_bet) & (state.stacks[current_player] > 0)
        
        # Can raise if player has enough chips and hasn't reached max bet
        min_raise = state.max_bet * 2 - state.bets[current_player]
        can_raise = (state.stacks[current_player] >= min_raise) & (state.max_bet < state.stacks[current_player])
        
        return jnp.array([can_fold, can_call, can_raise], dtype=jnp.bool_)
    
    return jax.lax.cond(
        state.terminated,
        lambda: terminated_actions,
        get_active_actions
    )


def _calculate_rewards(state: State) -> Array:
    """Calculate final rewards for all players."""
    # Use the same shape as the input rewards
    rewards_shape = state.rewards.shape[0]
    rewards = jnp.zeros(rewards_shape, dtype=jnp.float32)
    
    # Get active players (not folded)
    player_mask = jnp.arange(rewards_shape) < state.num_players
    active_mask = (~state.folded[:rewards_shape] & player_mask)
    num_active = jnp.sum(active_mask)
    
    # If only one player remains, they win the pot
    def single_winner():
        winner = jnp.argmax(active_mask)
        return rewards.at[winner].set(state.pot)
    
    # If multiple players remain, determine winner by hand strength
    def multiple_winners():
        # Evaluate hand strengths for active players using vectorized operations
        def eval_player_hand(p):
            # Only evaluate if player is in range and active
            in_range = p < state.num_players
            return jax.lax.cond(
                active_mask[p] & in_range,
                lambda: _evaluate_hand(state.hole_cards[p], state.board_cards, state.round),
                lambda: jnp.int32(-1)  # Folded players get -1
            )
        
        hand_strengths = jax.vmap(eval_player_hand)(jnp.arange(rewards_shape))
        
        # Find the best hand strength among active players
        best_strength = jnp.max(jnp.where(active_mask, hand_strengths, -1))
        
        # All players with best strength split the pot
        winners_mask = (hand_strengths == best_strength) & active_mask
        num_winners = jnp.sum(winners_mask)
        pot_share = state.pot // jnp.maximum(num_winners, 1)  # Avoid division by zero
        
        return jnp.where(winners_mask, pot_share, 0.0).astype(jnp.float32)
    
    final_rewards = jax.lax.cond(
        num_active == 1,
        single_winner,
        multiple_winners
    )
    
    return final_rewards


def _evaluate_hand(hole_cards: Array, board_cards: Array, round: int) -> int:
    """Evaluate hand strength. Returns higher values for better hands."""
    # Simple hand evaluation - in a full implementation this would be much more complex
    # For now, just use highest card value as a placeholder
    
    # Number of board cards visible in each round: preflop=0, flop=3, turn=1, river=1
    num_board_cards = jnp.array([0, 3, 4, 5], dtype=jnp.int32)  # Cumulative counts
    
    # Get number of visible board cards for this round
    num_visible = jax.lax.select(round >= 4, 5, num_board_cards[round])
    
    # Create a mask for visible board cards
    visible_mask = jnp.arange(5) < num_visible
    visible_board = jnp.where(visible_mask, board_cards[:5], -1)
    
    # Combine hole cards and visible board cards
    all_cards = jnp.concatenate([hole_cards[:2], visible_board])
    valid_cards = jnp.where(all_cards >= 0, all_cards, 0)
    
    # Return highest card as simple hand strength
    return jnp.max(valid_cards)


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
    
    obs = jnp.zeros(obs_size, dtype=jnp.float32)
    idx = 0
    
    # Own hole cards - create one-hot encoding
    hole_cards = state.hole_cards[player_id, :2]
    for i in range(2):
        card = hole_cards[i]
        valid = card >= 0
        safe_card = jnp.clip(card, 0, 51)  # Ensure valid index
        obs = obs.at[idx + safe_card].set(obs[idx + safe_card] + jnp.where(valid, 1.0, 0.0))
    idx += 52
    
    # Visible board cards
    # Cumulative board cards: preflop=0, flop=3, turn=4, river=5
    num_board_cards = jnp.array([0, 3, 4, 5], dtype=jnp.int32)
    num_visible = jax.lax.select(state.round >= 4, 5, num_board_cards[state.round])
    
    for i in range(5):
        visible = i < num_visible
        card = state.board_cards[i]
        valid = (card >= 0) & visible
        safe_card = jnp.clip(card, 0, 51)  # Ensure valid index
        obs = obs.at[idx + safe_card].set(obs[idx + safe_card] + jnp.where(valid, 1.0, 0.0))
    idx += 52
    
    # Pot size (normalized by total chips in game)
    total_chips = jnp.sum(state.stacks) + state.pot
    obs = obs.at[idx].set(state.pot / jnp.maximum(total_chips, 1))
    idx += 1
    
    # Own stack (normalized)
    obs = obs.at[idx].set(state.stacks[player_id] / jnp.maximum(total_chips, 1))
    idx += 1
    
    # Current bets (normalized)
    for i in range(MAX_PLAYERS):
        obs = obs.at[idx + i].set(state.bets[i] / jnp.maximum(state.pot, 1))
    idx += MAX_PLAYERS
    
    # Folded status
    for i in range(MAX_PLAYERS):
        obs = obs.at[idx + i].set(state.folded[i])
    idx += MAX_PLAYERS
    
    # Current round (one-hot)
    valid_round = state.round < 4
    safe_round = jnp.clip(state.round, 0, 3)
    obs = obs.at[idx + safe_round].set(jnp.where(valid_round, 1.0, 0.0))
    
    return obs
