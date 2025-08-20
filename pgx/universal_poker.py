import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from .poker_eval.jax_evaluator_new import evaluate_hand
from .poker_eval.cardset import cards_to_cardset, cardset_to_cards, add_card_to_cardset, create_empty_cardset, cardset_or, cardset_and, cardset_not

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
    
    # Cards (using cardset representation for memory efficiency)
    hole_cardsets: Array = jnp.zeros((MAX_PLAYERS, 2), dtype=jnp.uint32)  # Each player's hole cards as cardset uint32[2]
    board_cardset: Array = jnp.zeros(2, dtype=jnp.uint32)  # Board cards as cardset uint32[2]
    
    # Pre-computed visible board cardsets for each round (optimization for observation)
    visible_board_cardsets: Array = jnp.zeros((4, 2), dtype=jnp.uint32)  # [round, cardset] for rounds 0-3
    
    # Pre-computed hand evaluations for final showdown
    hand_final_scores: Array = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)  # Final hand strength for each player
    
    # Betting info
    first_player: Array = jnp.array([1, 0, 0, 0], dtype=jnp.int32)  # first to act each round
    max_bet: Array = jnp.int32(0)
    
    # Action tracking
    num_actions_this_round: Array = jnp.int32(0)
    last_raiser: Array = jnp.int32(-1)
    
    # Pre-computed masks for performance optimization
    player_mask: Array = jnp.zeros(MAX_PLAYERS, dtype=jnp.bool_)  # Which positions are valid players
    active_mask: Array = jnp.zeros(MAX_PLAYERS, dtype=jnp.bool_)  # Which players are not folded/all-in
    
    @property
    def env_id(self) -> core.EnvId:
        return "universal_poker"


class UniversalPoker(core.Env):
    def __init__(self, num_players: int = 2, stack_size: int = 200, small_blind: int = 1, big_blind: int = 2, config_str: str = None):
        super().__init__()
        
        # Set default values
        self._num_players = num_players
        
        # Initialize stack_sizes as MAX_PLAYERS array
        self.stack_sizes = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)
        self.stack_sizes = self.stack_sizes.at[:num_players].set(stack_size)
        
        # Initialize blind_amounts as MAX_PLAYERS array with default small/big blind structure
        self.blind_amounts = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)
        self.blind_amounts = self.blind_amounts.at[0].set(small_blind)  # Small blind
        self.blind_amounts = self.blind_amounts.at[1].set(big_blind)    # Big blind
        
        # Parse config string if provided - this will override the defaults above
        if config_str is not None:
            self._parse_config_string(config_str)
    
    def _parse_config_string(self, config_str: str):
        """Parse GAMEDEF config string similar to C++ readGame function."""
        if not config_str.strip().startswith("GAMEDEF"):
            raise ValueError("Config string must start with 'GAMEDEF'")
        if not config_str.strip().endswith("END GAMEDEF"):
            raise ValueError("Config string must end with 'END GAMEDEF'")
        
        lines = config_str.strip().split('\n')
        
        for line in lines:
            line = line.strip().lower()
            
            # Skip comments and empty lines
            if line.startswith('#') or not line or line == "GAMEDEF" or line == "END GAMEDEF":
                continue
            
            # Parse each configuration line
            if line.startswith("stack"):
                # stack = 200 200 or stack = 200
                values = self._parse_config_line(line, "stack")
                if values:
                    # Reset stack_sizes array and populate with config values
                    self.stack_sizes = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)
                    num_values = min(len(values), MAX_PLAYERS)
                    self.stack_sizes = self.stack_sizes.at[:num_values].set(jnp.array(values[:num_values]))
                
            elif line.startswith("blind"):
                # blind = 1 2 0 0 ...
                values = self._parse_config_line(line, "blind")
                if values:
                    # Reset blind_amounts array and populate with config values
                    self.blind_amounts = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)
                    num_values = min(len(values), MAX_PLAYERS)
                    self.blind_amounts = self.blind_amounts.at[:num_values].set(jnp.array(values[:num_values]))
                
            elif line.startswith("numplayers"):
                # numplayers = 2
                values = self._parse_config_line(line, "numplayers")
                self._num_players = values[0]
    
    def _parse_config_line(self, line: str, key: str):
        """Parse a config line like 'key = value1 value2 ...' and return list of integer values."""
        try:
            # Remove the key and find the '=' sign
            if '=' in line:
                _, value_part = line.split('=', 1)
                value_part = value_part.strip()
                
                # Split values and convert to integers
                if value_part:
                    values = [int(x.strip()) for x in value_part.split() if x.strip()]
                    return values
            return []
        except ValueError:
            raise ValueError(f"Invalid config line: {line}")
        
    def _init(self, key: PRNGKey) -> State:
        return _init(key, self._num_players, self.stack_sizes, self.blind_amounts)
    
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


def _init(rng: PRNGKey, num_players: int, stack_sizes, blind_amounts) -> State:
    """Initialize a new poker hand."""
    # Initialize stacks using stack_sizes (already MAX_PLAYERS sized with zeros for inactive players)
    stacks = jnp.array(stack_sizes, dtype=jnp.int32)
    
    # Post blinds using blind_amounts array (already MAX_PLAYERS sized)
    bets = jnp.array(blind_amounts, dtype=jnp.int32)
    
    # Update stacks after posting blinds
    stacks = stacks - bets
    
    pot = jnp.int32(jnp.sum(bets))
    max_bet = jnp.int32(jnp.max(bets))
    
    # Deal hole cards using vectorized approach - deal to all MAX_PLAYERS positions
    rng, subkey = jax.random.split(rng)
    deck = jax.random.permutation(subkey, jnp.arange(52))
    
    # Deal 2 hole cards to each of MAX_PLAYERS positions (vectorized)
    all_hole_cards = deck[:MAX_PLAYERS*2].reshape(MAX_PLAYERS, 2)
    hole_cardsets = jax.vmap(cards_to_cardset)(all_hole_cards)
    
    # Deal 5 board cards 
    board_cards = deck[MAX_PLAYERS*2:MAX_PLAYERS*2+5]
    board_cardset = cards_to_cardset(board_cards)
    
    # Pre-compute final hand evaluations for all players (vectorized)
    # Tile board_cardset to match hole_cardsets shape: (MAX_PLAYERS, 2)
    board_cardsets_tiled = jnp.tile(board_cardset[None, :], (MAX_PLAYERS, 1))
    
    # Vectorized OR operation: combine all hole cards with board
    combined_cardsets = cardset_or(hole_cardsets, board_cardsets_tiled)
    
    # Vectorized hand evaluation over all combined cardsets
    hand_final_scores = jax.vmap(evaluate_hand)(combined_cardsets)
    
    # Pre-compute visible board cardsets for each round (optimization for observation)
    visible_board_cardsets = jnp.array([
        create_empty_cardset(),
        cards_to_cardset(board_cards[:3]),
        cards_to_cardset(board_cards[:4]),
	board_cardset
    ], dtype=jnp.uint32)
    
    # Initialize pre-computed masks for performance optimization
    player_mask = jnp.arange(MAX_PLAYERS) < num_players
    active_mask = player_mask.copy()  # Initially all players are active (not folded/all-in)
    
    # Determine first player to act (after big blind in preflop)
    current_player = jnp.int32((2 % num_players) if num_players > 2 else 0)
    
    state = State(
        num_players=jnp.int32(num_players),
        stacks=stacks,
        bets=bets,
        pot=pot,
        max_bet=max_bet,
        hole_cardsets=hole_cardsets,
        board_cardset=board_cardset,
        visible_board_cardsets=visible_board_cardsets,
        hand_final_scores=hand_final_scores,
        current_player=jnp.int32(current_player),
        player_mask=player_mask,
        active_mask=active_mask,
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
    
    # Check if game is terminated (using pre-computed masks)
    num_active = jnp.sum(new_state.active_mask)
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
    
    # Calculate amounts for call/raise actions (ensure int32 types)
    call_amount = jnp.int32(state.max_bet - state.bets[current_player])
    actual_call = jnp.minimum(call_amount, state.stacks[current_player])
    
    # Use the maximum blind amount as the minimum bet when max_bet is 0
    min_blind = jnp.max(state.bets)  # Get the largest blind that was posted
    min_raise = jnp.where(state.max_bet == 0, min_blind, jnp.int32(state.max_bet * 2))
    raise_amount = jnp.int32(min_raise - state.bets[current_player])
    actual_raise = jnp.minimum(raise_amount, state.stacks[current_player])
    
    # Determine final amounts based on action
    chips_to_add = jnp.where(is_call, actual_call, jnp.where(is_raise, actual_raise, jnp.int32(0)))
    
    # Update state arrays (ensure int32 types)
    new_folded = state.folded.at[current_player].set(state.folded[current_player] | is_fold)
    new_bets = state.bets.at[current_player].add(chips_to_add)
    new_stacks = state.stacks.at[current_player].subtract(chips_to_add)
    new_pot = jnp.int32(state.pot + chips_to_add)
    new_max_bet = jnp.where(is_raise, jnp.maximum(state.max_bet, new_bets[current_player]), state.max_bet)
    new_all_in = state.all_in.at[current_player].set(new_stacks[current_player] == 0)
    new_last_raiser = jnp.where(is_raise, jnp.int32(current_player), state.last_raiser)
    
    return state.replace(
        folded=new_folded,
        bets=new_bets,
        stacks=new_stacks,
        pot=new_pot,
        max_bet=new_max_bet,
        all_in=new_all_in,
        last_raiser=new_last_raiser,
        active_mask=(~new_folded & ~new_all_in & state.player_mask)
    )


def _is_betting_round_over(state: State) -> bool:
    """Check if the current betting round is over. (Phase 3.1: Optimized betting logic)"""
    # Use pre-computed active mask
    active_mask = state.active_mask
    
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
    new_round = jnp.int32(state.round + 1)
    
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
    num_actions_this_round = jnp.int32(state.num_actions_this_round + 1)
    
    return state.replace(
        current_player=next_player,
        num_actions_this_round=num_actions_this_round
    )


def _get_next_active_player(state: State) -> int:
    """Get the next active player (not folded or all-in). (Phase 2.1: Vectorized player finding)"""
    current = state.current_player
    
    # Use pre-computed active mask
    active_mask = state.active_mask
    
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
    next_player = jnp.int32(jnp.argmax(priorities))
    
    return next_player


def _get_first_player_for_round(state: State, round: int) -> int:
    """Get the first player to act in a given round."""
    # In post-flop rounds, small blind acts first
    # Preflop: player after big blind acts first
    start_pos = jnp.where(round > 0, jnp.int32(0), jnp.int32(2 % state.num_players))
    return _get_next_active_player_from(state, start_pos)


def _get_next_active_player_from(state: State, start_pos: int) -> int:
    """Get next active player starting from a position. (Phase 2.1: Vectorized player finding)"""
    # Use pre-computed active mask
    active_mask = state.active_mask
    
    # Create priority array: higher values for players that come after start_pos
    distances = (jnp.arange(MAX_PLAYERS) - start_pos) % MAX_PLAYERS
    priorities = MAX_PLAYERS - distances
    
    # Set inactive players to have very low priority
    priorities = jnp.where(active_mask, priorities, -1)
    
    # Find player with highest priority (closest active player from start_pos)
    next_player = jnp.int32(jnp.argmax(priorities))
    
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
    """Calculate final rewards for all players with proper side pot distribution."""
    # Get active players (not folded) - use pre-computed masks
    active_mask = (~state.folded & state.player_mask)
    num_active = jnp.sum(active_mask)
    
    # Handle single winner case (early fold scenarios)
    is_single_winner = num_active == 1
    reached_showdown = state.round >= 4
    is_showdown = (~is_single_winner) & reached_showdown
    
    def single_winner_case():
        single_winner_idx = jnp.int32(jnp.argmax(active_mask))
        rewards = jnp.zeros(MAX_PLAYERS, dtype=jnp.float32)
        return rewards.at[single_winner_idx].set(state.pot)
    
    def side_pot_calculation():
        # Use contributions (bets) and hand strengths for active players only - keep as uint32
        contributions = jnp.where(active_mask, state.bets, jnp.uint32(0))
        hand_strengths = jnp.where(active_mask, state.hand_final_scores, jnp.uint32(0))
        
        # --- 1. Identify Pot Layers ---
        # Find unique contribution levels to define pot boundaries
        # JAX requires concrete size for unique() - use large fill value to avoid duplicates
        all_pot_levels = jnp.unique(jnp.concatenate([jnp.array([0], dtype=jnp.uint32), contributions]), size=MAX_PLAYERS+1, fill_value=jnp.uint32(999999))
        # Only use levels that are not the fill value
        valid_levels_mask = all_pot_levels < 999999
        # Calculate increments only for consecutive valid levels  
        level_increments = jnp.diff(all_pot_levels, prepend=jnp.uint32(0))[1:]
        # Zero out increments for invalid transitions
        level_increments = jnp.where(valid_levels_mask[1:] & valid_levels_mask[:-1], level_increments, jnp.uint32(0))
        pot_levels = all_pot_levels
        
        # --- 2. Determine Player Eligibility for Each Pot Layer ---
        # Create a 2D boolean mask (num_players x num_pot_layers)
        # eligible_mask[i, j] is True if player `i` is eligible for pot layer `j`
        eligible_mask = contributions[:, jnp.newaxis] >= pot_levels[jnp.newaxis, 1:]
        
        # --- 3. Calculate the Size of Each Pot Layer ---
        # Count eligible players for each layer
        num_eligible_players = eligible_mask.sum(axis=0).astype(jnp.uint32)
        # Calculate the size of each pot layer
        pot_layer_sizes = level_increments * num_eligible_players
        
        # --- 4. Find the Winner(s) of Each Pot Layer ---
        # Mask out ineligible players' hand strengths by setting them to 0
        masked_strengths = jnp.where(eligible_mask, hand_strengths[:, jnp.newaxis], jnp.uint32(0))
        # Find the maximum hand strength for each pot layer
        max_strength_per_pot = masked_strengths.max(axis=0)
        # Identify all players with the winning hand strength for each pot (eligible winners only)
        is_winner_mask = eligible_mask & (masked_strengths == max_strength_per_pot)
        
        # --- 5. Distribute Winnings ---
        # Count winners for each pot to handle ties
        num_winners_per_pot = is_winner_mask.sum(axis=0).astype(jnp.uint32)
        # Avoid division by zero
        safe_num_winners = jnp.where(num_winners_per_pot > 0, num_winners_per_pot, jnp.uint32(1))
        # Calculate the payout per winner for each pot layer (integer division)
        payout_per_winner = pot_layer_sizes // safe_num_winners
        # Distribute the pot layer amounts to the winners
        winnings_matrix = jnp.where(is_winner_mask, payout_per_winner[jnp.newaxis, :], jnp.uint32(0))
        # Sum the winnings from all pot layers for each player
        total_winnings = winnings_matrix.sum(axis=1)
        
        # Convert to float32 for final return
        return total_winnings.astype(jnp.float32)
    
    def equal_split_case():
        equal_pot_share = jnp.float32(state.pot) / jnp.maximum(num_active, 1)
        return jnp.where(active_mask, equal_pot_share, 0.0)
    
    # Calculate final rewards based on game state
    final_rewards = jnp.where(
        is_single_winner,
        single_winner_case(),
        jnp.where(
            is_showdown,
            side_pot_calculation(),
            equal_split_case()
        )
    )
    
    # Return the appropriate slice for backward compatibility with existing reward array size
    return final_rewards[:state.rewards.shape[0]]




def _observe(state: State, player_id: int) -> Array:
    """Generate observation for a specific player."""
    # Own hole cards as cardset
    hole_cardset = state.hole_cardsets[player_id]
    
    # Visible board cards as cardset - use pre-computed values (optimized)
    visible_board_cardset = state.visible_board_cardsets[state.round]
    
    # Build observation array with appropriate types
    # Format: [hole_cardset[2], board_cardset[2], pot, stack, bets[10], folded[10], round]
    obs = jnp.concatenate([
        # Cardsets (flatten uint32[2] arrays for concatenation)
        hole_cardset.astype(jnp.int32),
        visible_board_cardset.astype(jnp.int32),
        # Game state (int32)
        jnp.array([state.pot, state.stacks[player_id]], dtype=jnp.int32),
        # Current bets (int32)
        state.bets[:MAX_PLAYERS],
        # Folded status (convert bool to int32)
        state.folded[:MAX_PLAYERS].astype(jnp.int32),
        # Current round (int32)
        jnp.array([state.round], dtype=jnp.int32)
    ])
    
    return obs
