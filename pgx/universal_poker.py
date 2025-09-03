import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from .poker_eval.evaluator import evaluate_hand
from .poker_eval.cardset import cards_to_cardset, create_empty_cardset, cardset_or

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

# Action types
FOLD = jnp.uint32(0)
CALL = jnp.uint32(1)
RAISE = jnp.uint32(2)

# Maximum number of players supported
MAX_PLAYERS = 10


@dataclass
class State(core.State):
    current_player: Array = jnp.uint32(0)
    observation: Array = jnp.zeros(0, dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(3, dtype=jnp.bool_)
    _step_count: Array = jnp.uint32(0)

    # Universal Poker specific fields
    num_players: Array = jnp.uint32(2)
    num_hole_cards: Array = jnp.uint32(2)
    num_board_cards: Array = jnp.array([0, 3, 1, 1], dtype=jnp.uint32)  # preflop, flop, turn, river
    num_ranks: Array = jnp.uint32(13)
    num_suits: Array = jnp.uint32(4)

    # Game state
    round: Array = jnp.uint32(0)  # 0=preflop, 1=flop, 2=turn, 3=river
    pot: Array = jnp.uint32(0)
    stacks: Array = None  # Number of players elements (uint32).
    bets: Array = None  # Number of players elements (uint32).
    previous_round_bets: Array = None  # Cumulative bets from previous rounds (uint32).
    folded: Array = None  # Number of players elements (bool).
    all_in: Array = None  # Number of players elements.

    # Cards (using cardset representation for memory efficiency)
    hole_cardsets: Array = None  # Each player's hole cards as cardset uint32[2]
    board_cardset: Array = jnp.zeros(2, dtype=jnp.uint32)  # Board cards as cardset uint32[2]

    # Pre-computed visible board cardsets for each round (optimization for observation)
    visible_board_cardsets: Array = jnp.zeros((4, 2), dtype=jnp.uint32)  # [round, cardset] for rounds 0-3

    # Pre-computed hand evaluations for final showdown
    hand_final_scores: Array = None  # Final hand strength for each player.

    # Betting info
    first_player: Array = jnp.array([3, 1, 1, 1], dtype=jnp.uint32)  # first to act each round
    max_bet: Array = jnp.uint32(0)
    min_raise: Array = jnp.uint32(0)  # Minimum raise amount

    # Action tracking
    num_actions_this_round: Array = jnp.uint32(0)
    last_raiser: Array = jnp.uint32(0)

    # Pre-computed masks for performance optimization
    player_mask: Array = None  # Which positions are valid players
    active_mask: Array = None  # Which players are not folded/all-in

    @property
    def env_id(self) -> core.EnvId:
        return "universal_poker"


class UniversalPoker(core.Env):
    def __init__(self, num_players: int = 2, config_str: str = None):
        super().__init__()

        # Set default values
        self._num_players = num_players
        assert num_players <= MAX_PLAYERS, f"{num_players} > {MAX_PLAYERS}"

        self._num_rounds = 4  # Default: preflop, flop, turn, river

        # Initialize stack_sizes as self._num_players array
        self.stack_sizes = 200 * jnp.ones(self._num_players, dtype=jnp.uint32)

        # Initialize blind_amounts as self._num_players array with default small/big blind structure
        self.blind_amounts = jnp.array(
            [1, 2]
            + [
                0,
            ]
            * (self._num_players - 2),
            dtype=jnp.uint32,
        )

        # Initialize first_player_array with default values
        # ACPC format uses 1-based indexing, so we convert to 0-based
        if self._num_players == 2:
            # Heads-up: dealer/SB acts first preflop, BB acts first post-flop
            # 1-based: [1, 2] -> 0-based: [0, 1]
            default_first_players = [0] + [1] * (self._num_rounds - 1)
        else:
            # Multi-way: hardcoded pattern [3, 1, 1, 1] in 1-based -> [2, 0, 0, 0] in 0-based
            preflop_first = (3 - 1) % self._num_players  # 3 -> 2 (0-based)
            postflop_first = (1 - 1) % self._num_players  # 1 -> 0 (0-based)
            default_first_players = [preflop_first] + [postflop_first] * (self._num_rounds - 1)
        self.first_player_array = jnp.array(default_first_players, dtype=jnp.uint32)

        # Parse config string if provided - this will override the defaults above
        if config_str is not None:
            self._parse_config_string(config_str)

    def _parse_config_string(self, config_str: str):
        """Parse GAMEDEF config string similar to C++ readGame function."""
        if not config_str.strip().startswith("GAMEDEF"):
            raise ValueError("Config string must start with 'GAMEDEF'")
        if not config_str.strip().endswith("END GAMEDEF"):
            raise ValueError("Config string must end with 'END GAMEDEF'")

        lines = config_str.strip().split("\n")

        for line in lines:
            line = line.strip().lower()

            # Skip comments and empty lines
            if line.startswith("#") or not line or line == "GAMEDEF" or line == "END GAMEDEF":
                continue

            # Parse each configuration line
            if line.startswith("stack"):
                # stack = 200 200 or stack = 200
                values = self._parse_config_line(line, "stack")
                if values:
                    # Reset stack_sizes array and populate with config values
                    self.stack_sizes = jnp.zeros(self._num_players, dtype=jnp.uint32)
                    num_values = min(len(values), self._num_players)
                    self.stack_sizes = self.stack_sizes.at[:num_values].set(
                        jnp.array(values[:num_values], dtype=jnp.uint32)
                    )

            elif line.startswith("blind"):
                # blind = 1 2 0 0 ...
                values = self._parse_config_line(line, "blind")
                if values:
                    # Reset blind_amounts array and populate with config values
                    self.blind_amounts = jnp.zeros(self._num_players, dtype=jnp.uint32)
                    num_values = min(len(values), self._num_players)
                    self.blind_amounts = self.blind_amounts.at[:num_values].set(
                        jnp.array(values[:num_values], dtype=jnp.uint32)
                    )

            elif line.startswith("numplayers"):
                # numplayers = 2
                values = self._parse_config_line(line, "numplayers")
                if values:
                    assert values[0] == self._num_players

            elif line.startswith("numrounds"):
                # numrounds = 4
                values = self._parse_config_line(line, "numrounds")
                if values:
                    self._num_rounds = values[0]
                    # Update first_player_array size when num_rounds changes
                    # ACPC format uses 1-based indexing, so we convert to 0-based
                    if self._num_players == 2:
                        # Heads-up: dealer/SB acts first preflop, BB acts first post-flop
                        default_first_players = [0] + [1] * (self._num_rounds - 1)
                    else:
                        # Multi-way: hardcoded pattern [3, 1, 1, 1] in 1-based -> [2, 0, 0, 0] in 0-based
                        preflop_first = (3 - 1) % self._num_players
                        postflop_first = (1 - 1) % self._num_players
                        default_first_players = [preflop_first] + [postflop_first] * (self._num_rounds - 1)
                    self.first_player_array = jnp.array(default_first_players, dtype=jnp.uint32)

            elif line.startswith("firstplayer"):
                # firstplayer = 3 1 1 1  (first player for each round, 1-based)
                values = self._parse_config_line(line, "firstplayer")
                if values:
                    # Convert from 1-based to 0-based indexing
                    zero_based_values = [(v - 1) % self._num_players for v in values]
                    # Create array with specified values, pad with zeros if needed
                    first_players = jnp.zeros(self._num_rounds, dtype=jnp.uint32)
                    num_values = min(len(zero_based_values), self._num_rounds)
                    self.first_player_array = first_players.at[:num_values].set(
                        jnp.array(zero_based_values[:num_values], dtype=jnp.uint32)
                    )

    def _parse_config_line(self, line: str, key: str):
        """Parse a config line like 'key = value1 value2 ...' and return list of integer values."""
        try:
            # Remove the key and find the '=' sign
            if "=" in line:
                _, value_part = line.split("=", 1)
                value_part = value_part.strip()

                # Split values and convert to integers
                if value_part:
                    values = [int(x.strip()) for x in value_part.split() if x.strip()]
                    return values
            return []
        except ValueError:
            raise ValueError(f"Invalid config line: {line}")

    def _init(self, rng: PRNGKey) -> State:
        """Initialize a new poker hand."""

        # Determine big blind amount for minimum participation check
        big_blind = jnp.max(self.blind_amounts)

        # Identify players who have enough chips to meaningfully participate
        initial_stacks = jnp.array(self.stack_sizes, dtype=jnp.uint32)
        sufficient_chips_mask = initial_stacks >= big_blind

        # Count active players
        num_active_players = jnp.sum(sufficient_chips_mask)
        should_terminate = num_active_players < 2

        # Auto-fold players with insufficient chips (vectorized)
        folded = ~sufficient_chips_mask

        # Deal hole cards using vectorized approach - deal to all num_players positions
        rng, subkey = jax.random.split(rng)
        deck = jax.random.permutation(subkey, jnp.arange(52))

        # Deal 2 hole cards to each of num_players positions (vectorized)
        all_hole_cards = deck[: self._num_players * 2].reshape(self._num_players, 2)
        hole_cardsets = jax.vmap(cards_to_cardset)(all_hole_cards)

        # Deal 5 board cards
        board_cards = deck[self._num_players * 2 : self._num_players * 2 + 5]
        board_cardset = cards_to_cardset(board_cards)

        # Pre-compute final hand evaluations for all players (vectorized)
        # Tile board_cardset to match hole_cardsets shape: (num_players, 2)
        board_cardsets_tiled = jnp.tile(board_cardset[None, :], (self._num_players, 1))

        # Vectorized OR operation: combine all hole cards with board
        combined_cardsets = cardset_or(hole_cardsets, board_cardsets_tiled)

        # Vectorized hand evaluation over all combined cardsets
        hand_final_scores = jax.vmap(evaluate_hand)(combined_cardsets)

        # Pre-compute visible board cardsets for each round (optimization for observation)
        visible_board_cardsets = jnp.array(
            [
                create_empty_cardset(),
                cards_to_cardset(board_cards[:3]),
                cards_to_cardset(board_cards[:4]),
                board_cardset,
            ],
            dtype=jnp.uint32,
        )

        # Initialize pre-computed masks for performance optimization
        player_mask = jnp.arange(self._num_players) < self._num_players

        # Create preliminary state with round = max uint32 (will advance to round 0)
        preliminary_state = State(
            round=jnp.uint32(4294967295),  # Will overflow to 0 when incremented in _advance_round
            num_players=self._num_players,
            stacks=initial_stacks,  # Use initial stacks before blind posting
            bets=jnp.zeros(self._num_players, dtype=jnp.uint32),  # No bets yet
            previous_round_bets=jnp.zeros(self._num_players, dtype=jnp.uint32),  # No previous bets yet
            folded=folded,
            all_in=jnp.zeros(self._num_players, dtype=jnp.bool_),  # No all-ins yet
            pot=jnp.uint32(0),  # No pot yet
            max_bet=jnp.uint32(0),  # No bets yet
            hole_cardsets=hole_cardsets,
            board_cardset=board_cardset,
            visible_board_cardsets=visible_board_cardsets,
            hand_final_scores=hand_final_scores,
            player_mask=player_mask,
            active_mask=~folded & player_mask,  # Initial active mask before all-ins
            rewards=jnp.zeros(self._num_players, dtype=jnp.float32),
            terminated=should_terminate,
        )

        # Use _advance_round to set up round 0 (sets min_raise, current_player, last_raiser, etc.)
        state_after_round_setup = self._advance_round(preliminary_state)

        # Now post blinds on the properly initialized round 0 state
        # Assign blinds to eligible players using vectorized operations
        num_blinds = jnp.sum(self.blind_amounts > 0)
        num_blinds_to_assign = jnp.minimum(num_blinds, num_active_players)

        # Use vectorized assignment for blind posting with JAX-compatible static indexing
        # Convert boolean mask to integer for eligible players
        is_eligible_int = sufficient_chips_mask.astype(jnp.uint32)

        # Get the 1-based order of eligible players using cumsum
        blind_order = jnp.cumsum(is_eligible_int)

        # Create 0-based indices for the blind_amounts array
        # We use jnp.maximum to avoid a -1 index for ineligible players
        blind_indices = jnp.maximum(0, blind_order - 1)

        # Gather the blinds according to the order, but cap at available blinds
        # Only assign blinds to players within the number of available blind slots
        should_get_blind = (blind_order > 0) & (blind_order <= num_blinds)
        assigned_blind_amounts = jnp.where(
            should_get_blind, jnp.take(self.blind_amounts, blind_indices, mode="clip"), 0
        )

        # Cap blind amounts at available stack for each player
        actual_blind_amounts = jnp.minimum(assigned_blind_amounts, initial_stacks)

        # Apply blinds to the round 0 state
        bets = actual_blind_amounts
        stacks_after_blinds = initial_stacks - bets
        pot = jnp.sum(bets)
        max_bet = jnp.max(bets)

        # Detect all-in players (those who used all their chips for blinds)
        all_in = stacks_after_blinds == 0

        # Update active mask to exclude folded and all-in players
        active_mask = ~folded & ~all_in & player_mask

        # Apply blind posting results to state
        state_with_blinds = state_after_round_setup.replace(
            stacks=stacks_after_blinds, bets=bets, pot=pot, max_bet=max_bet, all_in=all_in, active_mask=active_mask
        )

        # If game should terminate, calculate rewards immediately (vectorized)
        final_rewards = jax.lax.cond(
            should_terminate,
            lambda: self._calculate_rewards(state_with_blinds),
            lambda: jnp.zeros(self._num_players, dtype=jnp.float32),
        )

        # Create final state
        final_state = state_with_blinds.replace(rewards=final_rewards)

        # Set legal actions (will be empty if terminated)
        legal_action_mask = self._get_legal_actions(final_state)
        final_state = final_state.replace(legal_action_mask=legal_action_mask)

        return final_state

    def _step(self, state: core.State, action: Array, key) -> State:
        """Execute one step of the game."""
        del key
        assert isinstance(state, State)
        current_player = state.current_player

        # Apply action
        new_state = self._apply_action(state, action)

        # Check if game is terminated BEFORE advancing round/player (using pre-computed masks)
        num_active = jnp.sum(new_state.active_mask)
        betting_round_over = self._is_betting_round_over(new_state)
        in_last_betting_round = new_state.round == (self._num_rounds - 1)
        terminated = (num_active <= 1) | (betting_round_over & in_last_betting_round)

        # Combined function for terminated case: don't advance, calculate final rewards and legal actions
        def terminated_branch():
            final_rewards = self._calculate_rewards(new_state)
            final_legal_actions = jnp.ones(3, dtype=jnp.bool_)
            return new_state.replace(rewards=final_rewards, legal_action_mask=final_legal_actions)

        # Combined function for active case: advance game, keep current rewards and legal actions
        def active_branch():
            # Advance to next round or next player
            advanced_state = jax.lax.cond(
                betting_round_over, lambda s: self._advance_round(s), lambda s: self._next_player(s), new_state
            )

            # Since we already checked termination and we're in active branch,
            # game continues - calculate legal actions for advanced state
            legal_actions = self._get_legal_actions(advanced_state)

            return advanced_state.replace(rewards=advanced_state.rewards, legal_action_mask=legal_actions)

        new_state = new_state.replace(terminated=terminated)

        # Execute the appropriate branch based on initial termination status
        return jax.lax.cond(terminated, terminated_branch, active_branch)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        """Generate observation for a specific player."""
        assert isinstance(state, State)
        # Own hole cards as cardset
        hole_cardset = state.hole_cardsets[state.current_player]

        # Visible board cards as cardset - use pre-computed values (optimized)
        visible_board_cardset = state.visible_board_cardsets[state.round]

        # Build observation array with appropriate types
        # Format: [hole_cardset[2], board_cardset[2], pot, stack, bets[num_players], folded[num_players], round]
        obs = jnp.concatenate(
            [
                # Cardsets (already uint32[2] arrays for concatenation)
                hole_cardset,
                visible_board_cardset,
                # Game state (ensure uint32)
                jnp.array([state.pot, state.stacks[state.current_player]], dtype=jnp.uint32),
                # Current bets (ensure uint32)
                state.bets,
                # Folded status (convert bool to uint32)
                state.folded.astype(jnp.uint32),
                # Current round (ensure uint32)
                jnp.array([state.round], dtype=jnp.uint32),
            ]
        )

        return obs

    def _apply_action(self, state: State, action: int) -> State:
        """Apply the given action to the current state."""
        current_player = state.current_player

        # Create action masks
        is_fold = action == FOLD
        is_call = action == CALL
        is_raise = action == RAISE

        # Calculate amounts for call/raise actions (ensure uint32 types)
        call_amount = state.max_bet - state.bets[current_player]
        actual_call = jnp.minimum(call_amount, state.stacks[current_player])

        # Use stored min_raise amount
        min_raise = state.max_bet + state.min_raise
        raise_amount = min_raise - state.bets[current_player]
        actual_raise = jnp.minimum(raise_amount, state.stacks[current_player])

        # Determine final amounts based on action
        chips_to_add = jnp.where(is_call, actual_call, jnp.where(is_raise, actual_raise, 0)).astype(jnp.uint32)

        # Update state arrays (ensure uint32 types)
        new_folded = state.folded.at[current_player].set(state.folded[current_player] | is_fold)
        new_bets = state.bets.at[current_player].add(chips_to_add)
        new_stacks = state.stacks.at[current_player].subtract(chips_to_add)
        new_pot = state.pot + chips_to_add
        new_max_bet = jnp.where(is_raise, jnp.maximum(state.max_bet, new_bets[current_player]), state.max_bet)
        new_all_in = state.all_in.at[current_player].set(new_stacks[current_player] == 0)
        new_last_raiser = jnp.where(is_raise, jnp.uint32(current_player), state.last_raiser)

        # Update min_raise - calculate raise increment from previous player's bet
        bet_before = state.bets[(current_player - 1) % state.num_players]  # Get previous player's bet amount
        raise_increment = (new_bets[current_player] - bet_before).astype(
            jnp.uint32
        )  # This bet raised last player by raise_increment
        new_min_raise = jnp.maximum(state.min_raise, raise_increment)

        return state.replace(
            folded=new_folded,
            bets=new_bets,
            stacks=new_stacks,
            pot=new_pot,
            max_bet=new_max_bet,
            min_raise=new_min_raise,
            all_in=new_all_in,
            last_raiser=new_last_raiser,
            active_mask=(~new_folded & ~new_all_in & state.player_mask),
            num_actions_this_round=state.num_actions_this_round + 1,
        )

    def _is_betting_round_over(self, state: State) -> bool:
        """Check if the current betting round is over."""
        # Use pre-computed active mask
        active_mask = state.active_mask

        # If less than 2 active players, round is over
        active_count = jnp.sum(active_mask)

        # For betting check, only compare bets of active players directly with max_bet
        # This avoids creating intermediate arrays
        active_bets_match_max = jnp.where(
            active_mask, state.bets == state.max_bet, True  # Inactive players don't affect the condition
        )
        all_active_called = jnp.all(active_bets_match_max)

        # Combine conditions efficiently
        # Betting round is over if:
        # 1. Less than 2 active players, OR
        # 2. All active players have called AND all active players have had a chance to act
        all_players_acted = state.num_actions_this_round >= active_count
        return (active_count < 2) | (all_active_called & all_players_acted)

    def _advance_round(self, state: State) -> State:
        """Advance to the next betting round."""
        new_round = state.round + 1

        # Accumulate current round bets into previous_round_bets before reset
        new_previous_round_bets = state.previous_round_bets + state.bets

        # Reset betting for new round - use concrete shape
        bets = jnp.zeros(self._num_players, dtype=jnp.uint32)
        max_bet = jnp.uint32(0)
        num_actions_this_round = jnp.uint32(0)
        last_raiser = jnp.uint32(self._num_players)
        # Reset min_raise to big blind for new round
        big_blind = jnp.max(self.blind_amounts)
        min_raise = big_blind.astype(jnp.uint32)

        # Determine first player for new round
        current_player = self._get_first_player_for_round(state, new_round)

        return state.replace(
            round=new_round,
            bets=bets,
            previous_round_bets=new_previous_round_bets,
            max_bet=max_bet,
            min_raise=min_raise,
            current_player=current_player,
            num_actions_this_round=num_actions_this_round,
            last_raiser=last_raiser,
        )

    def _next_player(self, state: State) -> State:
        """Move to the next player."""
        next_player = self._get_next_active_player_from(state, state.current_player + 1)

        return state.replace(current_player=next_player)

    def _get_first_player_for_round(self, state: State, round: int) -> int:
        """Get the first player to act in a given round."""
        # Start searching for the first eligible player from first spot in
        # configured array first_player_array.
        start_pos = self.first_player_array[round]
        return self._get_next_active_player_from(state, start_pos)

    def _get_next_active_player_from(self, state: State, start_pos: int) -> int:
        """Get next active player starting from a position."""
        # Make sure start_pos wraps around the table given self._num_players.
        start_pos = start_pos % self._num_players

        # Use pre-computed active mask
        active_mask = state.active_mask

        # Create priority array: higher values for players that come after start_pos
        distances = (jnp.arange(self._num_players) - start_pos) % self._num_players
        priorities = self._num_players - distances

        # Set inactive players to have very low priority
        priorities = jnp.where(active_mask, priorities, jnp.int32(-1))

        # Find player with highest priority (closest active player from start_pos)
        next_player = jnp.argmax(priorities)

        return next_player

    def _get_legal_actions(self, state: State) -> Array:
        """Get legal actions for current player."""
        current_player = state.current_player

        # Calculate legal actions regardless of termination state
        can_fold = ~(state.all_in | state.folded)
        can_call = (state.bets <= state.max_bet) & (state.stacks > 0) & ~state.folded
        total_chips = state.stacks + state.bets
        can_raise = (total_chips > state.max_bet) & ~state.folded

        active_actions = jnp.column_stack([can_fold, can_call, can_raise])

        # Return computed actions for current_player only
        return active_actions[current_player]

    def _calculate_rewards(self, state: State) -> Array:
        """Calculate final rewards for all players with proper side pot distribution."""
        # Get active players (not folded) - use pre-computed masks
        active_mask = ~state.folded & state.player_mask
        num_active = jnp.sum(active_mask)

        # Handle single winner case (early fold scenarios)
        is_single_winner = num_active == 1
        reached_showdown = state.round >= self._num_rounds
        is_showdown = (~is_single_winner) & reached_showdown

        def single_winner_case():
            single_winner_idx = jnp.argmax(active_mask)
            rewards = jnp.zeros(self._num_players, dtype=jnp.float32)
            return rewards.at[single_winner_idx].set(state.pot)

        def side_pot_calculation():
            # Use total contributions (previous rounds + current round) from ALL players - keep as uint32
            total_contributions = state.previous_round_bets + state.bets
            # Only active players can win - mask out inactive players' hand strengths
            hand_strengths = jnp.where(active_mask, state.hand_final_scores, jnp.uint32(0))

            # --- 1. Identify Pot Layers ---
            # Find unique contribution levels to define pot boundaries
            # JAX requires concrete size for unique() - use large fill value to avoid duplicates
            all_pot_levels = jnp.unique(
                jnp.concatenate([jnp.array([0], dtype=jnp.uint32), total_contributions]),
                size=self._num_players + 1,
                fill_value=jnp.uint32(999999),
            )
            # Only use levels that are not the fill value
            valid_levels_mask = all_pot_levels < 999999
            # Calculate increments only for consecutive valid levels
            level_increments = jnp.diff(all_pot_levels, prepend=jnp.uint32(0))[1:]
            # Zero out increments for invalid transitions
            level_increments = jnp.where(
                valid_levels_mask[1:] & valid_levels_mask[:-1], level_increments, jnp.uint32(0)
            )
            pot_levels = all_pot_levels

            # --- 2. Determine Player Eligibility for Each Pot Layer ---
            # Create a 2D boolean mask (num_players x num_pot_layers)
            # eligible_mask[i, j] is True if player `i` is eligible for pot layer `j`
            eligible_mask = total_contributions[:, jnp.newaxis] >= pot_levels[jnp.newaxis, 1:]

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
            is_single_winner, single_winner_case(), jnp.where(is_showdown, side_pot_calculation(), equal_split_case())
        )

        # Return rewards for all players - now properly sized
        return final_rewards

    @property
    def id(self) -> core.EnvId:
        return "universal_poker"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self) -> int:
        return self._num_players
