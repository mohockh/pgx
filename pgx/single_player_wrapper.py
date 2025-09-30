"""Single-player wrapper for multi-player PGX environments.

Converts any multi-player PGX environment into a single-player interface by automatically
handling opponent turns. This enables direct use with single-agent RL algorithms like PPO.

Only depends on pgx.core API for maximum reusability.
"""
from typing import Callable, Optional, Sequence, Any

import jax
import jax.numpy as jnp
import flax.struct

import pgx.core as core
from pgx._src.types import Array, PRNGKey


@flax.struct.dataclass
class SPWState:
    """State for SinglePlayerWrapper with opponent parameters.

    This class wraps a PGX State and adds opponent network parameters
    and RNG keys, allowing JIT-compiled functions to handle dynamic
    opponent policies.

    Attributes:
        pgx_state: Core state from the underlying PGX environment
        opponent_params: PyTree containing network parameters for all opponent policies
        opponent_policy_rngs: PRNGKey for stochastic opponent policy sampling
    """
    # Core state from the underlying PGX environment
    pgx_state: core.State

    # Network parameters for all opponent policies
    opponent_params: Any  # PyTree

    # PRNGKeys for stochastic opponent policies
    opponent_policy_rngs: PRNGKey

    # --- Property delegations for compatibility ---
    # This makes SPWState behave like a standard PgxState
    # to the rest of the PPO algorithm.

    @property
    def current_player(self) -> Array:
        return self.pgx_state.current_player

    @property
    def observation(self) -> Array:
        return self.pgx_state.observation

    @property
    def rewards(self) -> Array:
        return self.pgx_state.rewards

    @property
    def terminated(self) -> Array:
        return self.pgx_state.terminated

    @property
    def truncated(self) -> Array:
        return self.pgx_state.truncated

    @property
    def legal_action_mask(self) -> Array:
        return self.pgx_state.legal_action_mask

    @property
    def _step_count(self) -> Array:
        """Delegate step count to underlying state."""
        return self.pgx_state._step_count

    @property
    def env_id(self) -> core.EnvId:
        """Delegate env_id to underlying state."""
        return self.pgx_state.env_id


class SinglePlayerWrapper(core.Env):
    """Wrapper that converts multi-player environment to single-player interface.

    Args:
        env: The multi-player PGX environment to wrap
        num_players: Total number of players in the game
        active_player_id: Which player ID the agent controls (default 0)
        opponent_policy_fns: Optional sequence of policy functions for each opponent (obs -> action).
                           Length must equal num_players. For the active player position,
                           pass None or any value (it will be ignored).
                           If None, uses random policy for all opponents. Must be JAX-traceable.
        opponent_network_fn: Optional network function (params, obs, rng) -> action for opponents.
                           If provided, will be used instead of opponent_policy_fns.
        max_steps_per_turn: Safety limit to prevent infinite loops (default 1000)
    """

    def __init__(
        self,
        env: core.Env,
        num_players: int,
        active_player_id: int = 0,
        opponent_policy_fns: Optional[Sequence[Optional[Callable[[Array], Array]]]] = None,
        opponent_network_fn: Optional[Callable[[Any, Array, PRNGKey], Array]] = None,
        max_steps_per_turn: int = 1000,
    ):
        super().__init__()
        self.env = env
        self.active_player_id = jnp.int32(active_player_id)
        self.max_steps_per_turn = max_steps_per_turn
        self._num_players = num_players
        self.opponent_network_fn = opponent_network_fn

        # Build array of policy functions, one per player
        self._opponent_policy_fns = []
        for player_id in range(num_players):
            if player_id == active_player_id:
                # Active player slot (won't be used, but need consistent array size)
                self._opponent_policy_fns.append(self._uniform_random_policy_fn)
            elif opponent_policy_fns is not None and opponent_policy_fns[player_id] is not None:
                # Custom policy for this opponent
                self._opponent_policy_fns.append(
                    self._make_custom_policy_fn(opponent_policy_fns[player_id])
                )
            else:
                # Default random policy
                self._opponent_policy_fns.append(self._uniform_random_policy_fn)

    def init_with_opponent_params(self, key: PRNGKey, initial_opponent_params: Any) -> SPWState:
        """Initialize environment with opponent parameters (for JIT-compatible training).

        Args:
            key: Random key for initialization
            initial_opponent_params: Network parameters for opponent policies

        Returns:
            SPWState with initialized environment and opponent parameters
        """
        key1, key2, key3 = jax.random.split(key, 3)

        # Initialize underlying environment
        pgx_state = self.env.init(key1)

        # Create SPWState
        spw_state = SPWState(
            pgx_state=pgx_state,
            opponent_params=initial_opponent_params,
            opponent_policy_rngs=key2
        )

        # Advance to active player's turn using while_loop
        def cond_fn(carry):
            spw_state, _, step_count = carry
            # Continue while not active player's turn AND not terminated AND under step limit
            return (spw_state.current_player != self.active_player_id) & ~spw_state.terminated & (step_count < self.max_steps_per_turn)

        def body_fn(carry):
            spw_state, rng_key, step_count = carry
            # Split key for opponent policy
            rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

            # Get opponent action using network function
            obs = self.env.observe(spw_state.pgx_state, spw_state.current_player)
            action = self.opponent_network_fn(spw_state.opponent_params, obs, subkey1)

            # Step environment
            new_pgx_state = self.env.step(spw_state.pgx_state, action)

            # Update SPWState
            spw_state = spw_state.replace(
                pgx_state=new_pgx_state,
                opponent_policy_rngs=subkey2
            )

            return spw_state, rng_key, step_count + 1

        # Run while loop to advance to active player
        spw_state, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (spw_state, key3, jnp.int32(0))
        )

        return spw_state

    def _init(self, key: PRNGKey) -> core.State:
        """Initialize environment and advance to active player's turn."""
        # Initialize underlying environment
        key1, key2 = jax.random.split(key)
        state = self.env.init(key1)

        # Advance to active player's turn using while_loop
        def cond_fn(carry):
            state, _, step_count = carry
            # Continue while not active player's turn AND not terminated AND under step limit
            return (state.current_player != self.active_player_id) & ~state.terminated & (step_count < self.max_steps_per_turn)

        def body_fn(carry):
            state, rng_key, step_count = carry
            # Split key for opponent policy
            rng_key, subkey = jax.random.split(rng_key)

            # Get opponent action using policy for current player
            # Use lax.switch for JAX-traceable indexing
            action = jax.lax.switch(
                state.current_player,
                self._opponent_policy_fns,
                state,
                subkey
            )

            # Step environment
            state = self.env.step(state, action)

            return state, rng_key, step_count + 1

        # Run while loop to advance to active player
        state, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (state, key2, jnp.int32(0))
        )

        return state

    def _uniform_random_policy_fn(self, state: core.State, key: PRNGKey) -> Array:
        """Random policy - sample uniformly from legal actions."""
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        return jax.random.categorical(key, logits)

    def _make_custom_policy_fn(self, policy_fn: Callable[[Array], Array]) -> Callable[[core.State, PRNGKey], Array]:
        """Create a custom policy function wrapper."""
        def custom_policy(state: core.State, key: PRNGKey) -> Array:
            obs = self.env.observe(state, state.current_player)
            return policy_fn(obs)
        return custom_policy

    def step_with_opponent_params(self, state: SPWState, action: Array) -> SPWState:
        """Execute active player's action with opponent parameters (for JIT-compatible training).

        Args:
            state: SPWState containing environment state and opponent parameters
            action: Action chosen by the active player

        Returns:
            Updated SPWState after active player action and opponent responses
        """
        # Step 1: Execute active player's action
        key1, key2 = jax.random.split(state.opponent_policy_rngs)
        new_pgx_state = self.env.step(state.pgx_state, action)

        # Track cumulative reward for active player
        cumulative_reward = new_pgx_state.rewards[self.active_player_id]

        # Update SPWState with new pgx_state and updated RNG
        spw_state = state.replace(
            pgx_state=new_pgx_state,
            opponent_policy_rngs=key2
        )

        # Step 2: Auto-play opponent turns until back to active player or terminated
        def cond_fn(carry):
            spw_state, _, _, step_count = carry
            # Continue while not active player's turn AND not terminated AND under step limit
            return (spw_state.current_player != self.active_player_id) & ~spw_state.terminated & (step_count < self.max_steps_per_turn)

        def body_fn(carry):
            spw_state, cumulative_reward, rng_key, step_count = carry
            # Split key for opponent policy
            rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)

            # Get opponent action using network function
            obs = self.env.observe(spw_state.pgx_state, spw_state.current_player)
            action = self.opponent_network_fn(spw_state.opponent_params, obs, subkey1)

            # Step environment
            new_pgx_state = self.env.step(spw_state.pgx_state, action)

            # Accumulate reward for active player
            cumulative_reward += new_pgx_state.rewards[self.active_player_id]

            # Update SPWState
            spw_state = spw_state.replace(
                pgx_state=new_pgx_state,
                opponent_policy_rngs=subkey2
            )

            return spw_state, cumulative_reward, rng_key, step_count + 1

        # Run while loop to auto-play opponents
        spw_state, cumulative_reward, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (spw_state, cumulative_reward, key2, jnp.int32(0))
        )

        # Update the active player's reward with the cumulative value
        updated_rewards = spw_state.pgx_state.rewards.at[self.active_player_id].set(cumulative_reward)
        updated_pgx_state = spw_state.pgx_state.replace(rewards=updated_rewards)  # type: ignore
        spw_state = spw_state.replace(pgx_state=updated_pgx_state)

        return spw_state

    def _step(self, state: core.State, action: Array, key: Optional[Array]) -> core.State:
        """Execute active player's action and auto-play opponent turns."""
        # Generate key if not provided
        if key is None:
            # Use a deterministic key based on step count for reproducibility
            key = jax.random.PRNGKey(state._step_count)

        # Step 1: Execute active player's action
        key1, key2 = jax.random.split(key)
        state = self.env.step(state, action)

        # Track cumulative reward for active player
        cumulative_reward = state.rewards[self.active_player_id]

        # Step 2: Auto-play opponent turns until back to active player or terminated
        def cond_fn(carry):
            state, _, _, step_count = carry
            # Continue while not active player's turn AND not terminated AND under step limit
            return (state.current_player != self.active_player_id) & ~state.terminated & (step_count < self.max_steps_per_turn)

        def body_fn(carry):
            state, cumulative_reward, rng_key, step_count = carry
            # Split key for opponent policy
            rng_key, subkey = jax.random.split(rng_key)

            # Get opponent action using policy for current player
            # Use lax.switch for JAX-traceable indexing
            action = jax.lax.switch(
                state.current_player,
                self._opponent_policy_fns,
                state,
                subkey
            )

            # Step environment
            state = self.env.step(state, action)

            # Accumulate reward for active player
            cumulative_reward += state.rewards[self.active_player_id]

            return state, cumulative_reward, rng_key, step_count + 1

        # Run while loop to auto-play opponents
        state, cumulative_reward, _, _ = jax.lax.while_loop(
            cond_fn,
            body_fn,
            (state, cumulative_reward, key2, jnp.int32(0))
        )

        # Update the active player's reward with the cumulative value
        # Keep other players' rewards from the final state to maintain zero-sum property
        updated_rewards = state.rewards.at[self.active_player_id].set(cumulative_reward)
        state = state.replace(rewards=updated_rewards)  # type: ignore

        return state

    def _observe(self, state: core.State, player_id: Array) -> Array:
        """Always observe from active player's perspective."""
        # Delegate to underlying environment, always observing active player
        return self.env.observe(state, self.active_player_id)

    @property
    def id(self) -> core.EnvId:
        return self.env.id

    @property
    def version(self) -> str:
        return self.env.version

    @property
    def num_players(self) -> int:
        # Single-player from the wrapper's perspective
        return 1
