"""Single-player wrapper for multi-player PGX environments.

Converts any multi-player PGX environment into a single-player interface by automatically
handling opponent turns. This enables direct use with single-agent RL algorithms like PPO.

Only depends on pgx.core API for maximum reusability.
"""
from typing import Callable, Optional

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.types import Array, PRNGKey


class SinglePlayerWrapper(core.Env):
    """Wrapper that converts multi-player environment to single-player interface.

    Args:
        env: The multi-player PGX environment to wrap
        active_player_id: Which player ID the agent controls (default 0)
        opponent_policy_fn: Policy function for opponents (obs -> action).
                          If None, uses random policy. Must be JAX-traceable.
        max_steps_per_turn: Safety limit to prevent infinite loops (default 1000)
    """

    def __init__(
        self,
        env: core.Env,
        active_player_id: int = 0,
        opponent_policy_fn: Optional[Callable[[Array], Array]] = None,
        max_steps_per_turn: int = 1000,
    ):
        super().__init__()
        self.env = env
        self.active_player_id = jnp.int32(active_player_id)
        self.max_steps_per_turn = max_steps_per_turn

        # Set opponent policy (will implement random policy later)
        self._opponent_policy_fn = opponent_policy_fn

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

            # Get opponent action (use random policy for now)
            action = self._get_opponent_action(state, subkey)

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

    def _get_opponent_action(self, state: core.State, key: PRNGKey) -> Array:
        """Get action from opponent policy."""
        if self._opponent_policy_fn is not None:
            # Use provided policy
            obs = self.env.observe(state, state.current_player)
            return self._opponent_policy_fn(obs)
        else:
            # Use random policy - sample uniformly from legal actions
            logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
            return jax.random.categorical(key, logits)

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

            # Get opponent action
            action = self._get_opponent_action(state, subkey)

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

        # Replace the rewards in state with cumulative reward for active player
        # For single-player interface, we only expose the active player's reward
        final_rewards = jnp.zeros(state.rewards.shape, dtype=state.rewards.dtype)
        final_rewards = final_rewards.at[self.active_player_id].set(cumulative_reward)
        state = state.replace(rewards=final_rewards)  # type: ignore

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
