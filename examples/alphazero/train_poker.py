#!/usr/bin/env python3
# Enhanced Universal Poker Training Script
# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import datetime
import os
import pickle
import time
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
from pgx.experimental import auto_reset
from pydantic import BaseModel
from pgx import universal_poker

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        HAS_TENSORBOARD = True
    except ImportError:
        HAS_TENSORBOARD = False
        print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class PokerConfig(BaseModel):
    """Configuration for Universal Poker training."""

    # Environment
    env_id: str = "universal_poker"
    seed: int = 0

    # Training
    max_num_iters: int = 1000
    learning_rate: float = 0.001
    weight_decay: float = 1e-6

    # Network architecture
    num_channels: int = 256
    num_layers: int = 4

    # Self-play
    selfplay_batch_size: int = 4096  # Larger default for vectorized training
    num_simulations: int = 32
    max_num_steps: int = 256

    # Training batch
    training_batch_size: int = 4096

    # Evaluation
    eval_interval: int = 10
    eval_batch_size: int = 32  # Reduced from 64 to avoid livelock issues

    # Checkpointing
    checkpoint_interval: int = 50

    # Logging
    tensorboard_dir: str = ""
    log_interval: int = 1

    # Performance
    vectorized_batch_size: int = 8192  # Size for vectorized operations

    class Config:
        extra = "forbid"


class PokerNet(hk.Module):
    """Poker-optimized feedforward network."""

    def __init__(self, num_actions: int, num_channels: int, num_layers: int, name="poker_net"):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_layers = num_layers

    def __call__(self, x, is_training=True):
        x = x.astype(jnp.float32)
        x = hk.Flatten()(x)

        # Shared trunk
        hidden = x
        for i in range(self.num_layers):
            hidden = hk.Linear(self.num_channels, name=f"hidden_{i}")(hidden)
            hidden = jax.nn.relu(hidden)

            # Skip connection every 2 layers
            if i > 0 and i % 2 == 1 and hidden.shape == x.shape:
                hidden = hidden + x

        # Policy head
        policy_hidden = hk.Linear(self.num_channels // 2, name="policy_hidden")(hidden)
        policy_hidden = jax.nn.relu(policy_hidden)
        logits = hk.Linear(self.num_actions, name="policy_out")(policy_hidden)

        # Value head
        value_hidden = hk.Linear(self.num_channels // 2, name="value_hidden")(hidden)
        value_hidden = jax.nn.relu(value_hidden)
        value_hidden = hk.Linear(self.num_channels // 4, name="value_hidden2")(value_hidden)
        value_hidden = jax.nn.relu(value_hidden)
        value = hk.Linear(1, name="value_out")(value_hidden)
        value = jnp.tanh(value).flatten()

        return logits, value


class TensorBoardLogger:
    """TensorBoard logging wrapper."""

    def __init__(self, log_dir: str):
        self.writer = None
        if HAS_TENSORBOARD and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, scalars: dict, step: int):
        if self.writer:
            for tag, value in scalars.items():
                self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int):
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def close(self):
        if self.writer:
            self.writer.close()


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray


def create_random_baseline(num_actions: int):
    """Create a random baseline policy that selects uniformly from legal actions."""

    def random_policy(observation):
        batch_size = observation.shape[0]
        # Create uniform logits for all actions - the masking will happen in the evaluation
        # Zero logits means uniform distribution after masking
        return jnp.zeros((batch_size, num_actions)), jnp.zeros(batch_size)

    return random_policy


def create_checkpoint_baseline(checkpoint_path: str, forward_fn, env):
    """Create a baseline policy from a loaded checkpoint."""
    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        model_params, model_state = checkpoint["model"]
        print(f"Loaded checkpoint baseline from: {checkpoint_path}")
        print(f"Checkpoint was from iteration: {checkpoint['iteration']}")

        def checkpoint_policy(observation):
            (logits, value), _ = forward_fn.apply(model_params, model_state, observation, is_training=False)
            return logits, value

        return checkpoint_policy

    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        print("Falling back to random baseline")
        return create_random_baseline(env.num_actions)


def create_poker_trainer(config: PokerConfig, env: pgx.core.Env, baseline_policy=None):
    """Create poker training functions."""

    # Use provided baseline or create default random baseline
    if baseline_policy is None:
        baseline = create_random_baseline(env.num_actions)
    else:
        baseline = baseline_policy

    # Network definition
    def forward_fn(x, is_training=True):
        net = PokerNet(num_actions=env.num_actions, num_channels=config.num_channels, num_layers=config.num_layers)
        return net(x, is_training=is_training)

    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
    optimizer = optax.nadamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)

    def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
        del rng_key
        model_params, model_state = model

        current_player = state.current_player
        state = jax.vmap(env.step)(state, action)

        (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_training=True)

        # Mask invalid actions
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            ),
            state,
        )

    # Vectorized selfplay using vmap instead of pmap for better CPU utilization
    def vectorized_selfplay_fn(batch_size: int):
        @jax.jit
        def _selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
            model_params, model_state = model

            def single_step(state, key):
                key1, key2 = jax.random.split(key)
                observation = state.observation

                (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_training=True)
                root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

                policy_output = mctx.gumbel_muzero_policy(
                    params=model,
                    rng_key=key1,
                    root=root,
                    recurrent_fn=recurrent_fn,
                    num_simulations=config.num_simulations,
                    invalid_actions=~state.legal_action_mask,
                    qtransform=mctx.qtransform_completed_by_mix_value,
                    gumbel_scale=1.0,
                )

                actor = state.current_player
                keys = jax.random.split(key2, batch_size)

                # Only step games that haven't terminated
                not_terminated = ~state.terminated

                # Step all games (JAX vmap requires uniform operations)
                # but we'll only keep results for non-terminated games
                new_state = jax.vmap(env.step)(state, policy_output.action)

                # For terminated games, do NOTHING - keep the old state unchanged
                # For non-terminated games, use the new state from env.step
                def selective_step_only(old_val, new_val):
                    # Handle different array dimensions by expanding not_terminated appropriately
                    if old_val.ndim == 1:
                        mask = not_terminated
                    else:
                        # Create mask with same number of dimensions as the values
                        mask_shape = (batch_size,) + (1,) * (old_val.ndim - 1)
                        mask = not_terminated.reshape(mask_shape)
                    return jnp.where(mask, new_val, old_val)  # Keep old state for terminated games

                state = jax.tree_util.tree_map(selective_step_only, state, new_state)

                discount = -1.0 * jnp.ones_like(value)
                discount = jnp.where(state.terminated, 0.0, discount)

                return state, SelfplayOutput(
                    obs=observation,
                    action_weights=policy_output.action_weights,
                    reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
                    terminated=state.terminated,
                    discount=discount,
                )

            # Initialize states
            rng_key, sub_key = jax.random.split(rng_key)
            keys = jax.random.split(sub_key, batch_size)
            state = jax.vmap(env.init)(keys)

            # Run self-play
            key_seq = jax.random.split(rng_key, config.max_num_steps)
            _, data = jax.lax.scan(single_step, state, key_seq)

            return data

        return _selfplay

    def compute_loss_input_fn(batch_size: int):
        @jax.jit
        def _compute_loss_input(data: SelfplayOutput) -> Sample:
            # Compute value targets
            value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

            def body_fn(carry, i):
                ix = config.max_num_steps - i - 1
                v = data.reward[ix] + data.discount[ix] * carry
                return v, v

            _, value_tgt = jax.lax.scan(
                body_fn,
                jnp.zeros(batch_size),
                jnp.arange(config.max_num_steps),
            )
            value_tgt = value_tgt[::-1, :]

            return Sample(
                obs=data.obs,
                policy_tgt=data.action_weights,
                value_tgt=value_tgt,
                mask=value_mask,
            )

        return _compute_loss_input

    def loss_fn(model_params, model_state, samples: Sample):
        (logits, value), model_state = forward.apply(model_params, model_state, samples.obs, is_training=True)

        policy_loss = optax.softmax_cross_entropy(logits, samples.policy_tgt)
        policy_loss = jnp.mean(policy_loss)

        value_loss = optax.l2_loss(value, samples.value_tgt)
        value_loss = jnp.mean(value_loss * samples.mask)

        total_loss = policy_loss + value_loss

        return total_loss, (model_state, policy_loss, value_loss)

    @jax.jit
    def train_step(model, opt_state, samples: Sample):
        model_params, model_state = model
        grads, (model_state, policy_loss, value_loss) = jax.grad(loss_fn, has_aux=True)(
            model_params, model_state, samples
        )
        updates, opt_state = optimizer.update(grads, opt_state, model_params)
        model_params = optax.apply_updates(model_params, updates)
        model = (model_params, model_state)
        return model, opt_state, policy_loss, value_loss

    def vectorized_evaluate_fn(batch_size: int):
        """Create a JIT-compiled evaluation function for a specific batch size."""

        @jax.jit
        def _evaluate(model, rng_key: jnp.ndarray):
            """Vectorized evaluation against baseline."""
            model_params, model_state = model
            my_player = 0
            max_eval_steps = 2000  # Safety limit to prevent infinite loops

            key, subkey = jax.random.split(rng_key)
            keys = jax.random.split(subkey, batch_size)
            state = jax.vmap(env.init)(keys)

            def body_fn(val):
                key, state, R, step_count = val
                (my_logits, _), _ = forward.apply(model_params, model_state, state.observation, is_training=False)
                opp_logits, _ = baseline(state.observation)
                is_my_turn = (state.current_player == my_player).reshape((-1, 1))
                logits = jnp.where(is_my_turn, my_logits, opp_logits)
                key, subkey = jax.random.split(key)
                action = jax.random.categorical(subkey, logits, axis=-1)

                # Only step games that haven't terminated
                not_terminated = ~state.terminated
                new_state = jax.vmap(env.step)(state, action)

                # Preserve terminated states, update only non-terminated ones
                def selective_update(old_val, new_val):
                    # Handle different array dimensions by expanding not_terminated appropriately
                    if old_val.ndim == 1:
                        mask = not_terminated
                    else:
                        # Create mask with same number of dimensions as the values
                        mask_shape = (batch_size,) + (1,) * (old_val.ndim - 1)
                        mask = not_terminated.reshape(mask_shape)
                    return jnp.where(mask, new_val, old_val)

                state = jax.tree_util.tree_map(selective_update, state, new_state)
                R = R + state.rewards[jnp.arange(batch_size), my_player]
                return (key, state, R, step_count + 1)

            _, final_state, R, final_steps = jax.lax.while_loop(
                lambda x: ~(x[1].terminated.all()) & (x[3] < max_eval_steps),
                body_fn,
                (key, state, jnp.zeros(batch_size), 0),
            )
            return R

        return _evaluate

    return (forward, optimizer, vectorized_selfplay_fn, compute_loss_input_fn, train_step, vectorized_evaluate_fn)


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Universal Poker Training")
    parser.add_argument(
        "--tensorboard_dir", type=str, default="runs/poker_training", help="Directory for TensorBoard logs"
    )
    parser.add_argument("--max_num_iters", type=int, default=1000, help="Maximum number of training iterations")
    parser.add_argument("--batch_size", type=int, default=4096, help="Self-play batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Optimizer weight decay")
    parser.add_argument("--num_simulations", type=int, default=32, help="MCTS simulations per move")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluation interval")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Checkpoint saving interval")
    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to use as baseline opponent. If not specified, uses random policy",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create config from args
    config = PokerConfig(
        max_num_iters=args.max_num_iters,
        selfplay_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_simulations=args.num_simulations,
        seed=args.seed,
        eval_interval=args.eval_interval,
        eval_batch_size=args.eval_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        tensorboard_dir=args.tensorboard_dir,
    )

    print("Enhanced Universal Poker Training")
    print("=" * 50)
    print(f"Config: {config}")
    print(f"JAX devices: {jax.local_devices()}")

    # Initialize logging
    logger = TensorBoardLogger(config.tensorboard_dir)

    # Create the environment
    config_str = """GAMEDEF
numplayers 8
numrounds 4
firstplayer 3 1 1 1
blind 1 2 0 0 0 0 0 0
stack 100 100 100 100 100 100 100 100
END GAMEDEF"""
    env = universal_poker.UniversalPoker(num_players=8, config_str=config_str)

    # Create baseline policy
    baseline_policy = None
    if args.baseline_checkpoint:
        print(f"Loading baseline from checkpoint: {args.baseline_checkpoint}")

        # We need to create the forward function first to load the checkpoint
        def forward_fn(x, is_training=True):
            net = PokerNet(num_actions=env.num_actions, num_channels=config.num_channels, num_layers=config.num_layers)
            return net(x, is_training=is_training)

        forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
        baseline_policy = create_checkpoint_baseline(args.baseline_checkpoint, forward, env)
    else:
        print("Using random baseline policy")

    # Create trainer functions
    (forward, optimizer, vectorized_selfplay_fn, compute_loss_input_fn, train_step, vectorized_evaluate_fn) = (
        create_poker_trainer(config, env, baseline_policy)
    )

    # Create JIT-compiled functions for the specific batch sizes
    vectorized_selfplay = vectorized_selfplay_fn(config.selfplay_batch_size)
    compute_loss_input = compute_loss_input_fn(config.selfplay_batch_size)
    vectorized_evaluate = vectorized_evaluate_fn(config.eval_batch_size)

    # Initialize model
    dummy_state = env.init(jax.random.PRNGKey(0))
    dummy_input = env.observe(dummy_state, dummy_state.current_player)
    model = forward.init(jax.random.PRNGKey(config.seed), dummy_input[None, :])
    opt_state = optimizer.init(model[0])

    print(f"Model initialized. Params shape: {jax.tree_util.tree_map(lambda x: x.shape, model[0])}")

    # Create checkpoint directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join("checkpoints", f"poker_{timestamp}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    rng_key = jax.random.PRNGKey(config.seed)
    start_time = time.time()

    for iteration in range(config.max_num_iters):
        iter_start = time.time()

        # Evaluation
        if iteration % config.eval_interval == 0:
            print(f"\n=== Iteration {iteration} - Evaluation ===")
            print(f"Starting evaluation with batch_size={config.eval_batch_size}")
            eval_start = time.time()
            rng_key, eval_key = jax.random.split(rng_key)
            R = vectorized_evaluate(model, eval_key)
            eval_time = time.time() - eval_start
            print(f"Evaluation completed in {eval_time:.2f}s")

            eval_metrics = {
                "eval/avg_reward": R.mean().item(),
                "eval/win_rate": ((R > 0).sum() / R.size).item(),
                "eval/draw_rate": ((R == 0).sum() / R.size).item(),
                "eval/lose_rate": ((R < 0).sum() / R.size).item(),
            }

            logger.log_scalars(eval_metrics, iteration)

            print(f"Evaluation results:")
            for key, value in eval_metrics.items():
                print(f"  {key}: {value:.4f}")

        # Self-play
        print(f"Starting self-play with batch_size={config.selfplay_batch_size}")
        selfplay_start = time.time()
        rng_key, selfplay_key = jax.random.split(rng_key)
        data = vectorized_selfplay(model, selfplay_key)
        selfplay_time = time.time() - selfplay_start
        print(f"Self-play completed in {selfplay_time:.2f}s")
        samples = compute_loss_input(data)

        # Training
        rng_key, shuffle_key = jax.random.split(rng_key)

        # Reshape and shuffle data
        samples_flat = jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), samples)

        num_samples = samples_flat.obs.shape[0]
        perm = jax.random.permutation(shuffle_key, num_samples)
        samples_shuffled = jax.tree_util.tree_map(lambda x: x[perm], samples_flat)

        # Train on minibatches
        num_updates = num_samples // config.training_batch_size
        policy_losses, value_losses = [], []

        for i in range(num_updates):
            start_idx = i * config.training_batch_size
            end_idx = (i + 1) * config.training_batch_size
            minibatch = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], samples_shuffled)

            model, opt_state, policy_loss, value_loss = train_step(model, opt_state, minibatch)
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
        avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else 0.0

        # Logging
        iter_time = time.time() - iter_start
        total_time = time.time() - start_time

        train_metrics = {
            "train/policy_loss": avg_policy_loss,
            "train/value_loss": avg_value_loss,
            "train/total_loss": avg_policy_loss + avg_value_loss,
            "timing/iteration_time": iter_time,
            "timing/total_time": total_time,
            "timing/iterations_per_hour": iteration / (total_time / 3600) if total_time > 0 else 0,
        }

        logger.log_scalars(train_metrics, iteration)

        if iteration % config.log_interval == 0:
            print(
                f"Iter {iteration:4d}: Policy Loss={avg_policy_loss:.6f}, "
                f"Value Loss={avg_value_loss:.6f}, Time={iter_time:.2f}s"
            )

        # Checkpointing
        if iteration % config.checkpoint_interval == 0:
            checkpoint = {
                "config": config,
                "model": jax.device_get(model),
                "opt_state": jax.device_get(opt_state),
                "iteration": iteration,
                "rng_key": rng_key,
            }

            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{iteration:06d}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(checkpoint, f)

            print(f"Saved checkpoint: {ckpt_path}")

    logger.close()
    print(f"\nTraining completed! Total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
