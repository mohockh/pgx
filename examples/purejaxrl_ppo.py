import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from pydantic import BaseModel
from pgx import universal_poker
from pgx.single_player_wrapper import SinglePlayerWrapper
from pgx.poker_eval.cardset import cardset_to_cards
import collections
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
#import gymnax
#from wrappers import LogWrapper, FlattenObservationWrapper


@jax.jit
def cardset_to_onehot(cardset: jnp.ndarray) -> jnp.ndarray:
    """Convert uint32[2] cardset to one-hot (52,) encoding.

    Args:
        cardset: uint32[2] array representing a set of cards

    Returns:
        One-hot encoded array of shape (52,) with 1.0 at positions of cards in the set
    """
    # Initialize one-hot vector for all 52 cards
    onehot = jnp.zeros(52, dtype=jnp.float32)

    # Check each possible card (0-51) and set bit if present in cardset
    for card_id in range(52):
        # ACPC bit position: (suit << 4) + rank
        bit_pos = (card_id // 13) * 16 + (card_id % 13)

        # Check if this bit is set in the cardset
        # For bit_pos < 32, check cardset[0], else check cardset[1]
        word_idx = bit_pos // 32
        bit_in_word = bit_pos % 32

        # Extract the bit
        is_set = (cardset[word_idx] >> bit_in_word) & jnp.uint32(1)

        # Set the corresponding position in one-hot
        onehot = onehot.at[card_id].set(is_set.astype(jnp.float32))

    return onehot


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


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # Handle both batched and unbatched inputs
        # x can be shape (obs_size,) or (batch, obs_size)
        is_batched = x.ndim > 1

        if is_batched:
            # Process batch of observations
            # x shape: (batch, obs_size)
            # Parse observation: [hole_cardset[2], board_cardset[2], other_features...]
            hole_cardsets = x[:, :2].astype(jnp.uint32)  # (batch, 2)
            visible_board_cardsets = x[:, 2:4].astype(jnp.uint32)  # (batch, 2)
            other_features = x[:, 4:]  # (batch, remaining)

            # Convert cardsets to one-hot encodings (vectorized over batch)
            hole_onehots = jax.vmap(cardset_to_onehot)(hole_cardsets)  # (batch, 52)
            board_onehots = jax.vmap(cardset_to_onehot)(visible_board_cardsets)  # (batch, 52)
        else:
            # Process single observation
            # x shape: (obs_size,)
            hole_cardset = x[:2].astype(jnp.uint32)  # (2,)
            visible_board_cardset = x[2:4].astype(jnp.uint32)  # (2,)
            other_features = x[4:]  # (remaining,)

            # Convert cardsets to one-hot encodings
            hole_onehots = cardset_to_onehot(hole_cardset)  # (52,)
            board_onehots = cardset_to_onehot(visible_board_cardset)  # (52,)

        # Embed the card representations
        hole_embed = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(hole_onehots)
        hole_embed = activation(hole_embed)

        board_embed = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(board_onehots)
        board_embed = activation(board_embed)

        # Concatenate embeddings with other features
        x = jnp.concatenate([hole_embed, board_embed, other_features], axis=-1)

        actor_mean = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean) + actor_mean
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            32, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic) + critic
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    episode_return: jnp.ndarray  # Accumulated return when episode ends
    episode_length: jnp.ndarray  # Length when episode ends


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Create the base environment (we'll wrap it with policies later in train())
    config_str = """GAMEDEF
numplayers 8
numrounds 4
firstplayer 3 1 1 1
blind 1 2 0 0 0 0 0 0
stack 100 100 100 100 100 100 100 100
END GAMEDEF"""
    base_env = universal_poker.UniversalPoker(num_players=8, config_str=config_str)
    num_players = 8
    active_player_id = 0

    # Module-level writer that will be set in train()
    tensorboard_writer = None


    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # Define opponent network policy function
    def opponent_network_policy(params, obs, rng):
        """Opponent policy using the same network architecture as the agent."""
        network = ActorCritic(base_env.num_actions, activation=config["ACTIVATION"])
        pi, _ = network.apply(params, obs)
        return pi.sample(seed=rng)

    def train_chunk(agent_train_state, opponent_params, rng):
        """JIT-compiled training chunk with fixed opponent parameters.

        Args:
            agent_train_state: Current training state for the agent
            opponent_params: Fixed opponent network parameters for this chunk
            rng: Random key

        Returns:
            Updated agent training state and metrics

        Note:
            This function captures 'writer' from the enclosing scope for TensorBoard logging
        """
        # Create environment with opponent network function
        env = SinglePlayerWrapper(
            base_env,
            num_players=num_players,
            active_player_id=active_player_id,
            opponent_network_fn=opponent_network_policy,
            max_steps_per_turn=1000,
            num_opponent_policies=config.get("OPPONENT_HISTORY_SIZE", 1)
        )

        network = ActorCritic(base_env.num_actions, activation=config["ACTIVATION"])

        # INIT ENV with opponent parameters
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        # Use init_with_opponent_params to create SPWState
        env_state = jax.vmap(env.init_with_opponent_params, in_axes=(0, None))(
            reset_rng, opponent_params
        )
        obsv = env_state.observation

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, episode_returns, episode_lengths = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV - use step_with_opponent_params for SPWState
                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
                env_state = jax.vmap(env.step_with_opponent_params)(env_state, action)

                # Extract data from PGX state
                obsv = env_state.observation
                # SinglePlayerWrapper puts the active player's cumulative reward at index active_player_id
                # env_state.rewards has shape (num_envs, num_players), we want (num_envs,)
                reward = env_state.rewards[:, env.active_player_id]
                done = env_state.terminated
                info = {}  # PGX doesn't use info dict

                # Update episode trackers
                episode_returns += reward
                episode_lengths += 1

                # Capture stats when episode ends, then reset trackers
                episode_return_info = jnp.where(done, episode_returns, 0.0)
                episode_length_info = jnp.where(done, episode_lengths, 0)

                episode_returns = jnp.where(done, 0.0, episode_returns)
                episode_lengths = jnp.where(done, 0, episode_lengths)

                # Auto-reset terminated environments - use init_with_opponent_params for SPWState
                env_state = jax.vmap(lambda s, rng: jax.lax.cond(
                    s.terminated,
                    lambda: env.init_with_opponent_params(rng, opponent_params),
                    lambda: s
                ))(env_state, reset_rng)
                # Update observation after potential reset
                obsv = env_state.observation

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info,
                    episode_return_info, episode_length_info
                )
                runner_state = (train_state, env_state, obsv, rng, episode_returns, episode_lengths)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng, episode_returns, episode_lengths = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, grads)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_and_grads = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                total_loss, grads = loss_and_grads
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, (total_loss, grads)
            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Debugging mode - adapted for PGX
            if config.get("DEBUG"):
                def callback(traj_batch, update_idx, loss_and_grad_info):
                    # Unpack loss and gradient info
                    (total_loss_info, grad_info) = loss_and_grad_info
                    total_loss, (value_loss, policy_loss, entropy) = total_loss_info

                    # Extract only non-zero episode returns (actual completions)
                    completed_returns = traj_batch.episode_return[traj_batch.episode_return != 0]
                    completed_lengths = traj_batch.episode_length[traj_batch.episode_length != 0]

                    if len(completed_returns) > 0:
                        mean_return = completed_returns.mean()
                        max_return = completed_returns.max()
                        min_return = completed_returns.min()
                        mean_length = completed_lengths.mean()
                        num_episodes = len(completed_returns)
                    else:
                        mean_return = max_return = min_return = mean_length = 0.0
                        num_episodes = 0

                    # Count positive/negative/zero rewards across ALL timesteps and envs
                    all_rewards = traj_batch.reward.flatten()
                    num_positive = (all_rewards > 0).sum()
                    num_negative = (all_rewards < 0).sum()
                    num_zero = (all_rewards == 0).sum()

                    # Calculate global step
                    global_step = update_idx * config["NUM_ENVS"] * config["NUM_STEPS"]

                    # Compute gradient statistics
                    # grad_info has shape (num_epochs, num_minibatches, ...) - take last epoch, last minibatch
                    last_grads = jax.tree_util.tree_map(lambda x: x[-1, -1], grad_info)

                    # Compute global gradient norm
                    grad_norm = jnp.sqrt(sum(
                        jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(last_grads)
                    ))

                    print(f"Update {int(update_idx):4d} | Step {int(global_step):7d} | "
                          f"Return: {mean_return:7.2f} (min={min_return:7.2f}, max={max_return:7.2f}) | "
                          f"Episodes: {int(num_episodes):3d} | Len: {mean_length:5.1f} | "
                          f"Rewards: +{int(num_positive)} -{int(num_negative)} 0:{int(num_zero)} | "
                          f"VLoss: {value_loss[-1].mean():.4f} | "
                          f"PLoss: {policy_loss[-1].mean():.4f} | "
                          f"Entropy: {entropy[-1].mean():.4f} | "
                          f"GradNorm: {grad_norm:.4f}")

                    # TensorBoard logging
                    if tensorboard_writer is not None:
                        # Scalar metrics
                        tensorboard_writer.add_scalar('metrics/mean_return', float(mean_return), int(global_step))
                        tensorboard_writer.add_scalar('metrics/mean_episode_length', float(mean_length), int(global_step))
                        tensorboard_writer.add_scalar('metrics/num_episodes', int(num_episodes), int(global_step))
                        tensorboard_writer.add_scalar('metrics/positive_rewards', int(num_positive), int(global_step))
                        tensorboard_writer.add_scalar('metrics/negative_rewards', int(num_negative), int(global_step))
                        tensorboard_writer.add_scalar('metrics/zero_rewards', int(num_zero), int(global_step))

                        # Histogram distributions
                        if len(completed_returns) > 0:
                            tensorboard_writer.add_histogram('distributions/episode_returns',
                                               np.array(completed_returns), int(global_step))

                        # Loss histograms (across all epochs and minibatches)
                        tensorboard_writer.add_histogram('distributions/value_loss',
                                           np.array(value_loss.flatten()), int(global_step))
                        tensorboard_writer.add_histogram('distributions/policy_loss',
                                           np.array(policy_loss.flatten()), int(global_step))
                        tensorboard_writer.add_histogram('distributions/entropy',
                                           np.array(entropy.flatten()), int(global_step))

                        # Gradient metrics
                        tensorboard_writer.add_scalar('gradients/grad_norm', float(grad_norm), int(global_step))

                        # Per-parameter gradient norms and histograms
                        param_grad_norms = []
                        for path, grad in jax.tree_util.tree_leaves_with_path(last_grads):
                            # Convert path to string (e.g., 'params/Dense_0/kernel')
                            path_str = '/'.join(str(k.key) for k in path if hasattr(k, 'key'))
                            param_norm = jnp.sqrt(jnp.sum(jnp.square(grad)))
                            param_grad_norms.append(float(param_norm))

                            # Log histogram for this parameter's gradients
                            if grad.size > 1:  # Only log histograms for non-scalar parameters
                                tensorboard_writer.add_histogram(f'gradients/{path_str}',
                                                   np.array(grad.flatten()), int(global_step))

                        # Histogram of all parameter gradient norms
                        if param_grad_norms:
                            tensorboard_writer.add_histogram('gradients/param_grad_norms',
                                               np.array(param_grad_norms), int(global_step))

                jax.debug.callback(callback, traj_batch, update_idx, loss_info)

            runner_state = (train_state, env_state, last_obs, rng, episode_returns, episode_lengths)
            return runner_state, metric

        # Run the update step for one chunk
        rng, _rng = jax.random.split(rng)
        # Initialize episode trackers
        episode_returns = jnp.zeros(config["NUM_ENVS"])
        episode_lengths = jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32)
        runner_state = (agent_train_state, env_state, obsv, _rng, episode_returns, episode_lengths)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["CHUNK_SIZE"])
        )
        train_state = runner_state[0]
        return train_state, metric

    def train(rng):
        """Outer training loop (not JIT-compiled) that manages opponent policies."""
        nonlocal tensorboard_writer

        # INIT NETWORK
        network = ActorCritic(
            base_env.num_actions, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        dummy_state = base_env.init(jax.random.PRNGKey(0))
        init_x = base_env.observe(dummy_state, active_player_id)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Initialize TensorBoard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logdir = f"{config.get('TENSORBOARD_DIR', 'runs/ppo_poker')}_{timestamp}"
        tensorboard_writer = SummaryWriter(logdir)
        print(f"TensorBoard logs will be saved to: {logdir}")

        # Initialize opponent history
        opponent_history = collections.deque(maxlen=config.get("OPPONENT_HISTORY_SIZE", 100))
        opponent_history.append(network_params)

        # JIT compile the training chunk
        train_chunk_compiled = jax.jit(train_chunk)

        # Outer loop: iterate through chunks, updating opponent params between chunks
        num_chunks = int(config["NUM_UPDATES"] // config["CHUNK_SIZE"])
        all_metrics = []

        for chunk_idx in range(num_chunks):
            # Store current agent params in history
            opponent_history.append(train_state.params)

            # Create policy pool by sampling from history with replacement
            pool_size = config.get("OPPONENT_HISTORY_SIZE", 1)
            sampled_policies = random.choices(list(opponent_history), k=pool_size)

            # Stack policies into a single PyTree (policy pool)
            policy_pool = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *sampled_policies)

            # Run training chunk with policy pool
            rng, _rng = jax.random.split(rng)
            train_state, metrics = train_chunk_compiled(train_state, policy_pool, _rng)
            all_metrics.append(metrics)

            # Optional: logging outside JIT
            if config.get("DEBUG") and chunk_idx % config.get("LOG_INTERVAL", 10) == 0:
                print(f"Chunk {chunk_idx + 1}/{num_chunks} completed")

        # Close TensorBoard writer
        if tensorboard_writer is not None:
            tensorboard_writer.close()
            print(f"TensorBoard logging complete. View with: tensorboard --logdir={logdir}")

        return {"train_state": train_state, "metrics": all_metrics}

    return train


if __name__ == "__main__":
    config = {
        "LR": 2.5e-3,
        "NUM_ENVS": 128,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 128,
        "GAMMA": 1.0,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.02,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": True,
        "CHUNK_SIZE": 10,  # Number of updates per JIT-compiled chunk
        "OPPONENT_LAG_UPDATES": 1,  # Number of chunks between opponent policy updates
        "OPPONENT_HISTORY_SIZE": 15,  # Max number of past policies to keep
        "LOG_INTERVAL": 1,  # Log every N chunks
        "TENSORBOARD_DIR": "/tmp/ppo_poker",  # TensorBoard log directory
    }
    rng = jax.random.PRNGKey(30)
    train_fn = make_train(config)
    out = train_fn(rng)
