# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "tqdm",
#     "dreamer4",
#     "adam-atan2-pytorch>=0.3.6",
#     "memmap-replay-buffer>=0.1.10",
#     "torch-einops-utils>=0.1.6",
#     "fire",
#     "gymnasium[mujoco]",
#     "tensorboard",
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

# contributed by @CarsonBurke in https://github.com/lucidrains/dreamer4/pull/29

from __future__ import annotations

import json
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from copy import deepcopy
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, Mapping
from math import sqrt

import fire
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils._pytree import tree_map

from einops import rearrange
from adam_atan2_pytorch import MuonAdamAtan2
from torch_einops_utils import safe_cat, temp_eval, lens_to_mask

from memmap_replay_buffer.replay_buffer import ReplayBuffer, ReplayDatasetTimeWindow

from dreamer4.dreamer4 import (
    Actions,
    DynamicsWorldModel,
    Experience,
    ShortcutTrainMode,
    StateTokenizer,
    divisible_by,
    exists,
    tree_map_tensor
)

from .compile_runtime import (
    CallTiming,
    action_pair_or_empty,
    build_compile_runtime,
    compile_mode_uses_cuda_graphs,
    configure_sac,
    cudagraph_mark_step_begin,
    normalize_compile_mode,
    print_compile_timing_report,
    record_timed_block,
    synchronize_if_cuda,
)
from .profiling import PROFILE_TOTAL_STEPS, TrainingProfiler


_runtime_cleanups = ContextVar("runtime_cleanups", default = None)


def close_runtime_resources(fn):
    """Ensure resources registered by a training run close on every exit path."""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        cleanups = []
        token = _runtime_cleanups.set(cleanups)
        try:
            return fn(*args, **kwargs)
        finally:
            for cleanup in reversed(cleanups):
                cleanup()
            _runtime_cleanups.reset(token)

    return wrapped


def register_runtime_cleanup(cleanup):
    cleanups = _runtime_cleanups.get()
    if cleanups is not None:
        cleanups.append(cleanup)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def repeat_batch_to_size(batch, size: int):
    tensor = next((value for value in batch.values() if torch.is_tensor(value)), None)

    if not exists(tensor) or tensor.shape[0] == size:
        return batch

    batch_size = tensor.shape[0]
    assert 0 < batch_size <= size

    indices = torch.arange(size, device = tensor.device) % batch_size
    return tree_map(lambda value: value.index_select(0, indices) if torch.is_tensor(value) else value, batch)


def resampling_dataloader_iterator(dataloader: DataLoader):
    """Iterate forever while recreating the loader iterator each epoch.

    ``itertools.cycle`` caches yielded batches, which would freeze the sampled
    replay windows after the first pass.  The old replay path needs each new
    optimizer step to call the dataset again so its random window start is
    resampled.
    """

    while True:
        yield from dataloader

# tokenizer


def _plain_silu_mlp(
    dim_in: int,
    dim_out: int,
    *,
    hidden_dim: int,
    depth: int,
):
    """Build the legacy state tokenizer's plain SiLU MLP.

    The current state tokenizer is intentionally transformer-shaped even when
    temporal attention is disabled.  The old Gym trainer used a small,
    feed-forward encoder/decoder instead.  Keep this implementation local to
    the trainer so that the world model's state-tokenizer API remains unchanged.
    """

    layers = []
    dim = dim_in
    for _ in range(max(1, depth)):
        layers.extend((nn.Linear(dim, hidden_dim), nn.SiLU()))
        dim = hidden_dim
    layers.append(nn.Linear(dim, dim_out))
    return nn.Sequential(*layers)


class OldStateTokenizer(nn.Module):
    """Legacy plain-SiLU MLP state tokenizer.

    It mirrors :class:`StateTokenizer`'s tensor contract while deliberately
    omitting attention, RMSNorm, and time-cache handling from the modern
    tokenizer.  All leading dimensions are treated as independent states.
    """

    def __init__(
        self,
        dim_state: int,
        num_latent_tokens: int = 4,
        dim_latent: int = 32,
        dim: int = 256,
        depth: int = 2,
    ):
        super().__init__()
        self.dim_state = dim_state
        self.num_latent_tokens = num_latent_tokens
        self.dim_latent = dim_latent
        latent_dim = num_latent_tokens * dim_latent

        self.encoder = _plain_silu_mlp(
            dim_state,
            latent_dim,
            hidden_dim = dim,
            depth = depth,
        )
        self.decoder = _plain_silu_mlp(
            latent_dim,
            dim_state,
            hidden_dim = dim,
            depth = depth,
        )

    def encode(self, state: Tensor, **_kwargs):
        shape = state.shape
        latents = self.encoder(state.reshape(-1, shape[-1]))
        latents = latents.tanh()
        return latents.reshape(*shape[:-1], self.num_latent_tokens, self.dim_latent)

    @torch.no_grad()
    def tokenize(self, state: Tensor, **kwargs):
        return self.encode(state, **kwargs)

    def decode(self, latents: Tensor, **_kwargs):
        shape = latents.shape
        latents = latents.reshape(-1, self.num_latent_tokens * self.dim_latent)
        recon = self.decoder(latents)
        return recon.reshape(*shape[:-2], self.dim_state)

    def forward(
        self,
        state: Tensor,
        *,
        return_latents = False,
        return_recon = False,
        **kwargs,
    ):
        latents = self.encode(state, **kwargs)
        if return_latents and not return_recon:
            return latents

        recon = self.decode(latents, **kwargs)
        loss = F.mse_loss(recon, state)
        if return_recon:
            return loss, recon, latents
        return loss

class ObservationTokenizer(nn.Module):
    def __init__(
        self,
        obs_dim: int = 17,
        num_latent_tokens: int = 4,
        dim_latent: int = 32,
        hidden_dim: int = 256,
        depth: int = 2,
        eps: float = 1e-5,
        old_state_tokenizer: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_latent_tokens = num_latent_tokens
        self.dim_latent = dim_latent
        self.eps = eps

        tokenizer_cls = OldStateTokenizer if old_state_tokenizer else StateTokenizer
        tokenizer_kwargs = dict(
            dim_state = obs_dim,
            num_latent_tokens = num_latent_tokens,
            dim_latent = dim_latent,
            dim = hidden_dim,
            depth = depth,
        )
        if not old_state_tokenizer:
            tokenizer_kwargs.update(attn_every = 0, attn_dim_head = 64)
        self.tokenizer = tokenizer_cls(**tokenizer_kwargs)
        self.old_state_tokenizer = old_state_tokenizer

        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    @property
    def device(self):
        return self.obs_mean.device

    @torch.no_grad()
    def set_normalization(self, observations: Tensor):
        observations = observations.to(self.device, dtype = torch.float32)
        self.obs_mean.copy_(observations.mean(dim = 0))
        self.obs_std.copy_(observations.std(dim = 0, correction = 0).clamp(min = self.eps))

    def normalize(self, observations: Tensor):
        return (observations - self.obs_mean) / self.obs_std

    def denormalize(self, observations: Tensor):
        return observations * self.obs_std + self.obs_mean

    def encode(self, observations: Tensor, **kwargs):
        observations = self.normalize(observations.float())
        return self.tokenizer.encode(observations, **kwargs)

    def decode(self, latents: Tensor, **kwargs):
        recon = self.tokenizer.decode(latents, **kwargs)
        return self.denormalize(recon)

    def forward(
        self,
        observations: Tensor,
        *,
        return_latents = False,
        return_recon = False,
        **kwargs
    ):
        observations = self.normalize(observations.float())

        if return_latents and not return_recon:
            return self.tokenizer.encode(observations, **kwargs)

        if return_recon:
            loss, recon, latents = self.tokenizer(observations, return_recon = True, **kwargs)
            return loss, self.denormalize(recon), latents

        return self.tokenizer(observations, **kwargs)


# env helpers

VectorizationMode = Literal["async", "sync"]


class RolloutTimingWrapper(gym.vector.VectorWrapper):
    """Accumulate vector-environment reset and step wall time per rollout."""

    def __init__(self, env):
        super().__init__(env)
        self.reset_seconds = 0.
        self.step_seconds = 0.

    def reset(self, **kwargs):
        start = perf_counter()
        try:
            return self.env.reset(**kwargs)
        finally:
            self.reset_seconds += perf_counter() - start

    def step(self, actions):
        start = perf_counter()
        try:
            return self.env.step(actions)
        finally:
            self.step_seconds += perf_counter() - start

    def pop_timings(self):
        timings = self.reset_seconds, self.step_seconds
        self.reset_seconds = 0.
        self.step_seconds = 0.
        return timings


def pop_rollout_env_timings(env):
    current = env

    while current is not None:
        if isinstance(current, RolloutTimingWrapper):
            return current.pop_timings()
        current = getattr(current, "env", None)

    return 0., 0.


def available_cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def make_env(
    env_name: str,
    seed: int | None,
    *,
    vectorized = True,
    num_envs = 8,
    episode_stats_buffer_length = 100,
    env_vectorization_mode: VectorizationMode = "async",
):
    if env_vectorization_mode not in ("async", "sync"):
        raise ValueError(f"env_vectorization_mode must be 'async' or 'sync', got {env_vectorization_mode!r}")

    if vectorized:
        vector_kwargs = {"context": "spawn"} if env_vectorization_mode == "async" else None
        env = gym.make_vec(
            env_name,
            num_envs = num_envs,
            vectorization_mode = env_vectorization_mode,
            vector_kwargs = vector_kwargs,
        )
        env = RolloutTimingWrapper(env)
        env = gym.wrappers.vector.RecordEpisodeStatistics(env, buffer_length = episode_stats_buffer_length)
    else:
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length = episode_stats_buffer_length)

    if exists(seed):
        env.action_space.seed(seed)

    return env


def reset_env(env, seed = None):
    out = env.reset(seed = seed) if exists(seed) else env.reset()
    return out[0] if isinstance(out, tuple) else out


def inspect_env_spaces(env_name: str, expected_obs_dim: int | None):
    env = gym.make(env_name)

    try:
        observation_space = env.observation_space
        action_space = env.action_space

        if not isinstance(observation_space, spaces.Box):
            raise ValueError(f"{env_name} must have a Box observation space, got {type(observation_space).__name__}")

        if len(observation_space.shape) != 1:
            raise ValueError(f"{env_name} must have a flat 1D observation space, got shape {observation_space.shape}")

        actual_obs_dim = int(observation_space.shape[0])

        if exists(expected_obs_dim) and actual_obs_dim != expected_obs_dim:
            raise ValueError(f"{env_name} produced obs dim {actual_obs_dim}, expected {expected_obs_dim}")

        if not isinstance(action_space, spaces.Box):
            raise ValueError(f"{env_name} must have a continuous Box action space, got {type(action_space).__name__}")

        if len(action_space.shape) != 1:
            raise ValueError(f"{env_name} must have a flat 1D action space, got shape {action_space.shape}")

        action_lows = np.asarray(action_space.low, dtype = np.float32)
        action_highs = np.asarray(action_space.high, dtype = np.float32)

        if not np.isfinite(action_lows).all() or not np.isfinite(action_highs).all():
            raise ValueError(
                f"{env_name} action bounds must be finite for beta policy targets; "
                f"got lows={action_lows}, highs={action_highs}"
            )

        if not (action_lows < action_highs).all():
            raise ValueError(
                f"{env_name} action lows must be strictly below highs; "
                f"got lows={action_lows}, highs={action_highs}"
            )

        low, high = float(action_lows[0]), float(action_highs[0])
        if not np.allclose(action_lows, low) or not np.allclose(action_highs, high):
            raise ValueError(
                f"{env_name} has per-dimension action bounds, but this trainer only supports a single shared "
                f"continuous_target_action_range; got lows={action_lows}, highs={action_highs}"
            )

        return actual_obs_dim, int(np.prod(action_space.shape)), (low, high)
    finally:
        env.close()


def obs_array(obs):
    obs = np.asarray(obs, dtype = np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    return obs


def collect_random_observations(
    env_name: str,
    *,
    num_steps = 4096,
    num_envs = 8,
    seed = 42,
    env_vectorization_mode: VectorizationMode = "async",
):
    env = make_env(
        env_name,
        seed,
        vectorized = True,
        num_envs = num_envs,
        env_vectorization_mode = env_vectorization_mode,
    )
    obs = reset_env(env, seed = seed)

    observations = [obs_array(obs)]

    for _ in tqdm(range(num_steps), desc = "random obs"):
        action = env.action_space.sample()
        step_out = env.step(action)
        obs = step_out[0]
        observations.append(obs_array(obs))

    env.close()

    return torch.from_numpy(np.concatenate(observations, axis = 0))


def obs_to_latents_fn(tokenizer: ObservationTokenizer):
    @torch.no_grad()
    def inner(_world_model, obs, time_cache):
        state = obs["state"]
        if not torch.is_tensor(state):
            state = torch.tensor(state, device = tokenizer.device, dtype = torch.float32)
        else:
            state = state.to(tokenizer.device, dtype = torch.float32)

        if state.ndim == 1:
            state = rearrange(state, "d -> 1 d")

        latents = tokenizer(state, return_latents = True)
        latents = rearrange(latents, "b n d -> b 1 n d")
        return latents, time_cache

    return inner


def experience_to_cpu(exp: Experience):
    if not isinstance(exp, Experience) or exp.payload.device.type != "cuda":
        return exp.cpu()

    device = exp.payload.device

    def copy_to_pinned_cpu(tensor: Tensor):
        if tensor.device.type != "cuda":
            return tensor.cpu()

        cpu_tensor = torch.empty_like(tensor, device = "cpu", pin_memory = True)
        cpu_tensor.copy_(tensor, non_blocking = True)
        return cpu_tensor

    values = tree_map_tensor(copy_to_pinned_cpu, vars(exp))
    torch.cuda.current_stream(device).synchronize()
    return Experience(**values)


# training helpers

def log_scalars(writer: SummaryWriter | None, scalars: dict[str, float], step: int):
    if not exists(writer):
        return

    for key, value in scalars.items():
        if value is None:
            continue
        writer.add_scalar(key, float(value), step)

def get_rollout_episode_stats(exp: Experience):
    """Treat each returned environment trajectory as one fixed-horizon episode."""
    if not exists(exp.episode_return):
        return [], []

    returns = exp.episode_return
    lengths = exp.lens

    if not exists(lengths):
        lengths = torch.full_like(returns, exp.rewards.shape[1])

    # interact_with_env appends one bootstrap state to externally truncated
    # trajectories. It is not an environment transition and should not count
    # toward the synthetic episode length.
    if exists(exp.is_truncated) and exists(exp.terminals):
        has_bootstrap_padding = exp.is_truncated & ~exp.terminals
        lengths = lengths - has_bootstrap_padding.long()

    return returns.tolist(), lengths.tolist()

def unique_parameters(params):
    return list(dict.fromkeys(params))

def make_optimizer(
    params,
    *,
    lr: float,
    weight_decay: float,
    use_muon: bool,
    muon_params = (),
):
    params = list(dict.fromkeys(params))
    fused = len(params) > 0 and all(param.is_cuda for param in params)

    if not use_muon:
        return AdamW(params, lr = lr, weight_decay = weight_decay, fused = fused)

    optimizer_param_set = set(params)
    muon_params = [
        param for param in dict.fromkeys(muon_params)
        if param in optimizer_param_set
    ]

    if len(muon_params) == 0:
        return AdamW(params, lr = lr, weight_decay = weight_decay, fused = fused)

    return MuonAdamAtan2(
        muon_params = muon_params,
        params = params,
        lr = lr,
        weight_decay = weight_decay,
    )


def module_parameters(module: nn.Module | None):
    return [] if not exists(module) else list(module.parameters())


def optimizer_parameters(optimizer: Optimizer):
    return [param for group in optimizer.param_groups for param in group["params"]]


def agent_optimizer_parameters(
    world_model: DynamicsWorldModel,
    full_transformer_finetuning: bool,
):
    if full_transformer_finetuning:
        return list(world_model.parameters())

    return world_model.agent_parameters()


def disjoint_optimizer_param_groups(
    optimizer: AdamW,
    named_params: list[tuple[str, list[nn.Parameter]]],
    rest_name: str,
):
    optimizer_param_set = set(optimizer_parameters(optimizer))
    seen = set()
    groups = []

    for name, params in named_params:
        group_params = [
            param for param in params
            if param in optimizer_param_set and param not in seen
        ]

        if len(group_params) == 0:
            continue

        seen.update(group_params)
        groups.append((name, group_params))

    rest = [param for param in optimizer_parameters(optimizer) if param not in seen]

    if len(rest) > 0:
        groups.append((rest_name, rest))

    return groups


def clip_grad_norm_by_group(
    groups: list[tuple[str, list[nn.Parameter]]],
    max_grad_norm: float,
):
    metrics = {}
    total_norm_sq = 0.

    for name, params in groups:
        params_with_grad = [param for param in params if exists(param.grad)]

        if len(params_with_grad) == 0:
            continue

        norm = float(clip_grad_norm_(params_with_grad, max_grad_norm))
        metrics[f"{name}_grad_norm"] = norm
        total_norm_sq += norm ** 2

    return sqrt(total_norm_sq), metrics


def world_model_clip_groups(world_model: DynamicsWorldModel, optimizer: Optimizer):
    reward_head = getattr(world_model, "to_reward_pred", None)
    terminal_head = getattr(world_model, "to_state_terminal_pred", None)

    # The restored legacy heads use the current model's explicit module names.
    # Keep the historical lookup above as the false/default path so existing
    # clipping groups and checkpoint behavior remain unchanged.
    if getattr(world_model, "old_reward_head", False):
        reward_head = getattr(world_model, "to_agent_embed_reward_pred", reward_head)
    if getattr(world_model, "old_terminal_head", False):
        terminal_head = getattr(world_model, "to_latent_state_terminal_pred", terminal_head)

    return disjoint_optimizer_param_groups(
        optimizer,
        [
            ("reward_head", module_parameters(reward_head)),
            ("action_head", module_parameters(getattr(world_model, "action_embedder", None))),
            ("terminal_head", module_parameters(terminal_head)),
            ("state_head", module_parameters(getattr(world_model, "to_state_pred", None))),
            ("agent_state_head", module_parameters(getattr(world_model, "to_agent_state_pred", None))),
            ("latent_ar_head", module_parameters(getattr(world_model, "latent_ar", None))),
        ],
        "trunk",
    )


def agent_clip_groups(world_model: DynamicsWorldModel, optimizer: Optimizer):
    return disjoint_optimizer_param_groups(
        optimizer,
        [
            ("policy_head", module_parameters(getattr(world_model, "policy_head", None))),
            ("action_unembed", list(world_model.action_embedder.unembed_parameters())),
            ("value_head", module_parameters(getattr(world_model, "value_head", None))),
            ("actor_critic_value_head", module_parameters(getattr(world_model, "actor_critic_value_head", None))),
            ("critic_state_embedder", module_parameters(getattr(world_model, "critic_state_embedder", None))),
        ],
        "agent_rest",
    )

def trim_prompt_from_dream(dream: Experience, prompt_length: int, horizon: int):
    if prompt_length <= 0:
        return dream

    prompted_tensors = (
        dream.latents,
        dream.video,
        dream.proprio,
        dream.critic_state,
        dream.rewards,
        dream.actions.discrete if exists(dream.actions) else None,
        dream.actions.continuous if exists(dream.actions) else None,
    )
    generated_tensors = (
        dream.agent_embed,
        dream.policy_input,
        dream.values,
        dream.log_probs.discrete if exists(dream.log_probs) else None,
        dream.log_probs.continuous if exists(dream.log_probs) else None,
        *(dream.old_action_unembeds or ()),
    )

    available_lengths = [
        t.shape[1] - prompt_length
        for t in prompted_tensors
        if exists(t)
    ]
    available_lengths.extend([
        t.shape[1]
        for t in generated_tensors
        if exists(t)
    ])

    actual_horizon = min(horizon, *available_lengths)

    def trim_prompted(t: Tensor | None):
        return t[:, prompt_length:prompt_length + actual_horizon] if exists(t) else None

    def trim_generated(t: Tensor | None):
        return t[:, :actual_horizon] if exists(t) else None

    actions = Actions(
        trim_prompted(dream.actions.discrete),
        trim_prompted(dream.actions.continuous),
    ) if exists(dream.actions) else None

    log_probs = Actions(
        trim_generated(dream.log_probs.discrete),
        trim_generated(dream.log_probs.continuous),
    ) if exists(dream.log_probs) else None

    old_action_unembeds = tuple(trim_generated(t) for t in dream.old_action_unembeds) if exists(dream.old_action_unembeds) else None

    lens = None
    if exists(dream.lens):
        lens = (dream.lens - prompt_length).clamp(min = 0, max = actual_horizon)

    rewards = trim_prompted(dream.rewards)
    episode_return = None
    if exists(rewards):
        if exists(lens):
            mask = lens_to_mask(lens, actual_horizon)
            episode_return = (rewards * mask.float()).sum(dim = 1)
        else:
            episode_return = rewards.sum(dim = 1)

    return Experience(
        latents = trim_prompted(dream.latents),
        video = trim_prompted(dream.video),
        proprio = trim_prompted(dream.proprio),
        critic_state = trim_prompted(dream.critic_state),
        agent_embed = trim_generated(dream.agent_embed),
        policy_input = trim_generated(dream.policy_input),
        rewards = rewards,
        terminals = dream.terminals,
        actions = actions,
        log_probs = log_probs,
        old_action_unembeds = old_action_unembeds,
        values = trim_generated(dream.values),
        step_size = dream.step_size,
        lens = lens,
        is_truncated = dream.is_truncated,
        agent_index = dream.agent_index,
        is_from_world_model = dream.is_from_world_model,
        episode_return = episode_return,
    )


class FrozenPolicyPrior(nn.Module):
    def __init__(self, world_model: DynamicsWorldModel):
        super().__init__()
        self.policy_all_tokens = world_model.policy_all_tokens
        self.policy_head = deepcopy(world_model.policy_head)
        self.action_embedder = deepcopy(world_model.action_embedder)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def refresh_from(self, world_model: DynamicsWorldModel):
        self.policy_all_tokens = world_model.policy_all_tokens
        self.policy_head.load_state_dict(world_model.policy_head.state_dict())
        self.action_embedder.load_state_dict(world_model.action_embedder.state_dict())
        self.to(world_model.device)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def action_unembeds(self, policy_inputs: Tensor):
        policy_embed = self.policy_head(policy_inputs.detach())
        return self.action_embedder.unembed(policy_embed, pred_head_index = 0)


@torch.no_grad()
def agent_approx_kl_metrics(world_model: DynamicsWorldModel, dream: Experience):
    assert exists(dream.agent_embed)
    assert exists(dream.actions)
    assert exists(dream.log_probs)

    policy_inputs = dream.policy_input if world_model.policy_all_tokens and exists(dream.policy_input) else dream.agent_embed
    policy_inputs = policy_inputs.detach()
    policy_embed = world_model.policy_head(policy_inputs)
    policy_time = policy_embed.shape[1]

    def align_time(t):
        if not exists(t):
            return None

        return t[:, :policy_time]

    discrete_actions = align_time(dream.actions.discrete)
    continuous_actions = align_time(dream.actions.continuous)
    old_log_probs = Actions(*(align_time(t) for t in dream.log_probs))

    log_probs = world_model.action_embedder.log_probs(
        policy_embed,
        pred_head_index = 0,
        discrete_targets = discrete_actions,
        continuous_targets = continuous_actions,
    )

    old_log_probs = safe_cat(old_log_probs, dim = -1)
    log_probs = safe_cat(log_probs, dim = -1)

    log_ratio = log_probs.sum(dim = -1) - old_log_probs.sum(dim = -1)
    approx_kl = log_ratio.exp() - 1. - log_ratio

    lens = dream.lens
    if exists(lens):
        is_truncated = dream.is_truncated if exists(dream.is_truncated) else torch.ones_like(lens, dtype = torch.bool)
        learnable_lens = (lens - is_truncated.long()).clamp(min = 0, max = policy_time)
        mask = lens_to_mask(learnable_lens, policy_time)
        approx_kl = approx_kl[mask]
    else:
        approx_kl = approx_kl.reshape(-1)

    if approx_kl.numel() == 0:
        approx_kl = log_ratio.new_zeros((1,))

    return {
        "approx_kl_mean": approx_kl.mean(),
        "approx_kl_min": approx_kl.min(),
        "approx_kl_max": approx_kl.max(),
    }


@torch.no_grad()
def agent_prior_kl_metrics(world_model: DynamicsWorldModel, dream: Experience, policy_prior: FrozenPolicyPrior):
    assert exists(dream.agent_embed)

    policy_inputs = dream.policy_input if world_model.policy_all_tokens and exists(dream.policy_input) else dream.agent_embed
    policy_inputs = policy_inputs.detach()
    policy_embed = world_model.policy_head(policy_inputs)
    current_unembeds = world_model.action_embedder.unembed(policy_embed, pred_head_index = 0)
    prior_unembeds = policy_prior.action_unembeds(policy_inputs)

    if world_model.pmpo_reverse_kl:
        current_unembeds, prior_unembeds = prior_unembeds, current_unembeds

    kl_values = None
    for kl_term in world_model.action_embedder.kl_div(current_unembeds, prior_unembeds):
        if not exists(kl_term):
            continue

        if kl_term.ndim == 3 and kl_term.shape[-1] == 1:
            kl_term = rearrange(kl_term, "b t 1 -> b t")

        kl_values = kl_term if not exists(kl_values) else kl_values + kl_term

    assert exists(kl_values)

    lens = dream.lens
    if exists(lens):
        policy_time = kl_values.shape[1]
        is_truncated = dream.is_truncated if exists(dream.is_truncated) else torch.ones_like(lens, dtype = torch.bool)
        learnable_lens = (lens - is_truncated.long()).clamp(min = 0, max = policy_time)
        mask = lens_to_mask(learnable_lens, policy_time)
        kl_values = kl_values[mask]
    else:
        kl_values = kl_values.reshape(-1)

    if kl_values.numel() == 0:
        kl_values = policy_inputs.new_zeros((1,))

    return {
        "prior_kl_mean": kl_values.mean(),
        "prior_kl_min": kl_values.min(),
        "prior_kl_max": kl_values.max(),
    }


def attach_policy_prior_unembeds(dream: Experience, policy_prior: FrozenPolicyPrior):
    assert exists(dream.agent_embed)
    policy_inputs = dream.policy_input if policy_prior.policy_all_tokens and exists(dream.policy_input) else dream.agent_embed
    dream.old_action_unembeds = policy_prior.action_unembeds(policy_inputs)
    return dream


def train_tokenizer(
    tokenizer: ObservationTokenizer,
    observations: Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    max_grad_norm: float,
    device: torch.device,
    writer: SummaryWriter | None,
    training_profiler: TrainingProfiler | None = None,
):
    tokenizer.to(device)
    observations = observations.to(device)
    tokenizer.set_normalization(observations)

    if steps <= 0:
        return None

    dataset = TensorDataset(observations)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = len(dataset) >= batch_size)
    optimizer = AdamW(tokenizer.parameters(), lr = learning_rate, fused = device.type == "cuda")
    iterator = cycle(dataloader) if len(dataloader) > 0 else None

    last_loss = None
    pbar = tqdm(range(steps), desc = "tokenizer")

    for step in pbar:
        profiler = training_profiler
        record = profiler.record if exists(profiler) else lambda *_, **__: nullcontext()

        with record("tokenizer/data/load_batch"):
            batch = next(iterator)[0]

        with record("tokenizer/model/forward", cuda = True):
            loss = tokenizer(batch)
        with record("tokenizer/model/backward", cuda = True):
            loss.backward()

        with record("tokenizer/optimization/grad_clip", cuda = True):
            clip_grad_norm_(tokenizer.parameters(), max_grad_norm)
        with record("tokenizer/optimization/optimizer_step", cuda = True):
            optimizer.step()
            optimizer.zero_grad()

        last_loss = loss.item()
        pbar.set_postfix(loss = f"{last_loss:.4f}")
        log_scalars(writer, {"tokenizer/recon_loss": last_loss}, step)

    if exists(training_profiler):
        training_profiler.flush_timings()

    return last_loss


@torch.no_grad()
def eval_tokenizer_on_experience(
    tokenizer: ObservationTokenizer,
    exp: Experience,
    *,
    max_samples: int,
):
    states = exp.critic_state

    if not exists(states):
        return None

    batch, time = states.shape[:2]
    device = states.device

    if exists(exp.lens):
        mask = lens_to_mask(exp.lens, time)
        states = states[mask]
    else:
        states = states.reshape(batch * time, states.shape[-1])

    if states.numel() == 0:
        return None

    if max_samples > 0 and states.shape[0] > max_samples:
        indices = torch.linspace(0, states.shape[0] - 1, max_samples, device = device).long()
        states = states[indices]

    sample_count = states.shape[0]

    with temp_eval(tokenizer):
        loss = tokenizer(states)

    return loss.item(), sample_count


def train_world_model(
    world_model: DynamicsWorldModel,
    optimizer: Optimizer,
    replay: ReplayBuffer,
    *,
    world_model_loss_fn: Callable,
    world_model_loss_timing: CallTiming | None,
    world_model_step_timing: CallTiming | None,
    steps: int,
    batch_size: int,
    sequence_length: int | None,
    static_batch_shape: bool,
    max_grad_norm: float,
    global_step: int,
    writer: SummaryWriter | None,
    training_profiler: TrainingProfiler | None,
    shortcut_train_mode: ShortcutTrainMode,
    desc: str,
    old_replay_sampling: bool = False,
    update_loss_ema: bool = True,
):
    if len(replay) == 0 or steps <= 0:
        return global_step, {}

    last_metrics = {}
    grad_clip_groups = world_model_clip_groups(world_model, optimizer)

    dataset = ReplayDatasetTimeWindow(
        replay,
        window_length = sequence_length,
        include_metadata = True
    )

    if old_replay_sampling:
        # Legacy world-model replay consumed a shuffled pass over windows and
        # kept a short final batch.  In particular, do not pad partial WM
        # batches back to ``batch_size`` below; this is part of the old
        # trainer's observable behavior.  Prompt sampling (below) is the path
        # that intentionally uses replacement.
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            drop_last = False,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            drop_last = len(dataset) >= batch_size
        )

    dataloader_iter = (
        resampling_dataloader_iterator(dataloader)
        if old_replay_sampling and len(dataloader) > 0
        else cycle(dataloader) if len(dataloader) > 0
        else None
    )

    pbar = tqdm(range(steps), desc = desc, dynamic_ncols = True)

    for step in pbar:
        profiler = training_profiler
        record = profiler.record if exists(profiler) else lambda *_, **__: nullcontext()
        model_step = profiler.step if exists(profiler) else lambda *_, **__: nullcontext()

        if exists(world_model_step_timing):
            synchronize_if_cuda(world_model.device)
            step_start = perf_counter()

        with model_step("world_model/step", cuda = True):
            with record("world_model/data/load_batch"):
                batch = next(dataloader_iter)

                if static_batch_shape and not old_replay_sampling:
                    batch = repeat_batch_to_size(batch, batch_size)

            with record("world_model/data/prepare_experience", cuda = True):
                exp = Experience.from_buffer_dict(batch)
                exp.lens = batch.get('_lens')
                exp = exp.to(world_model.device)

                # Mask episode flags when the sampled window does not reach its end.
                reaches_episode_end = batch.get('_reaches_episode_end')
                if exists(reaches_episode_end):
                    reaches_episode_end = reaches_episode_end.to(world_model.device)
                    if exists(exp.terminals): exp.terminals = exp.terminals & reaches_episode_end
                    if exists(exp.is_truncated): exp.is_truncated = exp.is_truncated & reaches_episode_end

            def forward_backward():
                with record("world_model/model/forward", cuda = True):
                    loss, losses = world_model_loss_fn(
                        exp.latents,
                        exp.rewards,
                        exp.terminals,
                        exp.actions.continuous if exists(exp.actions) else None,
                        exp.lens,
                        shortcut_train_mode,
                        update_loss_ema,
                    )

                with record("world_model/model/backward", cuda = True):
                    loss.backward()
                return loss, losses

            loss, losses = record_timed_block(world_model_loss_timing, world_model.device, forward_backward)

            with record("world_model/optimization/grad_clip", cuda = True):
                norm, grad_metrics = clip_grad_norm_by_group(grad_clip_groups, max_grad_norm)
            with record("world_model/optimization/optimizer_step", cuda = True):
                optimizer.step()
                optimizer.zero_grad()

            with record("world_model/metrics", cuda = True):
                last_metrics = {
                    "world_model/loss": loss.item(),
                    "world_model/flow_loss": losses.flow.item(),
                    "world_model/shortcut_loss": losses.shortcut.item(),
                    "world_model/reward_loss": (losses.agent_embed_reward.mean() + losses.latent_state_reward.mean()).item(),
                    "world_model/reward_mae": losses.reward_mae.item(),
                    "world_model/terminal_loss": (losses.agent_embed_terminal.mean() + losses.latent_state_terminal.mean()).item(),
                    "world_model/agent_state_pred_loss": losses.agent_state_pred.item(),
                    "world_model/grad_norm": float(norm),
                    "extra/flow_loss_raw": losses.diagnostics.raw_flow.item(),
                    "extra/shortcut_loss_raw": losses.diagnostics.raw_shortcut.item(),
                    "extra/dynamics_loss_raw": losses.diagnostics.raw_dynamics.item(),
                    "extra/dynamics_loss_normalized": losses.diagnostics.normalized_dynamics.item(),
                    "extra/reward_loss_raw": losses.diagnostics.raw_reward.item(),
                    "extra/terminal_loss_raw": losses.diagnostics.raw_terminal.item(),
                }
                last_metrics.update({
                    f"world_model/{key}": value
                    for key, value in grad_metrics.items()
                })

                log_scalars(writer, last_metrics, global_step)
                pbar.set_postfix(loss = f"{last_metrics['world_model/loss']:.3f}")
                global_step += 1

            if exists(world_model_step_timing):
                synchronize_if_cuda(world_model.device)
                world_model_step_timing.record(perf_counter() - step_start)

            del exp, loss, losses, grad_metrics

    if exists(training_profiler):
        training_profiler.flush_timings()

    return global_step, last_metrics


def train_agent_in_imagination(
    world_model: DynamicsWorldModel,
    optimizer: Optimizer,
    policy_prior: FrozenPolicyPrior | None,
    replay: ReplayBuffer,
    *,
    generate_rollout_fn: Callable,
    learn_from_dream_fn: Callable,
    learn_from_dream_timing: CallTiming | None,
    imagination_step_timing: CallTiming | None,
    steps: int,
    batch_size: int,
    horizon: int,
    prompt_length: int,
    prompt_probability: float,
    static_generate_shape: bool,
    generate_steps: int,
    max_grad_norm: float,
    objective: Literal["ppo", "pmpo", "spo"],
    use_delight_gating: bool,
    global_step: int,
    writer: SummaryWriter | None,
    training_profiler: TrainingProfiler | None,
    old_replay_sampling: bool = False,
    full_transformer_finetuning: bool = False,
    preservation_sequence_length: int | None = None,
    preservation_batch_size: int | None = None,
):
    if steps <= 0:
        return global_step, {}

    last_metrics = {}
    pbar = tqdm(range(steps), desc = "imagination", leave = False)
    grad_clip_groups = agent_clip_groups(world_model, optimizer)

    prompt_dataset = ReplayDatasetTimeWindow(
        replay_buffer = replay,
        window_length = prompt_length,
    )

    if static_generate_shape and prompt_length > 0 and prompt_probability != 1.:
        raise ValueError("static compiled generation requires prompt_probability=1.0 when prompt_length > 0")

    if old_replay_sampling:
        # Prompt windows in the legacy trainer were independent draws from
        # replay (with replacement), rather than a shuffled pass over every
        # available window.
        prompt_sampler = RandomSampler(
            prompt_dataset,
            replacement = True,
            num_samples = batch_size,
        )
        prompt_dataloader = DataLoader(
            prompt_dataset,
            batch_size = batch_size,
            sampler = prompt_sampler,
            drop_last = False,
        )
    else:
        prompt_dataloader = DataLoader(
            prompt_dataset,
            batch_size = batch_size,
            shuffle = True,
            drop_last = len(prompt_dataset) >= batch_size
        )

    prompt_iterator = (
        resampling_dataloader_iterator(prompt_dataloader)
        if old_replay_sampling and prompt_length > 0 and len(prompt_dataloader) > 0
        else cycle(prompt_dataloader) if prompt_length > 0 and len(prompt_dataloader) > 0
        else None
    )

    preservation_iterator = None
    if full_transformer_finetuning:
        preservation_sequence_length = (
            horizon
            if preservation_sequence_length is None
            else preservation_sequence_length
        )
        preservation_batch_size = (
            batch_size
            if preservation_batch_size is None
            else preservation_batch_size
        )

        if preservation_sequence_length <= 0:
            raise ValueError("preservation_sequence_length must be positive")

        if preservation_batch_size <= 0:
            raise ValueError("preservation_batch_size must be positive")

        preservation_dataset = ReplayDatasetTimeWindow(
            replay_buffer = replay,
            window_length = preservation_sequence_length,
            include_metadata = True,
        )
        preservation_dataloader = DataLoader(
            preservation_dataset,
            batch_size = preservation_batch_size,
            shuffle = True,
            drop_last = not old_replay_sampling and len(preservation_dataset) >= preservation_batch_size,
        )
        if len(preservation_dataloader) > 0:
            preservation_iterator = (
                resampling_dataloader_iterator(preservation_dataloader)
                if old_replay_sampling
                else cycle(preservation_dataloader)
            )

        if not exists(preservation_iterator):
            raise ValueError("full_transformer_finetuning requires at least one replay window")

    if static_generate_shape and prompt_length > 0 and not exists(prompt_iterator):
        raise ValueError("static compiled generation requires at least one prompt window")

    for _ in pbar:
        profiler = training_profiler
        record = profiler.record if exists(profiler) else lambda *_, **__: nullcontext()
        if exists(profiler):
            profiler.start_step()

        if exists(imagination_step_timing):
            synchronize_if_cuda(world_model.device)
            step_start = perf_counter()

        with torch.no_grad():
            use_prompt = prompt_length > 0 and (static_generate_shape or random.random() < prompt_probability)
            prompts = None

            if use_prompt and exists(prompt_iterator):
                with record("imagination/data/load_prompt", cuda = True):
                    prompt_batch = next(prompt_iterator)

                    prompt_batch = tree_map(lambda t: t.to(world_model.device) if torch.is_tensor(t) else t, prompt_batch)

                    if static_generate_shape:
                        prompt_batch = repeat_batch_to_size(prompt_batch, batch_size)

                    prompts = dict(
                        prompt_latents = prompt_batch.get('latents'),
                        prompt_proprio = prompt_batch.get('proprio'),
                        prompt_discrete_actions = prompt_batch.get('actions_discrete'),
                        prompt_continuous_actions = prompt_batch.get('actions_continuous'),
                        prompt_rewards = prompt_batch.get('rewards')[:, :-1] if exists(prompt_batch.get('rewards')) else None,
                    )

            generation_horizon = horizon + prompt_length if exists(prompts) else horizon
            actual_batch_size = prompt_batch.get('latents').shape[0] if exists(prompts) else batch_size

            has_prompt = exists(prompts)
            prompts = prompts or {}

            if exists(getattr(generate_rollout_fn, "timing", None)) and generate_rollout_fn.timing.cuda_graphs:
                cudagraph_mark_step_begin()

            with record("imagination/model/generate", cuda = True):
                dream = generate_rollout_fn(
                    generation_horizon,
                    actual_batch_size,
                    prompts.get("prompt_latents"),
                    prompts.get("prompt_proprio"),
                    prompts.get("prompt_discrete_actions"),
                    prompts.get("prompt_continuous_actions"),
                    prompts.get("prompt_rewards"),
                )

            if has_prompt:
                with record("imagination/data/trim_prompt", cuda = True):
                    dream = trim_prompt_from_dream(dream, prompt_length, horizon)

            if objective == "pmpo" and exists(policy_prior):
                with record("imagination/model/policy_prior", cuda = True):
                    dream = attach_policy_prior_unembeds(dream, policy_prior)

            preservation_exp = None
            if full_transformer_finetuning:
                with record("imagination/data/load_preservation_batch", cuda = True):
                    preservation_batch = next(preservation_iterator)

                    if static_generate_shape and not old_replay_sampling:
                        preservation_batch = repeat_batch_to_size(
                            preservation_batch,
                            preservation_batch_size,
                        )

                    preservation_exp = Experience.from_buffer_dict(preservation_batch)
                    preservation_exp.lens = preservation_batch.get("_lens")
                    preservation_exp = preservation_exp.to(world_model.device)

        dream_actions = action_pair_or_empty(dream.actions)
        dream_log_probs = action_pair_or_empty(dream.log_probs)
        dream_old_action_unembeds = action_pair_or_empty(dream.old_action_unembeds)
        preservation_actions = action_pair_or_empty(
            preservation_exp.actions if exists(preservation_exp) else None
        )

        def forward_backward():
            with record("imagination/model/learn_forward", cuda = True):
                policy_loss, value_loss, value_diagnostics = learn_from_dream_fn(
                    dream.latents,
                    dream.proprio,
                    dream.critic_state,
                    dream.agent_embed,
                    dream.policy_input,
                    dream.rewards,
                    dream.terminals,
                    dream_actions.discrete,
                    dream_actions.continuous,
                    dream_log_probs.discrete,
                    dream_log_probs.continuous,
                    dream_old_action_unembeds.discrete,
                    dream_old_action_unembeds.continuous,
                    dream.values,
                    dream.step_size,
                    dream.lens,
                    dream.is_truncated,
                    dream.is_from_world_model,
                    dream.episode_return,
                    preservation_exp.latents if exists(preservation_exp) else None,
                    preservation_exp.proprio if exists(preservation_exp) else None,
                    preservation_exp.rewards if exists(preservation_exp) else None,
                    preservation_actions.discrete,
                    preservation_actions.continuous,
                    preservation_exp.lens if exists(preservation_exp) else None,
                )

            loss = policy_loss + value_loss
            with record("imagination/model/backward", cuda = True):
                loss.backward()
            return policy_loss, value_loss, value_diagnostics, loss

        policy_loss, value_loss, value_diagnostics, loss = record_timed_block(
            learn_from_dream_timing,
            world_model.device,
            forward_backward,
        )

        with record("imagination/optimization/grad_clip", cuda = True):
            norm, grad_metrics = clip_grad_norm_by_group(grad_clip_groups, max_grad_norm)
        with record("imagination/optimization/optimizer_step", cuda = True):
            optimizer.step()
            optimizer.zero_grad()

        with record("imagination/metrics", cuda = True):
            agent_metrics = agent_approx_kl_metrics(world_model, dream)
            if objective == "pmpo" and exists(policy_prior):
                agent_metrics.update(agent_prior_kl_metrics(world_model, dream, policy_prior))

        dream_return = dream.episode_return.mean().item() if exists(dream.episode_return) else 0.
        dream_len = dream.lens.float().mean().item() if exists(dream.lens) else horizon
        action_std = dream.actions.continuous.std().item() if exists(dream.actions) and exists(dream.actions.continuous) else 0.
        policy_objective = policy_loss.detach()
        if "preservation/loss" in value_diagnostics:
            policy_objective = policy_objective - value_diagnostics["preservation/loss"]

        last_metrics = {
            "imagination/loss": loss.item(),
            "imagination/policy_loss": policy_objective.item(),
            "imagination/value_loss": value_loss.item(),
            "imagination/dream_return": dream_return,
            "imagination/dream_length": dream_len,
            "imagination/action_std": action_std,
            "imagination/prompt_length": prompt_length if has_prompt else 0,
            "imagination/grad_norm": float(norm),
            "imagination/approx_kl_mean": agent_metrics["approx_kl_mean"].item(),
            "imagination/approx_kl_min": agent_metrics["approx_kl_min"].item(),
            "imagination/approx_kl_max": agent_metrics["approx_kl_max"].item(),
            "imagination/prior_kl_mean": agent_metrics.get("prior_kl_mean", torch.tensor(0.)).item(),
            "imagination/prior_kl_min": agent_metrics.get("prior_kl_min", torch.tensor(0.)).item(),
            "imagination/prior_kl_max": agent_metrics.get("prior_kl_max", torch.tensor(0.)).item(),
        }
        last_metrics.update({
            key: value.item()
            for key, value in value_diagnostics.items()
        })
        last_metrics.update({
            f"imagination/{key}": value
            for key, value in grad_metrics.items()
        })

        log_scalars(writer, last_metrics, global_step)
        pbar.set_postfix(dream_return = f"{dream_return:.1f}", loss = f"{loss.item():.3f}")
        global_step += 1

        if exists(imagination_step_timing):
            synchronize_if_cuda(world_model.device)
            imagination_step_timing.record(perf_counter() - step_start)

        if exists(profiler):
            profiler.advance()

        # dream tensors can live in the compiled generate path's CUDA Graph pool
        # and must not outlive this iteration - the next replay overwrites them

        del dream
        del agent_metrics, grad_metrics
        del dream_actions, dream_log_probs, dream_old_action_unembeds
        del preservation_exp, preservation_actions
        del policy_loss, value_loss, value_diagnostics, loss

    if exists(training_profiler):
        training_profiler.flush_timings()

    return global_step, last_metrics


def save_checkpoint(
    path: Path,
    *,
    loop: int,
    tokenizer: ObservationTokenizer,
    world_model: DynamicsWorldModel,
    world_optimizer: Optimizer,
    agent_optimizer: Optimizer,
):
    path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(
        dict(
            loop = loop,
            tokenizer = tokenizer.state_dict(),
            world_model = world_model.state_dict(),
            world_optimizer = optimizer_state_dict_for_checkpoint(world_optimizer),
            agent_optimizer = optimizer_state_dict_for_checkpoint(agent_optimizer),
        ),
        str(path),
    )


def optimizer_state_dict_for_checkpoint(optimizer: Optimizer):
    state_dict = optimizer.state_dict()
    state_dict["param_groups"] = [
        {
            key: value
            for key, value in group.items()
            if not callable(value)
        }
        for group in state_dict["param_groups"]
    ]
    return state_dict


def load_optimizer_state_if_compatible(optimizer: Optimizer, state_dict, name: str):
    callable_group_values = [
        {
            key: value
            for key, value in group.items()
            if callable(value)
        }
        for group in optimizer.param_groups
    ]

    try:
        optimizer.load_state_dict(state_dict)
        return True
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"skipping {name} optimizer state: {exc}")
        return False
    finally:
        for group, callable_values in zip(optimizer.param_groups, callable_group_values):
            group.update(callable_values)


def resolve_log_dir(
    log_dir: str,
    *,
    run_name: str,
    checkpoint_path: str | None,
    unique_log_dir: bool,
    checkpoint_folder: str | None,
):
    if not run_name:
        raise ValueError("run_name is required and must be non-empty")

    log_root = Path(log_dir)

    if checkpoint_folder is None and exists(checkpoint_path):
        checkpoint = Path(checkpoint_path)
        checkpoint_folder_parts = Path("checkpoints").parts

        if (
            len(checkpoint_folder_parts) > 0 and
            len(checkpoint.parts) > len(checkpoint_folder_parts) and
            checkpoint.parts[-len(checkpoint_folder_parts) - 1:-1] == checkpoint_folder_parts
        ):
            return checkpoint.parents[len(checkpoint_folder_parts)]

        return checkpoint.parent

    # Retain unique_log_dir for config compatibility; run_name is the run directory.
    _ = unique_log_dir
    return log_root / run_name


def serialize_hyperparameters(hyperparameters: dict):
    return "```json\n" + json.dumps(hyperparameters, indent = 2, sort_keys = True, default = str) + "\n```"


def validate_dynamics_loss_normalization(
    use_dynamics_loss_normalization: bool,
    freeze_dynamics_loss_normalization_after_pretraining: bool,
):
    if freeze_dynamics_loss_normalization_after_pretraining and not use_dynamics_loss_normalization:
        raise ValueError(
            "freeze_dynamics_loss_normalization_after_pretraining requires "
            "use_dynamics_loss_normalization=true"
        )

    if use_dynamics_loss_normalization and not freeze_dynamics_loss_normalization_after_pretraining:
        warnings.warn(
            "Dynamics loss RMS statistics remain active after world-model pretraining. "
            "Set freeze_dynamics_loss_normalization_after_pretraining=true to freeze them.",
            UserWarning,
            stacklevel = 2,
        )


@close_runtime_resources
def main(
    *,
    env_name: str,
    num_loops: int,
    rollouts_per_loop: int,
    num_envs: int,
    env_vectorization_mode: VectorizationMode,
    max_timesteps: int,
    replay_size: int,
    seed: int,
    cpu: bool,
    obs_dim: int | None,
    num_latent_tokens: int,
    dim_latent: int,
    model_dim: int,
    attn_heads: int,
    attn_dim_head: int,
    depth: int,
    time_block_every: int,
    spatial_flex_attention_backend: str,
    final_special_cross_attn: bool,
    reward_encoder_type: Literal["symexp_two_hot", "hl_gauss"],
    use_symexp_value_head: bool,
    use_tdmpc2_value_head: bool = False,
    use_symlog_hl_gauss_value_head: bool = False,
    use_hl_gauss_actor_critic: bool = False,
    use_baseline_symexp_value_head_init: bool = False,
    reward_encoder_num_bins: int,
    value_encoder_num_bins: int,
    reward_use_symlog: bool,
    value_use_symlog: bool,
    reward_decode_symlog_expected_raw: bool,
    value_decode_symlog_expected_raw: bool,
    prob_shortcut_train: float | None,
    pretrain_world_model_finetune_steps: int,
    pretrain_world_model_combined_steps: int,
    policy_all_tokens: bool,
    world_model_batch_size: int,
    world_model_train_steps: int,
    world_model_train_sequence_length: int,
    world_model_learning_rate: float,
    use_dynamics_loss_normalization: bool,
    freeze_dynamics_loss_normalization_after_pretraining: bool,
    use_paper_aligned_shortcut_objective: bool,
    imagination_batch_size: int,
    imagination_horizon: int,
    imagination_prompt_length: int,
    imagination_prompt_probability: float,
    imagination_train_steps: int,
    imagination_generate_steps: int,
    imagination_use_time_cache: bool,
    preservation_sequence_length: int,
    agent_learning_rate: float,
    use_muon_optimizer: bool,
    optimizer_weight_decay: float,
    objective: Literal["ppo", "pmpo", "spo"],
    pmpo_pos_to_neg_weight: float,
    pmpo_kl_div_loss_weight: float,
    use_delight_gating: bool,
    full_transformer_finetuning: bool,
    agent_predicts_state: bool,
    agent_state_pred_loss_weight: float,
    pretrain_tokenizer_steps: int,
    pretrain_tokenizer_observations: int,
    tokenizer_batch_size: int,
    tokenizer_learning_rate: float,
    tokenizer_eval_every: int,
    tokenizer_eval_batch_size: int,
    max_grad_norm: float,
    use_tensorboard: bool,
    log_dir: str,
    run_name: str,
    run_details: str,
    checkpoint_folder: str | None,
    checkpoint_every: int,
    checkpoint_path: str | None,
    clear_log_dir: bool,
    unique_log_dir: bool,
    compile: bool | None,
    compile_interact: bool,
    compile_world_model: bool,
    compile_generate: bool,
    compile_learn: bool,
    compile_backend: str,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool | None,
    compile_generate_cudagraphs: bool,
    compile_options: Mapping[str, Any] | None = None,
    optimize_generate_compilation: bool,
    compile_generate_with_time_cache: bool,
    track_compile_performance: bool,
    allow_tf32: bool,
    require_cuda: bool,
    return_compile_timings: bool,
    collect_timings: bool,
    profile: bool,
    profile_memory: bool,
    profile_record_shapes: bool,
    profile_with_stack: bool,
    profile_with_flops: bool,
    normalize_advantages: bool = True,
    add_action_embed_to_agent_token: bool = False,
    old_continuous_action_encoder: bool = False,
    old_reward_shifting: bool = False,
    old_pmpo: bool = False,
    old_reward_head: bool = False,
    old_terminal_head: bool = False,
    old_state_tokenizer: bool = False,
    old_dynamics_ff: bool = False,
    old_policy_head: bool = False,
    old_value_head: bool = False,
    old_replay_sampling: bool = False,
    old_embedding_init: bool = False,
    reward_range: tuple[float, float] = (-4., 4.),
    reward_encoder_sigma_to_bin_ratio: float = 0.75,
    reward_min_max_value_on_bin_center: bool = True,
    value_range: tuple[float, float] = (-10., 10.),
    value_encoder_sigma_to_bin_ratio: float = 0.75,
    value_min_max_value_on_bin_center: bool = True,
    value_head_init_value: float | None = None,
    hl_gauss_actor_critic_range: tuple[float, float] = (-1200., 1200.),
    hl_gauss_actor_critic_num_bins: int = 255,
    hl_gauss_actor_critic_sigma_to_bin_ratio: float = 0.75,
    hl_gauss_actor_critic_min_max_value_on_bin_center: bool = False,
    activation_memory_budget: float = 0.9,
):
    hyperparameters = dict(locals())

    if not run_details:
        raise ValueError("run_details is required and must be non-empty")

    if full_transformer_finetuning and objective != "pmpo":
        raise ValueError("full_transformer_finetuning requires objective='pmpo' for the policy-prior loss")

    if full_transformer_finetuning and pmpo_kl_div_loss_weight <= 0.:
        raise ValueError("full_transformer_finetuning requires pmpo_kl_div_loss_weight > 0")

    validate_dynamics_loss_normalization(
        use_dynamics_loss_normalization,
        freeze_dynamics_loss_normalization_after_pretraining,
    )

    if exists(checkpoint_path) and clear_log_dir:
        raise ValueError("clear_log_dir must be false when checkpoint_path is set, otherwise resume state would be deleted")

    configured_profile_steps = (
        (
            pretrain_world_model_finetune_steps + pretrain_world_model_combined_steps
            if not exists(checkpoint_path) and num_loops > 0
            else 0
        )
        + num_loops * (world_model_train_steps + imagination_train_steps)
    )
    if profile and configured_profile_steps < PROFILE_TOTAL_STEPS:
        raise ValueError(
            f"profile mode requires at least {PROFILE_TOTAL_STEPS} configured world-model/imagination steps "
            f"(5 warmup + 4 active), but this run has {configured_profile_steps}"
        )

    compile_mode = normalize_compile_mode(compile_mode)
    obs_dim, action_dim, action_range = inspect_env_spaces(env_name, obs_dim)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")

    if compile is None:
        compile = device.type == "cuda"

    configure_sac(activation_memory_budget)

    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required for this run, but no CUDA device is available")

    if env_vectorization_mode not in ("async", "sync"):
        raise ValueError(f"env_vectorization_mode must be 'async' or 'sync', got {env_vectorization_mode!r}")

    allocated_cpus = available_cpu_count()
    if env_vectorization_mode == "async" and num_envs > allocated_cpus:
        print(
            f"warning: async vectorization requested {num_envs} environment workers, "
            f"but this process can use only {allocated_cpus} CPUs"
        )

    if allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    run_log_dir = resolve_log_dir(
        log_dir,
        run_name = run_name,
        checkpoint_path = checkpoint_path,
        unique_log_dir = unique_log_dir,
        checkpoint_folder = checkpoint_folder,
    )
    checkpoint_root = (
        run_log_dir / "checkpoints"
        if checkpoint_folder is None
        else Path(checkpoint_folder)
    )

    if not exists(checkpoint_path) and run_log_dir.exists():
        raise FileExistsError(
            f"run directory already exists for run_name={run_name!r}: {run_log_dir}. "
            "Use a new run_name or pass checkpoint_path to continue the existing run."
        )

    if clear_log_dir:
        shutil.rmtree(run_log_dir, ignore_errors = True)

    event_writer = SummaryWriter(str(run_log_dir)) if use_tensorboard or collect_timings else None
    writer = event_writer if use_tensorboard else None
    if exists(event_writer):
        register_runtime_cleanup(event_writer.close)
    training_profiler = TrainingProfiler(
        run_log_dir,
        device,
        writer = event_writer,
        collect_timings = collect_timings,
        profile = profile,
        profile_memory = profile_memory,
        profile_record_shapes = profile_record_shapes,
        profile_with_stack = profile_with_stack,
        profile_with_flops = profile_with_flops,
    )
    register_runtime_cleanup(training_profiler.close)
    if exists(writer):
        writer.add_text("run/details", run_details, 0)
        writer.add_text("run/hyperparameters", serialize_hyperparameters(hyperparameters), 0)

    tokenizer = ObservationTokenizer(
        obs_dim = obs_dim,
        num_latent_tokens = num_latent_tokens,
        dim_latent = dim_latent,
        old_state_tokenizer = old_state_tokenizer,
    ).to(device)

    tokenizer_loss = None

    if not exists(checkpoint_path):
        pretrain_obs = collect_random_observations(
            env_name,
            num_steps = max(1, pretrain_tokenizer_observations // num_envs),
            num_envs = num_envs,
            seed = seed,
            env_vectorization_mode = env_vectorization_mode,
        )[:pretrain_tokenizer_observations]

        assert pretrain_obs.shape[-1] == obs_dim, f"{env_name} produced obs dim {pretrain_obs.shape[-1]}, expected {obs_dim}"

        tokenizer_loss = train_tokenizer(
            tokenizer,
            pretrain_obs,
            steps = pretrain_tokenizer_steps,
            batch_size = tokenizer_batch_size,
            learning_rate = tokenizer_learning_rate,
            max_grad_norm = max_grad_norm,
            device = device,
            writer = writer,
            training_profiler = training_profiler,
        )

    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)

    env = make_env(
        env_name,
        seed,
        vectorized = True,
        num_envs = num_envs,
        episode_stats_buffer_length = max(100, num_envs * max_timesteps),
        env_vectorization_mode = env_vectorization_mode,
    )
    register_runtime_cleanup(env.close)

    value_encoder_kwargs = dict(
        reward_range = value_range,
        num_bins = value_encoder_num_bins,
    )

    if not (use_symexp_value_head or use_tdmpc2_value_head):
        value_encoder_kwargs.update(
            sigma_to_bin_ratio = value_encoder_sigma_to_bin_ratio,
            min_max_value_on_bin_center = value_min_max_value_on_bin_center,
            use_symlog = value_use_symlog,
            decode_symlog_expected_raw = value_decode_symlog_expected_raw,
        )

    if use_symlog_hl_gauss_value_head:
        value_encoder_kwargs.update(
            use_symlog = True,
            decode_symlog_expected_raw = True,
        )

    hl_gauss_actor_critic_kwargs = None

    if use_hl_gauss_actor_critic:
        hl_gauss_actor_critic_kwargs = dict(
            reward_range = hl_gauss_actor_critic_range,
            num_bins = hl_gauss_actor_critic_num_bins,
            sigma_to_bin_ratio = hl_gauss_actor_critic_sigma_to_bin_ratio,
            min_max_value_on_bin_center = hl_gauss_actor_critic_min_max_value_on_bin_center,
            use_symlog = False,
            decode_symlog_expected_raw = False,
        )

    world_model = DynamicsWorldModel(
        dim = model_dim,
        dim_latent = dim_latent,
        max_steps = 64,
        num_latent_tokens = num_latent_tokens,
        num_spatial_tokens = num_latent_tokens,
        num_register_tokens = 1,
        dim_critic_state = obs_dim,
        depth = depth,
        time_block_every = time_block_every,
        transformer_kwargs = dict(
            spatial_flex_attention_backend = spatial_flex_attention_backend,
        ),
        final_special_cross_attn = final_special_cross_attn,
        num_discrete_actions = 0,
        num_continuous_actions = action_dim,
        continuous_dist_type = "beta",
        continuous_target_action_range = action_range,
        reward_encoder_type = reward_encoder_type,
        use_symexp_value_head = use_symexp_value_head,
        use_tdmpc2_value_head = use_tdmpc2_value_head,
        use_symlog_hl_gauss_value_head = use_symlog_hl_gauss_value_head,
        use_hl_gauss_actor_critic = use_hl_gauss_actor_critic,
        use_baseline_symexp_value_head_init = use_baseline_symexp_value_head_init,
        reward_encoder_kwargs = dict(reward_range = reward_range, num_bins = reward_encoder_num_bins, sigma_to_bin_ratio = reward_encoder_sigma_to_bin_ratio, min_max_value_on_bin_center = reward_min_max_value_on_bin_center, use_symlog = reward_use_symlog, decode_symlog_expected_raw = reward_decode_symlog_expected_raw),
        value_encoder_kwargs = value_encoder_kwargs,
        hl_gauss_actor_critic_kwargs = hl_gauss_actor_critic_kwargs,
        predict_terminals = True,
        continuous_action_loss_weight = 0.,
        discrete_action_loss_weight = 0.,
        agent_predicts_state = agent_predicts_state,
        agent_state_pred_loss_weight = agent_state_pred_loss_weight,
        prob_shortcut_train = prob_shortcut_train,
        policy_all_tokens = policy_all_tokens,
        gae_discount_factor = 0.99,
        pmpo_pos_to_neg_weight = pmpo_pos_to_neg_weight,
        pmpo_kl_div_loss_weight = pmpo_kl_div_loss_weight,
        ppo_eps_clip = 0.2,
        normalize_advantages = normalize_advantages,
        add_action_embed_to_agent_token = add_action_embed_to_agent_token,
        old_continuous_action_encoder = old_continuous_action_encoder,
        old_reward_shifting = old_reward_shifting,
        old_pmpo = old_pmpo,
        old_reward_head = old_reward_head,
        old_terminal_head = old_terminal_head,
        old_dynamics_ff = old_dynamics_ff,
        old_policy_head = old_policy_head,
        old_value_head = old_value_head,
        old_embedding_init = old_embedding_init,
        value_head_init_value = value_head_init_value,
        policy_entropy_weight = 0.01,
        use_loss_normalization = use_dynamics_loss_normalization,
        use_paper_aligned_shortcut_objective = use_paper_aligned_shortcut_objective,
        attn_heads = attn_heads,
        attn_dim_head = attn_dim_head,
        policy_head_mlp_depth = 2,
        value_head_mlp_depth = 2,
    ).to(device)

    world_params = world_model.world_model_parameters()
    agent_params = agent_optimizer_parameters(world_model, full_transformer_finetuning)

    world_optimizer = make_optimizer(
        world_params,
        lr = world_model_learning_rate,
        weight_decay = optimizer_weight_decay,
        use_muon = use_muon_optimizer,
        muon_params = world_model.muon_parameters(),
    )
    agent_optimizer = make_optimizer(
        agent_params,
        lr = agent_learning_rate,
        weight_decay = optimizer_weight_decay,
        use_muon = use_muon_optimizer,
        muon_params = world_model.muon_parameters(),
    )

    start_loop = 0
    if exists(checkpoint_path):
        pkg = torch.load(checkpoint_path, map_location = device, weights_only = True)
        tokenizer.load_state_dict(pkg["tokenizer"])
        world_model.load_state_dict(pkg["world_model"])
        load_optimizer_state_if_compatible(world_optimizer, pkg["world_optimizer"], "world")
        load_optimizer_state_if_compatible(agent_optimizer, pkg["agent_optimizer"], "agent")
        start_loop = int(pkg.get("loop", 0)) + 1

    remaining_profile_steps = (
        (
            pretrain_world_model_finetune_steps + pretrain_world_model_combined_steps
            if start_loop == 0 and num_loops > 0
            else 0
        )
        + max(0, num_loops - start_loop) * (world_model_train_steps + imagination_train_steps)
    )
    if profile and remaining_profile_steps < PROFILE_TOTAL_STEPS:
        raise ValueError(
            f"profile mode requires {PROFILE_TOTAL_STEPS} remaining world-model/imagination steps, "
            f"but the loaded run has {remaining_profile_steps}"
        )

    policy_prior = FrozenPolicyPrior(world_model).to(device) if objective == "pmpo" else None
    compile_interact = compile or compile_interact
    compile_world_model = compile or compile_world_model
    compile_generate = compile or compile_generate
    compile_learn = compile or compile_learn
    static_compile_shapes = compile_dynamic is False

    if compile_generate and static_compile_shapes and imagination_prompt_length > 0 and imagination_prompt_probability != 1.:
        raise ValueError("compile_generate requires imagination_prompt_probability=1.0 when imagination_prompt_length > 0")

    compile_runtime = build_compile_runtime(
        world_model,
        device = device,
        compile_interact = compile_interact,
        interact_max_timesteps = max_timesteps,
        compile_world_model = compile_world_model,
        compile_generate = compile_generate,
        compile_learn = compile_learn,
        compile_backend = compile_backend,
        compile_mode = compile_mode,
        compile_fullgraph = compile_fullgraph,
        compile_dynamic = compile_dynamic,
        compile_generate_cudagraphs = compile_generate_cudagraphs,
        compile_options = compile_options,
        optimize_generate_compilation = optimize_generate_compilation,
        compile_generate_with_time_cache = compile_generate_with_time_cache,
        track_compile_performance = track_compile_performance,
        imagination_generate_steps = imagination_generate_steps,
        imagination_use_time_cache = imagination_use_time_cache,
        imagination_max_timesteps = imagination_horizon + imagination_prompt_length,
        objective = objective,
        use_delight_gating = use_delight_gating,
        store_old_action_unembeds = objective == "pmpo" and not exists(policy_prior),
        full_transformer_finetuning = full_transformer_finetuning,
    )

    memmap_path = run_log_dir / "replay_buffer"

    if start_loop == 0:
        shutil.rmtree(memmap_path, ignore_errors=True)
        replay = None
    else:
        replay = ReplayBuffer.from_folder(memmap_path)

    wm_step = 0
    imagination_step = 0
    env_step = 0
    did_world_model_pretrain = start_loop > 0

    print(f"training {env_name} from {obs_dim} raw observations on {device}")
    print(f"tensorboard log dir: {run_log_dir.absolute()}" if exists(event_writer) else "tensorboard disabled")
    if collect_timings:
        print(f"timing log dir: {training_profiler.log_dir.absolute()}")
    if profile:
        print("profile mode: 5 warmup steps, 4 measured steps, then exit")
    if compile_interact or compile_world_model or compile_generate or compile_learn:
        compiled_paths = [
            name
            for name, enabled in (
                ("interact_forward", compile_interact),
                ("world_model_loss", compile_world_model),
                ("imagination_generate", compile_generate),
                ("imagination_learn", compile_learn),
            )
            if enabled
        ]
        print(f"torch.compile enabled for: {', '.join(compiled_paths)}")
        if device.type == "cuda":
            cuda_graph_paths = [timing.name for timing in compile_runtime.timings if timing.cuda_graphs]

            if len(cuda_graph_paths) > 0:
                print(f"CUDA Graph capture enabled for: {', '.join(cuda_graph_paths)}")
            elif compile_mode_uses_cuda_graphs(compile_mode):
                print("CUDA Graph capture disabled for compiled paths")

        fallback_paths = [
            f"{timing.name}:{compile_mode}->{timing.compile_mode}"
            for timing in compile_runtime.timings
            if timing.compiled and exists(timing.compile_mode) and timing.compile_mode != compile_mode
        ]
        if len(fallback_paths) > 0:
            print(f"torch.compile mode fallback for: {', '.join(fallback_paths)}")
    if exists(tokenizer_loss):
        print(f"tokenizer pretrain recon loss: {tokenizer_loss:.4f}")

    pbar = tqdm(range(start_loop, num_loops), desc = "loops")

    for loop in pbar:
        with temp_eval(world_model):
            tokenizer_eval_loss_sum = 0.
            tokenizer_eval_sample_count = 0
            episodic_returns = []
            rollout_horizon_returns = []

            for rollout_idx in range(rollouts_per_loop):
                rollout_start = perf_counter()
                with training_profiler.record("rollout/model/interact_with_env"):
                    exp = world_model.interact_with_env(
                        env,
                        seed = seed if loop == 0 and rollout_idx == 0 else None,
                        max_timesteps = max_timesteps,
                        env_is_vectorized = True,
                        store_agent_embed = False,
                        store_old_action_unembeds = False,
                        obs_to_latents_fn = obs_to_latents_fn(tokenizer),
                        forward_fn = compile_runtime.interact_forward,
                    )

                rollout_seconds = perf_counter() - rollout_start
                reset_seconds, env_step_seconds = pop_rollout_env_timings(env)
                training_profiler.log_timing("rollout/environment/reset", reset_seconds * 1000.)
                training_profiler.log_timing("rollout/environment/step_total", env_step_seconds * 1000.)
                training_profiler.log_timing(
                    "rollout/model_and_orchestration",
                    max(rollout_seconds - reset_seconds - env_step_seconds, 0.) * 1000.,
                )
                training_profiler.flush_timings()

                if tokenizer_eval_every > 0 and divisible_by(loop, tokenizer_eval_every):
                    with training_profiler.record("rollout/tokenizer/evaluate", cuda = True):
                        tokenizer_eval_loss = eval_tokenizer_on_experience(
                            tokenizer,
                            exp,
                            max_samples = tokenizer_eval_batch_size,
                        )

                    if exists(tokenizer_eval_loss):
                        loss, sample_count = tokenizer_eval_loss
                        tokenizer_eval_loss_sum += loss * sample_count
                        tokenizer_eval_sample_count += sample_count

                if not exists(replay):
                    replay = Experience.create_memmap_replay_buffer(
                        exp,
                        memmap_path,
                        max_episodes = replay_size * num_envs,
                        max_timesteps = max_timesteps + 10,
                        circular = True,
                    )

                with training_profiler.record("rollout/replay/store"):
                    cpu_exp = experience_to_cpu(exp)
                    for single_exp in cpu_exp.unbind():
                        data, meta = single_exp.to_buffer_dict()
                        data, meta = tree_map_tensor(lambda t: rearrange(t, '1 ... -> ...'), (data, meta))
                        replay.store_episode(**data, **meta)

                rollout_returns, rollout_lengths = get_rollout_episode_stats(cpu_exp)
                rollout_horizon_returns.extend(rollout_returns)
                episodic_returns.extend(rollout_returns)

                rollout_steps = cpu_exp.rewards.shape[1]
                has_bootstrap_padding = exists(cpu_exp.is_truncated) and exists(cpu_exp.terminals) and (cpu_exp.is_truncated & ~cpu_exp.terminals).any()
                if has_bootstrap_padding:
                    rollout_steps -= 1

                env_step += rollout_steps * cpu_exp.rewards.shape[0]

                for episode_return, episode_length in zip(rollout_returns, rollout_lengths):
                    log_scalars(
                        writer,
                        {
                            "charts/episodic_return": episode_return,
                            "charts/episodic_length": episode_length,
                        },
                        env_step,
                    )

            avg_return = float(np.mean(episodic_returns)) if len(episodic_returns) > 0 else None
            avg_horizon_return = float(np.mean(rollout_horizon_returns)) if len(rollout_horizon_returns) > 0 else 0.

            valid_lens = replay.episode_lens[replay.episode_lens > 0]
            avg_length = float(valid_lens.mean().item()) if len(valid_lens) > 0 else 0.

            tokenizer_policy_recon_loss = tokenizer_eval_loss_sum / tokenizer_eval_sample_count if tokenizer_eval_sample_count > 0 else None

            log_scalars(
                writer,
                {
                    "rollout/average_return": avg_return,
                    "rollout/average_horizon_return": avg_horizon_return,
                    "rollout/replay_size": len(replay),
                    "rollout/average_length": avg_length,
                    "tokenizer/policy_recon_loss": tokenizer_policy_recon_loss,
                },
                loop,
            )

        if not did_world_model_pretrain:
            wm_step, _ = train_world_model(
                world_model,
                world_optimizer,
                replay,
                world_model_loss_fn = compile_runtime.world_model_loss,
                world_model_loss_timing = compile_runtime.world_model_loss_timing,
                world_model_step_timing = compile_runtime.world_model_step_timing,
                steps = pretrain_world_model_finetune_steps,
                batch_size = world_model_batch_size,
                sequence_length = world_model_train_sequence_length,
                static_batch_shape = compile_world_model and static_compile_shapes,
                max_grad_norm = max_grad_norm,
                global_step = wm_step,
                writer = writer,
                training_profiler = training_profiler,
                shortcut_train_mode = "finetune",
                desc = "world model pretrain finetune",
                old_replay_sampling = old_replay_sampling,
                update_loss_ema = True,
            )

            wm_step, _ = train_world_model(
                world_model,
                world_optimizer,
                replay,
                world_model_loss_fn = compile_runtime.world_model_loss,
                world_model_loss_timing = compile_runtime.world_model_loss_timing,
                world_model_step_timing = compile_runtime.world_model_step_timing,
                steps = pretrain_world_model_combined_steps,
                batch_size = world_model_batch_size,
                sequence_length = world_model_train_sequence_length,
                static_batch_shape = compile_world_model and static_compile_shapes,
                max_grad_norm = max_grad_norm,
                global_step = wm_step,
                writer = writer,
                training_profiler = training_profiler,
                shortcut_train_mode = "combined",
                desc = "world model pretrain combined",
                old_replay_sampling = old_replay_sampling,
                update_loss_ema = True,
            )

            did_world_model_pretrain = True

        wm_step, wm_metrics = train_world_model(
            world_model,
            world_optimizer,
            replay,
            world_model_loss_fn = compile_runtime.world_model_loss,
            world_model_loss_timing = compile_runtime.world_model_loss_timing,
            world_model_step_timing = compile_runtime.world_model_step_timing,
            steps = world_model_train_steps,
            batch_size = world_model_batch_size,
            sequence_length = world_model_train_sequence_length,
            static_batch_shape = compile_world_model and static_compile_shapes,
            max_grad_norm = max_grad_norm,
            global_step = wm_step,
            writer = writer,
            training_profiler = training_profiler,
            shortcut_train_mode = "combined",
            desc = "world model",
            old_replay_sampling = old_replay_sampling,
            update_loss_ema = not freeze_dynamics_loss_normalization_after_pretraining,
        )

        world_model.train()
        if exists(policy_prior):
            policy_prior.refresh_from(world_model)

        imagination_step, imagination_metrics = train_agent_in_imagination(
            world_model,
            agent_optimizer,
            policy_prior,
            replay,
            generate_rollout_fn = compile_runtime.generate_rollout,
            learn_from_dream_fn = compile_runtime.learn_from_dream,
            learn_from_dream_timing = compile_runtime.learn_from_dream_timing,
            imagination_step_timing = compile_runtime.imagination_step_timing,
            steps = imagination_train_steps,
            batch_size = imagination_batch_size,
            horizon = imagination_horizon,
            prompt_length = imagination_prompt_length,
            prompt_probability = imagination_prompt_probability,
            static_generate_shape = compile_generate and static_compile_shapes,
            generate_steps = imagination_generate_steps,
            max_grad_norm = max_grad_norm,
            objective = objective,
            use_delight_gating = use_delight_gating,
            global_step = imagination_step,
            writer = writer,
            training_profiler = training_profiler,
            old_replay_sampling = old_replay_sampling,
            full_transformer_finetuning = full_transformer_finetuning,
            preservation_sequence_length = preservation_sequence_length,
            preservation_batch_size = world_model_batch_size,
        )

        postfix_return = avg_return if exists(avg_return) else avg_horizon_return
        postfix = {"return": f"{postfix_return:.1f}", "replay": len(replay)}
        if wm_metrics:
            postfix["wm"] = f"{wm_metrics['world_model/loss']:.2f}"
        if imagination_metrics:
            postfix["dream"] = f"{imagination_metrics['imagination/dream_return']:.1f}"
        pbar.set_postfix(postfix)

        if checkpoint_every > 0 and divisible_by(loop + 1, checkpoint_every):
            with training_profiler.record("checkpoint/save"):
                save_checkpoint(
                    checkpoint_root / f"loop_{loop + 1}.pt",
                    loop = loop,
                    tokenizer = tokenizer,
                    world_model = world_model,
                    world_optimizer = world_optimizer,
                    agent_optimizer = agent_optimizer,
                )

        training_profiler.flush_timings(flush_writer = True)

    print_compile_timing_report(compile_runtime.timings)

    if exists(event_writer):
        event_writer.flush()

    if return_compile_timings:
        return compile_runtime.timings

if __name__ == "__main__":
    fire.Fire(main)
