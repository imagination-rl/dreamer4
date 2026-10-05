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
import multiprocessing
import os
import pickle
import random
import shutil
import subprocess
import warnings
from contextlib import nullcontext
from copy import deepcopy
from contextvars import ContextVar
from functools import partial, wraps
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
from torch.optim import Adam, AdamW, Optimizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils._pytree import tree_map

from einops import rearrange
from adam_atan2_pytorch import MuonAdamAtan2
from torch_einops_utils import safe_cat, temp_eval, lens_to_mask, pad_right_at_dim_to

from memmap_replay_buffer.replay_buffer import ReplayBuffer, ReplayDatasetTimeWindow, from_numpy as replay_from_numpy

from dreamer4.dreamer4 import (
    Actions,
    DynamicsWorldModel,
    Experience,
    ShortcutTrainMode,
    StateTokenizer,
    calc_gae,
    combine_experiences,
    default,
    divisible_by,
    exists,
    tree_map_tensor,
    z_score,
)

from .compile_runtime import (
    CallTiming,
    WorldModelTrainingLoss,
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
from .audit_ablations import AUDIT_ABLATION_FLAGS, AuditDiagnostics, isolated_rng, resolve_audit_ablations
from . import offline_pretraining as offline
from .env_worker import grouped_env_worker as _grouped_env_worker


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


# Flip-onset windows (report/10-4-26-instability-audit, report/10-4-26-pretrain-data). From the same start states, 3-8 %
# of the dreams end flipped and 9-41 % of the real rollouts; the policy acts near uniform on the states where the body
# tips. With a flip-onset fraction above 0, that share of the replay windows holds a real upright-to-flipped transition
# of the HalfCheetah root angle. The other windows are the normal windows.

FLIP_ONSET_ANGLE_INDEX = 1


class FlipOnsets:
    """The (replay slot, first flipped row) of each upright-to-flipped transition in the real episodes of a replay.

    refresh() reads only the slots that the replay wrote since the last call. The flip test is that of the audits:
    |wrap(root angle)| > pi / 2.
    """

    def __init__(self, replay: ReplayBuffer):
        self.replay = replay
        self.rows_of: dict[int, np.ndarray] = {}
        self.read_episode_index = None
        self.read_num_episodes = None
        self.slots = np.zeros(0, dtype = np.int64)
        self.rows = np.zeros(0, dtype = np.int64)

    def _read(self, slot: int):
        self.rows_of.pop(slot, None)
        length = int(self.replay.episode_lens[slot])
        if length < 2 or bool(self.replay.meta_data["is_from_world_model"][slot]):
            return
        angle = np.asarray(self.replay.data["critic_state"][slot, :length, FLIP_ONSET_ANGLE_INDEX], dtype = np.float64)
        flipped = np.abs(np.angle(np.exp(1j * angle))) > np.pi / 2
        rows = np.flatnonzero(~flipped[:-1] & flipped[1:]) + 1
        if len(rows) > 0:
            self.rows_of[slot] = rows

    def refresh(self) -> "FlipOnsets":
        replay = self.replay
        episode_index, num_episodes = replay.episode_index, replay.num_episodes
        if (episode_index, num_episodes) == (self.read_episode_index, self.read_num_episodes):
            return self

        if self.read_episode_index is None:
            slots = range(replay.max_episodes)
        else:
            count = (episode_index - self.read_episode_index) % replay.max_episodes
            if count == 0:
                count = replay.max_episodes
            slots = [(self.read_episode_index + offset) % replay.max_episodes for offset in range(count)]

        for slot in slots:
            self._read(slot)

        self.read_episode_index, self.read_num_episodes = episode_index, num_episodes
        ordered = sorted(self.rows_of.items())
        self.slots = np.array([slot for slot, rows in ordered for _ in rows], dtype = np.int64)
        self.rows = np.concatenate([rows for _, rows in ordered]) if ordered else np.zeros(0, dtype = np.int64)
        return self


def flip_onsets_of(replay: ReplayBuffer) -> FlipOnsets:
    """The FlipOnsets of a replay, kept on the replay object, so each call reads only the new episodes."""

    onsets = getattr(replay, "_flip_onsets", None)
    if not exists(onsets):
        onsets = FlipOnsets(replay)
        replay._flip_onsets = onsets
    return onsets.refresh()


class FlipOnsetTimeWindow(ReplayDatasetTimeWindow):
    """ReplayDatasetTimeWindow that returns a flip-onset window with probability `fraction`.

    A flip-onset window starts `before` rows before a random onset row (the first flipped row), with `before` uniform in
    [min_before, max_before] and the window start in [0, episode length - window length]. A replay without onsets gives
    only normal windows.
    """

    def __init__(self, replay_buffer: ReplayBuffer, window_length: int, *, fraction: float, min_before: int, max_before: int, **kwargs):
        super().__init__(replay_buffer, window_length = window_length, **kwargs)
        if not 0. < fraction < 1.:
            raise ValueError(f"the flip-onset fraction must be in (0, 1), got {fraction}")
        if not 0 <= min_before <= max_before:
            raise ValueError(f"the flip-onset window needs 0 <= min_before <= max_before, got {min_before}, {max_before}")
        self.fraction = fraction
        self.min_before = min_before
        self.max_before = max_before
        self.onsets = flip_onsets_of(self.replay_buffer)

    def __getitem__(self, idx):
        if len(self.onsets.rows) == 0 or torch.rand(()).item() >= self.fraction:
            return super().__getitem__(idx)

        pick = torch.randint(0, len(self.onsets.rows), ()).item()
        episode_index, row = int(self.onsets.slots[pick]), int(self.onsets.rows[pick])
        episode_len = int(self.replay_buffer.episode_lens[episode_index])
        last_start = max(episode_len - self.window_length, 0)
        low = min(max(row - self.max_before, 0), last_start)
        high = min(max(row - self.min_before, 0), last_start)
        start = torch.randint(low, high + 1, ()).item()
        return self.window(episode_index, start)

    def window(self, episode_index: int, start: int):
        # The body of ReplayDatasetTimeWindow.__getitem__ with a given episode and start.
        episode_len = self.replay_buffer.episode_lens[episode_index]
        actual_len = min(self.window_length, episode_len - start)
        end = start + actual_len

        data = dict()
        for field in self.fields:
            name = self.fieldname_map.get(field, field)
            t = replay_from_numpy(self.replay_buffer.data[field][episode_index, start:end].copy())
            if actual_len < self.window_length:
                t = pad_right_at_dim_to(t, self.window_length, dim = 0)
            data[name] = t

        for field, memmap in self.meta_data.items():
            name = self.fieldname_map.get(field, field)
            data[name] = replay_from_numpy(memmap[episode_index].copy())

        data.update(
            _lens = torch.tensor(actual_len, dtype = torch.long),
            _reaches_episode_end = torch.tensor(end >= episode_len, dtype = torch.bool),
        )
        if self.return_indices:
            data.update(_index = torch.tensor(episode_index), _start = torch.tensor(start))
        return data


def replay_time_windows(replay: ReplayBuffer, window_length: int, *, flip_onset_fraction: float = 0., min_before: int = 1, max_before: int | None = None, **kwargs):
    """ReplayDatasetTimeWindow, or FlipOnsetTimeWindow for a flip-onset fraction above 0."""

    if flip_onset_fraction <= 0.:
        return ReplayDatasetTimeWindow(replay, window_length = window_length, **kwargs)
    return FlipOnsetTimeWindow(
        replay,
        window_length = window_length,
        fraction = flip_onset_fraction,
        min_before = min_before,
        max_before = default(max_before, window_length - 1),
        **kwargs,
    )

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


def _concat_vector_infos(infos: list[dict], sizes: list[int]) -> dict:
    """Join the info dicts of the env groups along the env axis.

    A key that a group does not return gets zeros (and a False mask for "_key").
    """

    merged = {}
    keys = dict.fromkeys(key for info in infos for key in info)

    for key in keys:
        present = next(info[key] for info in infos if key in info)

        if isinstance(present, dict):
            merged[key] = _concat_vector_infos([info.get(key, {}) for info in infos], sizes)
            continue

        present = np.asarray(present)
        merged[key] = np.concatenate([
            np.asarray(info[key]) if key in info else np.zeros((size, *present.shape[1:]), dtype = present.dtype)
            for info, size in zip(infos, sizes)
        ])

    return merged


class GroupedAsyncVectorEnv(gym.vector.VectorEnv):
    """num_envs envs in parallel_envs worker processes. Each worker steps a sync vector env of its envs.

    Env i gets seed + i, as in the sync and async vector envs, so the data is the same.
    Fewer processes than envs keep the CPU use near parallel_envs threads.
    """

    def __init__(self, env_name: str, *, num_envs: int, parallel_envs: int, context: str = "spawn"):
        if not (0 < parallel_envs <= num_envs):
            raise ValueError(f"parallel_envs must be in [1, num_envs={num_envs}], got {parallel_envs}")

        self.sizes = [len(chunk) for chunk in np.array_split(np.arange(num_envs), parallel_envs)]
        self.offsets = np.cumsum([0, *self.sizes]).tolist()

        probe = gym.make(env_name)
        self.num_envs = num_envs
        self.single_observation_space = probe.observation_space
        self.single_action_space = probe.action_space
        self.observation_space = gym.vector.utils.batch_space(probe.observation_space, num_envs)
        self.action_space = gym.vector.utils.batch_space(probe.action_space, num_envs)
        self.metadata = dict(probe.metadata, autoreset_mode = gym.vector.AutoresetMode.NEXT_STEP)
        self.spec = probe.spec
        self.render_mode = probe.render_mode
        probe.close()

        ctx = multiprocessing.get_context(context)
        self.pipes, self.processes = [], []

        for size in self.sizes:
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(target = _grouped_env_worker, args = (child_pipe, parent_pipe, env_name, size), daemon = True)
            process.start()
            child_pipe.close()
            self.pipes.append(parent_pipe)
            self.processes.append(process)

    def _receive(self):
        out = []
        for pipe in self.pipes:
            status, value = pipe.recv()
            if status != "ok":
                raise RuntimeError(f"GroupedAsyncVectorEnv worker failed:\n{value}")
            out.append(value)
        return out

    def _slice(self, value, index: int):
        return value[self.offsets[index]:self.offsets[index + 1]]

    def reset(self, *, seed = None, options = None):
        for index, pipe in enumerate(self.pipes):
            if seed is None:
                group_seed = None
            elif isinstance(seed, int):
                group_seed = seed + self.offsets[index]
            else:
                group_seed = list(self._slice(seed, index))

            group_options = options
            if options is not None and "reset_mask" in options:
                group_options = dict(options, reset_mask = self._slice(options["reset_mask"], index))

            pipe.send(("reset", dict(seed = group_seed, options = group_options)))

        out = self._receive()
        return np.concatenate([obs for obs, _ in out]), _concat_vector_infos([info for _, info in out], self.sizes)

    def step(self, actions):
        actions = np.asarray(actions)
        for index, pipe in enumerate(self.pipes):
            pipe.send(("step", self._slice(actions, index)))

        out = self._receive()
        obs, rewards, terminations, truncations = (np.concatenate([group[i] for group in out]) for i in range(4))
        return obs, rewards, terminations, truncations, _concat_vector_infos([group[4] for group in out], self.sizes)

    def close_extras(self, **kwargs):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout = 5)
            if process.is_alive():
                process.terminate()
        for pipe in self.pipes:
            pipe.close()


def make_env(
    env_name: str,
    seed: int | None,
    *,
    vectorized = True,
    num_envs = 8,
    parallel_envs: int | None = None,
    episode_stats_buffer_length = 100,
    env_vectorization_mode: VectorizationMode = "async",
):
    """parallel_envs sets the number of async worker processes. None gives one process per env."""

    if env_vectorization_mode not in ("async", "sync"):
        raise ValueError(f"env_vectorization_mode must be 'async' or 'sync', got {env_vectorization_mode!r}")

    if vectorized and env_vectorization_mode == "async" and exists(parallel_envs) and parallel_envs < num_envs:
        env = GroupedAsyncVectorEnv(env_name, num_envs = num_envs, parallel_envs = parallel_envs)
        env = RolloutTimingWrapper(env)
        env = gym.wrappers.vector.RecordEpisodeStatistics(env, buffer_length = episode_stats_buffer_length)
    elif vectorized:
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
    parallel_envs: int | None = None,
    seed = 42,
    env_vectorization_mode: VectorizationMode = "async",
):
    env = make_env(
        env_name,
        seed,
        vectorized = True,
        num_envs = num_envs,
        parallel_envs = parallel_envs,
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

    # MuonAdamAtan2 removes the Muon parameters through a set, which gives a
    # different parameter order in each process. A saved optimizer state then
    # maps to the wrong parameters on load. Keep the module order instead.
    #
    # Compatibility is broken: checkpoints saved before this change used a
    # random order that was not saved. Many non-Muon parameters have the same
    # shape, so the order cannot be recovered. Their optimizer state loads into
    # the wrong parameters, or fails with a shape error.
    muon_param_set = set(muon_params)

    return MuonAdamAtan2(
        muon_params = muon_params,
        params = [param for param in params if param not in muon_param_set],
        lr = lr,
        weight_decay = weight_decay,
        remove_muon_params_from_params = False,
    )


def make_ppo_reuse_optimizer(
    world_model: DynamicsWorldModel,
    *,
    lr: float,
    weight_decay: float,
):
    policy_params = unique_parameters(world_model.policy_head_parameters())
    value_params = unique_parameters(world_model.value_head_parameters())
    overlap = set(policy_params) & set(value_params)
    if overlap:
        raise ValueError("PPO reuse requires separate policy and value parameters")

    fused = all(param.is_cuda for param in (*policy_params, *value_params))
    return AdamW(
        [
            {"params": policy_params, "ppo_role": "policy"},
            {"params": value_params, "ppo_role": "value"},
        ],
        lr = lr,
        weight_decay = weight_decay,
        fused = fused,
    )


def make_cleanrl_updated_ppo_optimizer(
    world_model: DynamicsWorldModel,
    *,
    lr: float,
):
    """Build the optimizer from CleanRL updated PPO for the main model heads."""

    policy_params = unique_parameters(world_model.policy_head_parameters())
    value_params = unique_parameters(world_model.value_head_parameters())
    overlap = set(policy_params) & set(value_params)
    if overlap:
        raise ValueError("CleanRL updated PPO requires separate policy and value parameters")

    params = unique_parameters((*policy_params, *value_params))
    fused = len(params) > 0 and all(param.is_cuda for param in params)
    return Adam(params, lr = lr, eps = 1e-5, fused = fused)


def ppo_optimizer_role_parameters(optimizer: Optimizer, role: str):
    return [
        param
        for group in optimizer.param_groups
        if group.get("ppo_role") == role
        for param in group["params"]
    ]


def snapshot_optimizer_parameters(optimizer: Optimizer, params: list[nn.Parameter]):
    return [
        (
            param.detach().clone(),
            deepcopy(optimizer.state[param]) if param in optimizer.state else None,
        )
        for param in params
    ]


@torch.no_grad()
def restore_optimizer_parameters(
    optimizer: Optimizer,
    params: list[nn.Parameter],
    snapshot,
):
    for param, (value, state) in zip(params, snapshot):
        param.copy_(value)
        if state is None:
            optimizer.state.pop(param, None)
        else:
            optimizer.state[param] = deepcopy(state)


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

    # Empty (and skipped) unless behavior_cloning_in_world_model_updates puts the policy head in the optimizer.
    # It comes before action_head: the policy output layers (the action unembeds) are in the action embedder,
    # and the BC gradient must not scale down the gradient of the action embedding of the world model.
    policy_param_set = set(policy_head_parameters_or_empty(world_model))
    policy_params = [param for param in world_model.parameters() if param in policy_param_set]

    return disjoint_optimizer_param_groups(
        optimizer,
        [
            ("reward_head", module_parameters(reward_head)),
            ("policy_head", policy_params),
            ("action_head", module_parameters(getattr(world_model, "action_embedder", None))),
            ("terminal_head", module_parameters(terminal_head)),
            ("state_head", module_parameters(getattr(world_model, "to_state_pred", None))),
            ("agent_state_head", module_parameters(getattr(world_model, "to_agent_state_pred", None))),
            ("latent_ar_head", module_parameters(getattr(world_model, "latent_ar", None))),
        ],
        "trunk",
    )


def policy_head_parameters_or_empty(world_model) -> list:
    # Test doubles of the world model have no policy head.
    return world_model.policy_head_parameters() if hasattr(world_model, "policy_head_parameters") else []


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


def align_dream_transition_rewards(dream: Experience):
    """Pair each generated action with its next generated arrival reward."""
    if not exists(dream.rewards) or dream.rewards.shape[1] < 2:
        raise ValueError("dream transition alignment requires at least two generated states")

    aligned_rewards = torch.zeros_like(dream.rewards)
    aligned_rewards[:, :-1] = dream.rewards[:, 1:]
    dream.rewards = aligned_rewards

    if exists(dream.terminals) and dream.terminals.ndim > 1:
        aligned_terminals = torch.zeros_like(dream.terminals)
        aligned_terminals[:, :-1] = dream.terminals[:, 1:]
        dream.terminals = aligned_terminals

    if exists(dream.lens):
        reward_mask = lens_to_mask(dream.lens, dream.rewards.shape[1])
        dream.episode_return = (dream.rewards * reward_mask.float()).sum(dim = 1)
    else:
        dream.episode_return = dream.rewards.sum(dim = 1)

    return dream


def select_experience_batch(experience: Experience, indices: Tensor):
    batch_size = experience.latents.shape[0]

    def select(value):
        if isinstance(value, Actions):
            return Actions(*(select(item) for item in value))

        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            return value.index_select(0, indices)

        return value

    return Experience(**{name: select(value) for name, value in vars(experience).items()})


@torch.no_grad()
def freeze_ppo_targets(
    world_model: DynamicsWorldModel,
    dream: Experience,
    *,
    normalize_advantages: bool = True,
):
    rewards = dream.rewards.clone()
    values = dream.values.clone()
    batch, time = rewards.shape[:2]
    lens = dream.lens if exists(dream.lens) else torch.full(
        (batch,), time, device = rewards.device, dtype = torch.long
    )
    is_truncated = dream.is_truncated if exists(dream.is_truncated) else torch.ones(
        (batch,), device = rewards.device, dtype = torch.bool
    )

    valid_mask = lens_to_mask(lens, time)
    rewards.masked_fill_(~valid_mask, 0.)
    values.masked_fill_(~valid_mask, 0.)

    learn_mask = lens_to_mask(lens - is_truncated.long(), time)
    gae_mask = lens_to_mask((lens - 1).clamp(min = 0), time)

    if exists(dream.terminals):
        terminals = dream.terminals
        if terminals.ndim == 1:
            positions = (lens - 1).clamp(min = 0)
            terminals = torch.arange(time, device = rewards.device)[None, :] == positions[:, None]
            terminals = terminals & dream.terminals[:, None]
        gae_mask.masked_fill_(terminals.bool(), 0.)

    returns = calc_gae(
        rewards,
        values,
        masks = gae_mask,
        learn_masks = learn_mask,
        gamma = world_model.gae_discount_factor,
        lam = world_model.gae_lambda,
        use_accelerated = world_model.gae_use_accelerated,
    )
    advantages = returns - values
    if normalize_advantages:
        advantages = z_score(advantages, mask = learn_mask, eps = 1e-6)

    dream.frozen_returns = returns
    dream.frozen_advantages = advantages
    return dream


@torch.no_grad()
def flatten_cleanrl_updated_ppo_transitions(
    world_model: DynamicsWorldModel,
    dream: Experience,
):
    """Freeze GAE targets and flatten valid dream transitions into one batch."""

    dream = freeze_ppo_targets(
        world_model,
        dream,
        normalize_advantages = False,
    )
    batch, time = dream.rewards.shape[:2]
    lens = dream.lens if exists(dream.lens) else torch.full(
        (batch,),
        time,
        device = dream.rewards.device,
        dtype = torch.long,
    )
    is_truncated = dream.is_truncated if exists(dream.is_truncated) else torch.ones(
        batch,
        device = dream.rewards.device,
        dtype = torch.bool,
    )
    learn_mask = lens_to_mask((lens - is_truncated.long()).clamp(min = 0), time)
    transition_count = int(learn_mask.sum().item())
    if transition_count == 0:
        raise ValueError("CleanRL updated PPO needs at least one valid dream transition")

    def flatten(value):
        if not exists(value):
            return None
        if isinstance(value, Actions):
            return Actions(*(flatten(item) for item in value))
        if (
            torch.is_tensor(value)
            and value.ndim >= 2
            and tuple(value.shape[:2]) == (batch, time)
        ):
            return value[learn_mask].unsqueeze(1)
        return value

    return Experience(
        latents = flatten(dream.latents),
        video = flatten(dream.video),
        proprio = flatten(dream.proprio),
        critic_state = flatten(dream.critic_state),
        agent_embed = flatten(dream.agent_embed),
        policy_input = flatten(dream.policy_input),
        rewards = flatten(dream.rewards),
        terminals = None,
        actions = flatten(dream.actions),
        log_probs = flatten(dream.log_probs),
        old_action_unembeds = flatten(dream.old_action_unembeds),
        values = flatten(dream.values),
        frozen_returns = flatten(dream.frozen_returns),
        frozen_advantages = flatten(dream.frozen_advantages),
        step_size = dream.step_size,
        lens = torch.ones(transition_count, device = dream.rewards.device, dtype = torch.long),
        is_truncated = torch.zeros(
            transition_count,
            device = dream.rewards.device,
            dtype = torch.bool,
        ),
        agent_index = dream.agent_index,
        is_from_world_model = dream.is_from_world_model,
        episode_return = None,
    )


def policy_unembeds_from_inputs(
    world_model: DynamicsWorldModel,
    policy_inputs: Tensor,
):
    policy_embed = world_model.policy_head(policy_inputs.detach())
    return Actions(*world_model.action_embedder.unembed(policy_embed, pred_head_index = 0))


def exact_policy_kl(
    world_model: DynamicsWorldModel,
    policy_inputs: Tensor,
    behavior_unembeds: Actions,
    *,
    lens: Tensor | None = None,
    is_truncated: Tensor | None = None,
):
    current_unembeds = policy_unembeds_from_inputs(world_model, policy_inputs)
    return mean_action_distribution_kl(
        world_model.action_embedder,
        behavior_unembeds,
        current_unembeds,
        lens = lens,
        is_truncated = is_truncated,
    )


def mean_action_distribution_kl(
    action_embedder: nn.Module,
    behavior_unembeds: Actions,
    current_unembeds: Actions,
    *,
    lens: Tensor | None = None,
    is_truncated: Tensor | None = None,
):
    kl_terms = action_embedder.kl_div(behavior_unembeds, current_unembeds)
    kl_values = None

    for kl_term in kl_terms:
        if not exists(kl_term):
            continue

        if kl_term.ndim > 2:
            kl_term = kl_term.flatten(start_dim = 2).sum(dim = -1)

        kl_values = kl_term if not exists(kl_values) else kl_values + kl_term

    if not exists(kl_values):
        raise ValueError("exact policy KL requires at least one action distribution")

    if exists(lens):
        if not exists(is_truncated):
            is_truncated = torch.ones_like(lens, dtype = torch.bool)
        learnable_lens = (lens - is_truncated.long()).clamp(min = 0, max = kl_values.shape[1])
        mask = lens_to_mask(learnable_lens, kl_values.shape[1])
        kl_values = kl_values[mask]
    else:
        kl_values = kl_values.reshape(-1)

    if kl_values.numel() == 0:
        reference = next(value for value in behavior_unembeds if exists(value))
        return reference.new_zeros(())

    return kl_values.mean()


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

    def action_unembeds(self, policy_inputs: Tensor, *, detach_inputs: bool = True):
        if detach_inputs:
            policy_inputs = policy_inputs.detach()
        policy_embed = self.policy_head(policy_inputs)
        return self.action_embedder.unembed(policy_embed, pred_head_index = 0)


def policy_inputs_from_replay_context(
    world_model: DynamicsWorldModel,
    experience: Experience,
    *,
    step_size: int,
):
    actions = action_pair_or_empty(experience.actions)
    _, (embeds, _) = world_model(
        latents = experience.latents,
        signal_levels = world_model.max_steps - 1,
        step_sizes = step_size,
        rewards = experience.rewards,
        discrete_actions = actions.discrete,
        continuous_actions = actions.continuous,
        proprio = experience.proprio,
        latent_is_noised = True,
        latent_has_view_dim = experience.latents.ndim == 5,
        return_pred_only = True,
        return_intermediates = True,
    )
    return world_model.select_policy_input(embeds, experience.agent_index)


def rollout_context_windows(
    experience: Experience,
    *,
    num_windows: int,
    window_length: int,
    rng: np.random.Generator,
) -> Experience | None:
    """Random real windows of one rollout (the experience of the newest episodes), with the replay window layout."""

    total_time = experience.latents.shape[1]
    lens = experience.lens.cpu().numpy() if exists(experience.lens) else np.full(experience.latents.shape[0], total_time)
    usable = np.flatnonzero(lens >= window_length)
    if len(usable) == 0:
        return None

    episodes = rng.choice(usable, size = num_windows, replace = len(usable) < num_windows)
    starts = np.array([rng.integers(0, lens[episode] - window_length + 1) for episode in episodes])
    episode_index = torch.from_numpy(episodes)[:, None]
    time_index = torch.from_numpy(starts)[:, None] + torch.arange(window_length)[None, :]

    def take(t):
        return t[episode_index, time_index] if exists(t) else None

    actions = experience.actions
    return Experience(
        latents = take(experience.latents),
        proprio = take(experience.proprio),
        critic_state = take(experience.critic_state),
        rewards = take(experience.rewards),
        actions = Actions(take(actions.discrete), take(actions.continuous)) if exists(actions) else None,
        step_size = experience.step_size,
        lens = torch.full((num_windows,), window_length, dtype = torch.long),
        agent_index = experience.agent_index,
        is_from_world_model = False,
    )


@torch.no_grad()
def agent_head_inputs_from_replay_context(
    world_model: DynamicsWorldModel,
    experience: Experience,
    *,
    step_size: int,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """The inputs of the policy head and the value head on real windows, as in interact_with_env (clean latents).

    The forward runs in chunks of windows, in eval mode, without gradients.
    """

    was_training = world_model.training
    world_model.eval()
    policy_inputs, value_inputs = [], []

    for start in range(0, experience.latents.shape[0], chunk_size):
        rows = slice(start, start + chunk_size)
        take = lambda t: t[rows] if exists(t) else None
        actions = action_pair_or_empty(experience.actions)
        _, (embeds, _) = world_model(
            latents = take(experience.latents),
            signal_levels = world_model.max_steps - 1,
            step_sizes = step_size,
            rewards = take(experience.rewards),
            discrete_actions = take(actions.discrete),
            continuous_actions = take(actions.continuous),
            proprio = take(experience.proprio),
            latent_is_noised = True,
            latent_has_view_dim = experience.latents.ndim == 5,
            return_pred_only = True,
            return_intermediates = True,
        )
        policy_inputs.append(world_model.select_policy_input(embeds, experience.agent_index))

        value_input = default(embeds.critic, embeds.agent)[..., experience.agent_index, :]
        if exists(world_model.critic_state_embedder) and exists(experience.critic_state):
            value_input = value_input + world_model.critic_state_embedder(take(experience.critic_state))
        value_inputs.append(value_input)

    world_model.train(was_training)
    return torch.cat(policy_inputs), torch.cat(value_inputs)


class HeadReanchor:
    """Head re-anchor after a world-model phase (reports/9-28-26, item D1).

    A world-model phase changes the transformer features under the policy and value heads, so the acting policy
    changes without an RL step. Before the phase, capture() records the head outputs on real windows of the newest
    rollout. After the phase, fit() trains the policy and value heads, on the new (fixed) features of the same
    windows, back to those outputs: the policy with the exact action KL, and the value logits with the cross entropy
    to the recorded bin distributions. The transformer does not change. A new Adam optimizer runs each fit, so the
    agent optimizer state does not change.
    """

    def __init__(self, world_model: DynamicsWorldModel, *, steps: int, learning_rate: float, step_size: int, chunk_size: int):
        self.world_model = world_model
        self.steps = steps
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.chunk_size = chunk_size
        self.context = None

    def value_heads(self):
        heads = [self.world_model.value_head]
        if exists(self.world_model.actor_critic_value_head):
            heads.append(self.world_model.actor_critic_value_head)
        return heads

    def policy_unembeds(self, policy_input: Tensor) -> Actions:
        return Actions(*self.world_model.action_embedder.unembed(self.world_model.policy_head(policy_input), pred_head_index = 0))

    @torch.no_grad()
    def capture(self, context: Experience | None):
        self.context = None
        if not exists(context):
            return

        context = context.to(self.world_model.device)
        policy_input, value_input = agent_head_inputs_from_replay_context(
            self.world_model, context, step_size = self.step_size, chunk_size = self.chunk_size,
        )
        self.context = context
        self.target_policy = self.policy_unembeds(policy_input)
        self.target_values = [head(value_input).softmax(dim = -1) for head in self.value_heads()]

    def losses(self, policy_input: Tensor, value_input: Tensor) -> tuple[Tensor, Tensor]:
        policy_kl = mean_action_distribution_kl(
            self.world_model.action_embedder,
            self.target_policy,
            self.policy_unembeds(policy_input),
            lens = self.context.lens,
            is_truncated = torch.zeros_like(self.context.lens, dtype = torch.bool),
        )
        value_ce = sum(
            -(target * head(value_input).log_softmax(dim = -1)).sum(dim = -1).mean()
            for head, target in zip(self.value_heads(), self.target_values)
        )
        return policy_kl, value_ce

    def fit(self) -> dict[str, float]:
        if not exists(self.context):
            return {}

        policy_input, value_input = agent_head_inputs_from_replay_context(
            self.world_model, self.context, step_size = self.step_size, chunk_size = self.chunk_size,
        )
        params = [*self.world_model.policy_head_parameters(), *self.world_model.value_head_parameters()]
        optimizer = torch.optim.Adam(params, lr = self.learning_rate)

        with torch.no_grad():
            policy_kl_before, value_ce_before = self.losses(policy_input, value_input)

        for _ in range(self.steps):
            policy_kl, value_ce = self.losses(policy_input, value_input)
            optimizer.zero_grad(set_to_none = True)
            (policy_kl + value_ce).backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none = True)
        with torch.no_grad():
            policy_kl_after, value_ce_after = self.losses(policy_input, value_input)

        self.context = None
        return {
            "reanchor/policy_kl_before": float(policy_kl_before),
            "reanchor/policy_kl_after": float(policy_kl_after),
            "reanchor/value_ce_before": float(value_ce_before),
            "reanchor/value_ce_after": float(value_ce_after),
        }


@torch.no_grad()
def agent_approx_kl_metrics(
    world_model: DynamicsWorldModel,
    dream: Experience,
    *,
    include_cleanrl_metrics: bool = False,
):
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
    ratio = log_ratio.exp()
    approx_kl = ratio - 1. - log_ratio

    lens = dream.lens
    if exists(lens):
        is_truncated = dream.is_truncated if exists(dream.is_truncated) else torch.ones_like(lens, dtype = torch.bool)
        learnable_lens = (lens - is_truncated.long()).clamp(min = 0, max = policy_time)
        mask = lens_to_mask(learnable_lens, policy_time)
        approx_kl = approx_kl[mask]
        if include_cleanrl_metrics:
            log_ratio = log_ratio[mask]
            ratio = ratio[mask]
    else:
        approx_kl = approx_kl.reshape(-1)
        if include_cleanrl_metrics:
            log_ratio = log_ratio.reshape(-1)
            ratio = ratio.reshape(-1)

    if approx_kl.numel() == 0:
        approx_kl = policy_inputs.new_zeros((1,))

    metrics = {
        "approx_kl_mean": approx_kl.mean(),
        "approx_kl_min": approx_kl.min(),
        "approx_kl_max": approx_kl.max(),
    }

    if include_cleanrl_metrics:
        if log_ratio.numel() == 0:
            log_ratio = policy_inputs.new_zeros((1,))
            ratio = policy_inputs.new_ones((1,))

        metrics.update({
            "old_approx_kl": (-log_ratio).mean(),
            "clip_fraction": ((ratio - 1.).abs() > world_model.ppo_eps_clip).float().mean(),
        })

    return metrics


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


def replay_batch_to_experience(batch, device: torch.device):
    exp = Experience.from_buffer_dict(batch)
    exp.lens = batch.get('_lens')
    exp = exp.to(device)

    # Mask episode flags when the sampled window does not reach its end.
    reaches_episode_end = batch.get('_reaches_episode_end')
    if exists(reaches_episode_end):
        reaches_episode_end = reaches_episode_end.to(device)
        if exists(exp.terminals): exp.terminals = exp.terminals & reaches_episode_end
        if exists(exp.is_truncated): exp.is_truncated = exp.is_truncated & reaches_episode_end

    return exp


def world_model_loss_metrics(loss: Tensor, losses) -> dict[str, float]:
    return {
        "world_model/loss": loss.item(),
        "world_model/flow_loss": losses.flow.item(),
        "world_model/shortcut_loss": losses.shortcut.item(),
        "world_model/reward_loss": (losses.agent_embed_reward.mean() + losses.latent_state_reward.mean()).item(),
        "world_model/reward_mae": losses.reward_mae.item(),
        "world_model/terminal_loss": (losses.agent_embed_terminal.mean() + losses.latent_state_terminal.mean()).item(),
        "world_model/agent_state_pred_loss": losses.agent_state_pred.item(),
    }


@torch.no_grad()
def world_model_validation_metrics(
    world_model: DynamicsWorldModel,
    replay: ReplayBuffer,
    *,
    batch_size: int,
    sequence_length: int | None,
    num_batches: int,
    micro_batch_size: int | None = None,
):
    """Mean world-model loss terms on held-out replay windows, without an update.

    The eager loss avoids a second compiled graph for no_grad calls. The RNG
    state is restored, so validation does not change the training stream.
    With micro_batch_size, the loss runs on chunks of each batch (the same windows),
    and the metrics are the chunk means weighted by chunk size. The eager fp32 loss
    of 128 windows of 200 steps needs about 9 GB; two pretrain runs on one 16 GB GPU
    ran out of memory there. The noise draws of the loss differ from one pass.
    """

    if len(replay) == 0 or num_batches <= 0:
        return {}

    loss_fn = WorldModelTrainingLoss(world_model)
    dataset = ReplayDatasetTimeWindow(replay, window_length = sequence_length, include_metadata = True)
    dataloader = DataLoader(dataset, batch_size = min(batch_size, len(dataset)), shuffle = True)
    totals = {}

    with isolated_rng(world_model.device):
        for _, batch in zip(range(num_batches), cycle(dataloader)):
            exp = replay_batch_to_experience(batch, world_model.device)
            windows = exp.latents.shape[0]
            chunk = micro_batch_size if exists(micro_batch_size) and micro_batch_size < windows else windows
            chunk_totals = {}

            for start in range(0, windows, chunk):
                part = slice(start, start + chunk)
                continuous = exp.actions.continuous if exists(exp.actions) else None
                loss, losses = loss_fn(
                    exp.latents[part],
                    exp.rewards[part] if exists(exp.rewards) else None,
                    exp.terminals[part] if exists(exp.terminals) else None,
                    continuous[part] if exists(continuous) else None,
                    exp.lens[part] if exists(exp.lens) else None,
                    "combined",
                    False,
                )

                size = exp.latents[part].shape[0]
                for key, value in world_model_loss_metrics(loss, losses).items():
                    chunk_totals[key] = chunk_totals.get(key, 0.) + value * size

            for key, value in chunk_totals.items():
                totals[key] = totals.get(key, 0.) + value / windows

    return {key: value / num_batches for key, value in totals.items()}


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
    preserve_policy_during_world_model_updates: bool = False,
    world_model_policy_consistency_loss_weight: float = 1.,
    policy_consistency_step_size: int = 16,
    rehearsal_replay: ReplayBuffer | None = None,
    rehearsal_fraction: float = 0.,
    micro_batch_size: int | None = None,
    autocast_bf16: bool = False,
    extra_loss_fn: Callable | None = None,
    grad_clip_groups: list[tuple[str, list[nn.Parameter]]] | None = None,
    flip_onset_window_fraction: float = 0.,
):
    if len(replay) == 0 or steps <= 0:
        return global_step, {}

    # Gradient accumulation keeps the batch of `batch_size` windows on a GPU that
    # cannot hold it in one pass. Chunks have a fixed size, so a compiled loss
    # keeps one static shape.
    if exists(micro_batch_size) and micro_batch_size < batch_size:
        if batch_size % micro_batch_size != 0:
            raise ValueError(f"world_model_batch_size ({batch_size}) must be a multiple of world_model_micro_batch_size ({micro_batch_size})")
    else:
        micro_batch_size = None

    last_metrics = {}
    # A world model with a BC loss (behavior_cloning_in_world_model_updates) also trains the policy head here.
    # The last PPO minibatch leaves its gradient on the policy head, so clear it before the first BC backward.
    behavior_cloning = getattr(world_model, "has_action_loss_weight", False)
    if behavior_cloning:
        optimizer.zero_grad(set_to_none = True)
    grad_clip_groups = default(grad_clip_groups, world_model_clip_groups(world_model, optimizer))

    # Offline rehearsal replaces part of each batch with offline windows.
    # The total batch size, and thus the static compile shape, does not change.
    online_batch_size, rehearsal_batch_size = (
        offline.rehearsal_batch_sizes(batch_size, rehearsal_fraction)
        if exists(rehearsal_replay)
        else (batch_size, 0)
    )

    # Flip-onset windows: the windows hold the rows before and after the onset (min_before 1).
    dataset = replay_time_windows(
        replay,
        window_length = sequence_length,
        flip_onset_fraction = flip_onset_window_fraction,
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
            batch_size = online_batch_size,
            shuffle = True,
            drop_last = False,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size = online_batch_size,
            shuffle = True,
            drop_last = len(dataset) >= online_batch_size
        )

    dataloader_iter = (
        resampling_dataloader_iterator(dataloader)
        if old_replay_sampling and len(dataloader) > 0
        else cycle(dataloader) if len(dataloader) > 0
        else None
    )

    rehearsal_iter = None
    if rehearsal_batch_size > 0:
        rehearsal_dataset = ReplayDatasetTimeWindow(
            rehearsal_replay,
            window_length = sequence_length,
            include_metadata = True
        )
        rehearsal_iter = cycle(DataLoader(
            rehearsal_dataset,
            batch_size = rehearsal_batch_size,
            shuffle = True,
            drop_last = len(rehearsal_dataset) >= rehearsal_batch_size
        ))

    def load_batch():
        batch = next(dataloader_iter)

        if static_batch_shape and not old_replay_sampling:
            batch = repeat_batch_to_size(batch, online_batch_size)

        if exists(rehearsal_iter):
            rehearsal_batch = next(rehearsal_iter)

            if static_batch_shape and not old_replay_sampling:
                rehearsal_batch = repeat_batch_to_size(rehearsal_batch, rehearsal_batch_size)

            batch = offline.concat_replay_batches(batch, rehearsal_batch)

        return batch

    policy_snapshot = None
    policy_context_exp = None
    policy_context_is_truncated = None
    policy_target_unembeds = None

    if preserve_policy_during_world_model_updates:
        if world_model_policy_consistency_loss_weight <= 0.:
            raise ValueError("world_model_policy_consistency_loss_weight must be positive")

        consistency_batch = next(dataloader_iter)
        if static_batch_shape and not old_replay_sampling:
            consistency_batch = repeat_batch_to_size(consistency_batch, batch_size)

        policy_context_exp = Experience.from_buffer_dict(consistency_batch)
        policy_context_exp.lens = consistency_batch.get("_lens")
        policy_context_exp = policy_context_exp.to(world_model.device)
        if exists(policy_context_exp.lens):
            policy_context_is_truncated = torch.zeros_like(
                policy_context_exp.lens,
                dtype = torch.bool,
            )
        policy_snapshot = FrozenPolicyPrior(world_model).to(world_model.device)

        with torch.no_grad():
            target_policy_inputs = policy_inputs_from_replay_context(
                world_model,
                policy_context_exp,
                step_size = policy_consistency_step_size,
            )
            policy_target_unembeds = Actions(*policy_snapshot.action_unembeds(target_policy_inputs))

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
                batch = load_batch()

            with record("world_model/data/prepare_experience", cuda = True):
                exp = replay_batch_to_experience(batch, world_model.device)

            def world_model_loss_on(rows: slice | None, update_ema: bool):
                def take(t):
                    return t if not exists(t) or rows is None else t[rows]

                # bf16 autocast covers only the world-model loss. Imagination and
                # rollouts stay in fp32, so the PPO log-prob ratios stay exact.
                with torch.autocast("cuda", dtype = torch.bfloat16, enabled = autocast_bf16 and world_model.device.type == "cuda"):
                    return world_model_loss_fn(
                        take(exp.latents),
                        take(exp.rewards),
                        take(exp.terminals),
                        take(exp.actions.continuous) if exists(exp.actions) else None,
                        take(exp.lens),
                        shortcut_train_mode,
                        update_ema,
                    )

            def add_policy_consistency(loss):
                policy_consistency_loss = None
                if preserve_policy_during_world_model_updates:
                    current_policy_inputs = policy_inputs_from_replay_context(
                        world_model,
                        policy_context_exp,
                        step_size = policy_consistency_step_size,
                    )
                    current_policy_unembeds = Actions(*policy_snapshot.action_unembeds(
                        current_policy_inputs,
                        detach_inputs = False,
                    ))
                    policy_consistency_loss = mean_action_distribution_kl(
                        policy_snapshot.action_embedder,
                        policy_target_unembeds,
                        current_policy_unembeds,
                        lens = policy_context_exp.lens,
                        is_truncated = policy_context_is_truncated,
                    )
                    loss = loss + (
                        policy_consistency_loss * world_model_policy_consistency_loss_weight
                    )
                return loss, policy_consistency_loss

            # fixed_full_transformer_finetuning: the RL loss of this step. It runs before the world-model loss. A loss
            # with a graph joins the world-model loss in one backward. In chunks, its gradient is already in place.
            extra_loss = None

            def forward_backward():
                nonlocal extra_loss
                if exists(extra_loss_fn):
                    extra_loss = extra_loss_fn(step, exp)

                if not exists(micro_batch_size):
                    with record("world_model/model/forward", cuda = True):
                        loss, losses = world_model_loss_on(None, update_loss_ema)
                        loss, policy_consistency_loss = add_policy_consistency(loss)
                        if exists(extra_loss):
                            loss = loss + extra_loss

                    with record("world_model/model/backward", cuda = True):
                        loss.backward()
                    return loss, losses, policy_consistency_loss

                # Each chunk loss is a batch mean, so a weight of chunk rows / batch
                # rows gives the full-batch gradient (exact for full windows). The
                # loss EMA updates once for each step, as in the one-pass path.
                total_rows = exp.latents.shape[0]
                loss = 0.
                for chunk_index, start in enumerate(range(0, total_rows, micro_batch_size)):
                    rows = slice(start, min(start + micro_batch_size, total_rows))
                    chunk_weight = (rows.stop - rows.start) / total_rows
                    with record("world_model/model/forward", cuda = True):
                        chunk_loss, losses = world_model_loss_on(rows, update_loss_ema and chunk_index == 0)
                    with record("world_model/model/backward", cuda = True):
                        (chunk_loss * chunk_weight).backward()
                    loss = loss + chunk_loss.detach() * chunk_weight
                    del chunk_loss

                with record("world_model/model/forward", cuda = True):
                    loss, policy_consistency_loss = add_policy_consistency(loss)
                    if exists(extra_loss):
                        loss = loss + extra_loss

                # The chunk losses already ran backward and are detached, so only a
                # policy-consistency term (or an extra loss with a graph) needs a backward here.
                if loss.requires_grad:
                    with record("world_model/model/backward", cuda = True):
                        loss.backward()
                return loss, losses, policy_consistency_loss

            loss, losses, policy_consistency_loss = record_timed_block(
                world_model_loss_timing,
                world_model.device,
                forward_backward,
            )

            with record("world_model/optimization/grad_clip", cuda = True):
                norm, grad_metrics = clip_grad_norm_by_group(grad_clip_groups, max_grad_norm)
            with record("world_model/optimization/optimizer_step", cuda = True):
                optimizer.step()
                optimizer.zero_grad()

            post_update_policy_kl = None
            if preserve_policy_during_world_model_updates:
                with torch.no_grad():
                    post_update_policy_inputs = policy_inputs_from_replay_context(
                        world_model,
                        policy_context_exp,
                        step_size = policy_consistency_step_size,
                    )
                    post_update_policy_unembeds = Actions(*policy_snapshot.action_unembeds(
                        post_update_policy_inputs,
                        detach_inputs = False,
                    ))
                    post_update_policy_kl = mean_action_distribution_kl(
                        policy_snapshot.action_embedder,
                        policy_target_unembeds,
                        post_update_policy_unembeds,
                        lens = policy_context_exp.lens,
                        is_truncated = policy_context_is_truncated,
                    )

            with record("world_model/metrics", cuda = True):
                last_metrics = {
                    **world_model_loss_metrics(loss, losses),
                    "world_model/grad_norm": float(norm),
                    "extra/flow_loss_raw": losses.diagnostics.raw_flow.item(),
                    "extra/shortcut_loss_raw": losses.diagnostics.raw_shortcut.item(),
                    "extra/dynamics_loss_raw": losses.diagnostics.raw_dynamics.item(),
                    "extra/dynamics_loss_normalized": losses.diagnostics.normalized_dynamics.item(),
                    "extra/reward_loss_raw": losses.diagnostics.raw_reward.item(),
                    "extra/terminal_loss_raw": losses.diagnostics.raw_terminal.item(),
                }
                if isinstance(dataset, FlipOnsetTimeWindow):
                    last_metrics["extra/flip_onsets_in_replay"] = len(dataset.onsets.rows)
                if exists(extra_loss):
                    # world_model/loss stays the world-model loss alone.
                    last_metrics["world_model/loss"] = (loss - extra_loss).item()
                    last_metrics["joint/total_loss"] = loss.item()
                if behavior_cloning:
                    # The BC term of world_model/loss (sum over the MTP distances), and the NLL per action
                    # dimension of the next action (the head that the policy samples). The inputs have the noise
                    # levels of world-model training, not the clean inputs of acting, so this is not the fit of
                    # the acting policy.
                    last_metrics["world_model/behavior_cloning_loss"] = losses.continuous_actions.sum().item()
                    last_metrics["world_model/behavior_cloning_nll_next_action"] = losses.continuous_actions.reshape(-1)[0].item()
                if exists(policy_consistency_loss):
                    last_metrics["world_model/policy_consistency_loss_kl"] = policy_consistency_loss.item()
                    last_metrics["world_model/policy_consistency_kl"] = post_update_policy_kl.item()
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

            del exp, loss, losses, grad_metrics, policy_consistency_loss, post_update_policy_kl, extra_loss

    if exists(training_profiler):
        training_profiler.flush_timings()

    return global_step, last_metrics


# fixed_full_transformer_finetuning (paper section 3.3 and its footnote on full-transformer finetuning).
# The paper's phase 2 sums the dynamics loss and the BC and reward loss in one update of the transformer. Here the RL
# loss takes the place of BC: each optimizer step sums the world-model loss on replay windows (dynamics, reward,
# terminal) and the RL loss on dreams (PMPO with the policy-prior KL, and the value loss). One optimizer owns the
# transformer and the heads. The older full_transformer_finetuning flag uses two optimizers and two phases.

def dream_policy_inputs(
    model: DynamicsWorldModel,
    full_dream: Experience,
    prompt_length: int,
    rows: slice | None = None,
):
    """Agent embeds and policy inputs of the generated frames, from one clean forward over prompt and dream.

    A rollout with the time cache reads clean prompt frames and clean generated frames. Thus this forward gives the
    inputs that the policy and value heads read in the rollout, if the rollout latents are not clamped.
    """

    def take(t):
        return t if not exists(t) or rows is None else t[rows]

    actions = action_pair_or_empty(full_dream.actions)
    _, (embeds, _) = model(
        latents = take(full_dream.latents),
        signal_levels = model.max_steps - 1,
        step_sizes = full_dream.step_size,
        rewards = take(full_dream.rewards),
        discrete_actions = take(actions.discrete),
        continuous_actions = take(actions.continuous),
        proprio = take(full_dream.proprio),
        latent_is_noised = True,
        return_pred_only = True,
        return_intermediates = True,
    )

    agent_index = full_dream.agent_index
    agent_embeds = embeds.agent[..., agent_index, :][:, prompt_length:]
    policy_inputs = model.select_policy_input(embeds, agent_index)[:, prompt_length:] if model.policy_all_tokens else None
    return agent_embeds, policy_inputs


def batch_chunks(batch_size: int, chunk_size: int | None) -> list[slice]:
    chunk_size = chunk_size if exists(chunk_size) and chunk_size < batch_size else batch_size
    return [slice(start, min(start + chunk_size, batch_size)) for start in range(0, batch_size, chunk_size)]


def experience_rows(experience: Experience, rows: slice) -> Experience:
    return Experience(**tree_map_tensor(lambda t: t[rows] if t.ndim > 0 else t, vars(experience)))


class FrozenModelPolicyPrior(nn.Module):
    """The policy at the start of a loop: a frozen copy of the transformer and the policy head.

    FrozenPolicyPrior keeps the policy head only, so its output moves when the transformer moves. With the
    transformer in the RL update, the prior must stay a fixed function of the dream.
    """

    def __init__(self, world_model: DynamicsWorldModel):
        super().__init__()
        self.model = deepcopy(world_model)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def refresh_from(self, world_model: DynamicsWorldModel):
        self.model.load_state_dict(world_model.state_dict())
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def action_unembeds(self, full_dream: Experience, prompt_length: int, chunk_size: int | None = None):
        parts = [
            dream_policy_inputs(self.model, full_dream, prompt_length, rows)
            for rows in batch_chunks(full_dream.latents.shape[0], chunk_size)
        ]
        inputs = torch.cat([default(policy_inputs, agent_embeds) for agent_embeds, policy_inputs in parts])
        policy_embed = self.model.policy_head(inputs)
        return Actions(*self.model.action_embedder.unembed(policy_embed, pred_head_index = 0))


def joint_finetune_clip_groups(world_model: DynamicsWorldModel, optimizer: Optimizer):
    """The world-model clip groups, with the value heads in their own group and not in the trunk."""

    value_params = set(world_model.value_head_parameters())
    groups = []

    for name, params in world_model_clip_groups(world_model, optimizer):
        if name != "trunk":
            groups.append((name, params))
            continue

        for split_name, split_params in (
            ("value_head", [param for param in params if param in value_params]),
            ("trunk", [param for param in params if param not in value_params]),
        ):
            if len(split_params) > 0:
                groups.append((split_name, split_params))

    return groups


@torch.no_grad()
def closer_ppo_dream_targets(world_model: DynamicsWorldModel, dream: Experience) -> dict[str, Tensor]:
    """The targets of closer CleanRL PPO for one dream batch: frozen GAE returns and normalized advantages.

    train_cleanrl_updated_ppo_dream normalizes the advantages of each minibatch of transitions. Here the batch is all
    valid transitions of the dreams, because the transformer forward needs whole dreams.
    """

    frozen = freeze_ppo_targets(world_model, Experience(**vars(dream)), normalize_advantages = False)
    batch, time = frozen.rewards.shape[:2]
    lens = default(frozen.lens, torch.full((batch,), time, device = frozen.rewards.device, dtype = torch.long))
    is_truncated = default(frozen.is_truncated, torch.ones(batch, device = frozen.rewards.device, dtype = torch.bool))
    learn_mask = lens_to_mask((lens - is_truncated.long()).clamp(min = 0), time)

    advantages = frozen.frozen_advantages
    valid = advantages[learn_mask]
    return dict(
        frozen_returns = frozen.frozen_returns,
        frozen_advantages = (advantages - valid.mean()) / (valid.std() + 1e-8),
    )


class JointImaginationLoss:
    """The RL loss of one fixed_full_transformer_finetuning step (the extra_loss_fn of train_world_model).

    Each call generates one dream batch, reads the heads on a forward with the transformer gradient, and gives the
    RL loss: PMPO with the policy-prior KL, or the closer CleanRL PPO loss (one pass over the batch), plus the value loss. Without chunks, the loss keeps its graph, and train_world_model runs one
    backward for the sum with the world-model loss. With chunks (16 GB GPU), the call accumulates the same gradient:
    one backward through the heads on the full dream batch, then one transformer backward for each chunk.
    """

    # Rows of the gradient diagnostic: enough for a stable norm, and small enough for a shared 16 GB GPU.
    DIAGNOSTIC_WINDOWS = 16
    DIAGNOSTIC_DREAMS = 64

    def __init__(
        self,
        world_model: DynamicsWorldModel,
        policy_prior: FrozenModelPolicyPrior | None,
        replay: ReplayBuffer,
        *,
        objective: Literal["pmpo", "ppo"],
        value_loss_weight: float,
        generate_rollout_fn: Callable,
        batch_size: int,
        horizon: int,
        prompt_length: int,
        static_generate_shape: bool,
        loss_weight: float,
        micro_batch_size: int | None,
        use_delight_gating: bool,
        grad_diagnostics: bool,
        autocast_bf16: bool,
        global_step: int,
        writer: SummaryWriter | None,
        training_profiler: TrainingProfiler | None,
    ):
        self.world_model = world_model
        self.policy_prior = policy_prior
        self.objective = objective
        self.value_loss_weight = value_loss_weight
        self.generate_rollout_fn = generate_rollout_fn
        self.batch_size = batch_size
        self.horizon = horizon
        self.prompt_length = prompt_length
        self.static_generate_shape = static_generate_shape
        self.loss_weight = loss_weight
        self.micro_batch_size = micro_batch_size
        self.use_delight_gating = use_delight_gating
        self.grad_diagnostics = grad_diagnostics
        self.autocast_bf16 = autocast_bf16
        self.global_step = global_step
        self.writer = writer
        self.training_profiler = training_profiler
        self.last_metrics = {}

        self.prompt_iterator = None
        if prompt_length > 0:
            prompt_dataset = ReplayDatasetTimeWindow(replay_buffer = replay, window_length = prompt_length)
            if len(prompt_dataset) == 0:
                raise ValueError("fixed_full_transformer_finetuning requires at least one prompt window")
            self.prompt_iterator = cycle(DataLoader(
                prompt_dataset,
                batch_size = batch_size,
                shuffle = True,
                drop_last = len(prompt_dataset) >= batch_size,
            ))

    def record(self, name: str):
        return self.training_profiler.record(name, cuda = True) if exists(self.training_profiler) else nullcontext()

    @torch.no_grad()
    def generate(self):
        device = self.world_model.device
        prompts = {}
        batch_size = self.batch_size

        if exists(self.prompt_iterator):
            with self.record("imagination/data/load_prompt"):
                prompt_batch = next(self.prompt_iterator)
                prompt_batch = tree_map(lambda t: t.to(device) if torch.is_tensor(t) else t, prompt_batch)

                if self.static_generate_shape:
                    prompt_batch = repeat_batch_to_size(prompt_batch, self.batch_size)

                rewards = prompt_batch.get("rewards")
                prompts = dict(
                    prompt_latents = prompt_batch.get("latents"),
                    prompt_proprio = prompt_batch.get("proprio"),
                    prompt_discrete_actions = prompt_batch.get("actions_discrete"),
                    prompt_continuous_actions = prompt_batch.get("actions_continuous"),
                    prompt_rewards = rewards[:, :-1] if exists(rewards) else None,
                )
                batch_size = prompts["prompt_latents"].shape[0]

        if exists(getattr(self.generate_rollout_fn, "timing", None)) and self.generate_rollout_fn.timing.cuda_graphs:
            cudagraph_mark_step_begin()

        with self.record("imagination/model/generate"):
            full_dream = self.generate_rollout_fn(
                self.horizon + (self.prompt_length if exists(self.prompt_iterator) else 0),
                batch_size,
                prompts.get("prompt_latents"),
                prompts.get("prompt_proprio"),
                prompts.get("prompt_discrete_actions"),
                prompts.get("prompt_continuous_actions"),
                prompts.get("prompt_rewards"),
            )

        dream = trim_prompt_from_dream(full_dream, self.prompt_length, self.horizon)

        if exists(self.policy_prior):
            with self.record("imagination/model/policy_prior"):
                dream.old_action_unembeds = self.policy_prior.action_unembeds(full_dream, self.prompt_length, self.micro_batch_size)

        return full_dream, dream

    def rl_losses(self, dream: Experience, agent_embeds: Tensor, policy_inputs: Tensor | None):
        # The values and rewards of the rollout give the returns and advantages. Only the heads run here.
        fields = {**vars(dream), "agent_embed": agent_embeds, "policy_input": policy_inputs}
        if self.objective == "ppo":
            fields.update(closer_ppo_dream_targets(self.world_model, dream))

        return self.world_model.learn_from_experience(
            Experience(**fields),
            only_learn_policy_value_heads = False,
            use_given_agent_embeds = True,
            objective = self.objective,
            use_delight_gating = self.use_delight_gating,
            return_diagnostics = True,
        )

    def __call__(self, step: int, replay_experience: Experience) -> Tensor:
        world_model = self.world_model
        full_dream, dream = self.generate()
        chunks = batch_chunks(full_dream.latents.shape[0], self.micro_batch_size)

        if self.grad_diagnostics and step == 0:
            with self.record("imagination/grad_diagnostics"):
                log_scalars(self.writer, self.trunk_gradient_metrics(replay_experience, full_dream, dream), self.global_step)

        with self.record("imagination/model/learn_forward"):
            if len(chunks) == 1:
                agent_embeds, policy_inputs = dream_policy_inputs(world_model, full_dream, self.prompt_length)
                policy_loss, value_loss, diagnostics = self.rl_losses(dream, agent_embeds, policy_inputs)
                loss = (policy_loss + self.value_loss_weight * value_loss) * self.loss_weight
            else:
                with torch.no_grad():
                    parts = [dream_policy_inputs(world_model, full_dream, self.prompt_length, rows) for rows in chunks]

                # Leaf copies of the head inputs. The head backward gives the gradient of the loss at each input.
                agent_embeds = torch.cat([agent for agent, _ in parts]).requires_grad_()
                policy_inputs = torch.cat([policy for _, policy in parts]).requires_grad_() if exists(parts[0][1]) else None
                policy_loss, value_loss, diagnostics = self.rl_losses(dream, agent_embeds, policy_inputs)
                loss = (policy_loss + self.value_loss_weight * value_loss) * self.loss_weight

        if len(chunks) > 1:
            with self.record("imagination/model/backward"):
                loss.backward()

                # The transformer gradient: the same forward in chunks, with the input gradients of the full batch.
                for rows in chunks:
                    chunk_agent, chunk_policy = dream_policy_inputs(world_model, full_dream, self.prompt_length, rows)
                    outputs, gradients = [], []
                    for output, leaf in ((chunk_agent, agent_embeds), (chunk_policy, policy_inputs)):
                        if exists(leaf) and exists(leaf.grad):
                            outputs.append(output)
                            gradients.append(leaf.grad[rows])
                    torch.autograd.backward(outputs, gradients)

            loss = loss.detach()

        with torch.no_grad(), self.record("imagination/metrics"):
            current_inputs = default(policy_inputs, agent_embeds).detach()
            rollout_inputs = default(dream.policy_input, dream.agent_embed)
            prior_metrics = {}
            if exists(self.policy_prior):
                current_unembeds = policy_unembeds_from_inputs(world_model, current_inputs)
                kl_pair = (current_unembeds, dream.old_action_unembeds)
                if world_model.pmpo_reverse_kl:
                    kl_pair = kl_pair[::-1]

                # The KL before the update of this step. It is 0 at the first step of a loop.
                prior_metrics["imagination/prior_kl_mean"] = mean_action_distribution_kl(
                    world_model.action_embedder, *kl_pair, lens = dream.lens, is_truncated = dream.is_truncated,
                ).item()

            metrics = {
                "imagination/loss": loss.item(),
                "imagination/policy_loss": policy_loss.item(),
                "imagination/value_loss": value_loss.item(),
                "imagination/dream_return": dream.episode_return.mean().item() if exists(dream.episode_return) else 0.,
                "imagination/dream_length": dream.lens.float().mean().item() if exists(dream.lens) else self.horizon,
                "imagination/action_std": dream.actions.continuous.std().item() if exists(dream.actions) and exists(dream.actions.continuous) else 0.,
                "imagination/prompt_length": self.prompt_length if exists(self.prompt_iterator) else 0,
                **prior_metrics,
                # The largest difference between the head inputs of the learner and of the rollout.
                "joint/learner_input_error": (current_inputs - rollout_inputs).abs().max().item(),
                **{key: value.item() for key, value in diagnostics.items()},
            }
            log_scalars(self.writer, metrics, self.global_step)

        self.last_metrics = metrics
        self.global_step += 1
        return loss

    def trunk_gradient_metrics(self, replay_experience: Experience, full_dream: Experience, dream: Experience):
        """Transformer gradient norm of each loss term, and the cosine between the RL and world-model gradients.

        A diagnostic only: it uses a part of each batch, and it leaves the parameter gradients and the RNG unchanged.
        """

        world_model = self.world_model
        trunk = [param for param in world_model.transformer.parameters() if param.requires_grad]

        # One forward for each term: the compiled attention kernels do not permit a second backward through a graph.
        def trunk_gradient(loss: Tensor):
            gradients = torch.autograd.grad(loss, trunk, allow_unused = True)
            return torch.cat([
                (gradient if exists(gradient) else torch.zeros_like(param)).flatten().float()
                for gradient, param in zip(gradients, trunk)
            ])

        replay_part = experience_rows(replay_experience, slice(0, self.DIAGNOSTIC_WINDOWS))

        def world_model_term(select: Callable):
            # Each call starts from the same RNG state, so both terms use the same noise.
            with isolated_rng(world_model.device):
                with torch.autocast("cuda", dtype = torch.bfloat16, enabled = self.autocast_bf16 and world_model.device.type == "cuda"):
                    _, losses = WorldModelTrainingLoss(world_model)(
                        replay_part.latents,
                        replay_part.rewards,
                        replay_part.terminals,
                        replay_part.actions.continuous if exists(replay_part.actions) else None,
                        replay_part.lens,
                        "combined",
                        False,
                    )
                return trunk_gradient(select(losses))

        dynamics = world_model_term(lambda losses: (
            losses.flow * world_model.latent_flow_loss_weight + losses.shortcut * world_model.shortcut_loss_weight
        ))
        reward = world_model_term(lambda losses: (
            (losses.agent_embed_reward * world_model.reward_loss_weight).sum()
            + (losses.latent_state_reward * world_model.reward_loss_weight).sum()
        ))

        rows = slice(0, self.DIAGNOSTIC_DREAMS)
        dream_part = experience_rows(dream, rows)

        def rl_term(index: int):
            agent_embeds, policy_inputs = dream_policy_inputs(world_model, full_dream, self.prompt_length, rows)
            weight = self.loss_weight * (self.value_loss_weight if index == 1 else 1.)
            return trunk_gradient(self.rl_losses(dream_part, agent_embeds, policy_inputs)[index] * weight)

        policy, value = rl_term(0), rl_term(1)

        def cosine(a: Tensor, b: Tensor):
            return F.cosine_similarity(a, b, dim = 0, eps = 1e-12).item()

        return {
            "joint_diag/trunk_grad_norm_dynamics": dynamics.norm().item(),
            "joint_diag/trunk_grad_norm_reward": reward.norm().item(),
            "joint_diag/trunk_grad_norm_policy": policy.norm().item(),
            "joint_diag/trunk_grad_norm_value": value.norm().item(),
            "joint_diag/trunk_grad_norm_world_model": (dynamics + reward).norm().item(),
            "joint_diag/trunk_grad_norm_rl": (policy + value).norm().item(),
            "joint_diag/trunk_grad_cos_rl_world_model": cosine(policy + value, dynamics + reward),
            "joint_diag/trunk_grad_cos_policy_dynamics": cosine(policy, dynamics),
            "joint_diag/trunk_grad_cos_value_dynamics": cosine(value, dynamics),
        }


def call_learn_from_dream(
    learn_from_dream_fn: Callable,
    experience: Experience,
):
    actions = action_pair_or_empty(experience.actions)
    log_probs = action_pair_or_empty(experience.log_probs)
    old_action_unembeds = action_pair_or_empty(experience.old_action_unembeds)
    return learn_from_dream_fn(
        experience.latents,
        experience.proprio,
        experience.critic_state,
        experience.agent_embed,
        experience.policy_input,
        experience.rewards,
        experience.terminals,
        actions.discrete,
        actions.continuous,
        log_probs.discrete,
        log_probs.continuous,
        old_action_unembeds.discrete,
        old_action_unembeds.continuous,
        experience.values,
        experience.step_size,
        experience.lens,
        experience.is_truncated,
        experience.is_from_world_model,
        experience.episode_return,
        None,
        None,
        None,
        None,
        None,
        None,
        experience.frozen_returns,
        experience.frozen_advantages,
    )


CLEANRL_UPDATED_PPO_EPOCHS = 10
CLEANRL_UPDATED_PPO_NUM_MINIBATCHES = 32
CLEANRL_UPDATED_PPO_VALUE_LOSS_WEIGHT = .5
CLEANRL_UPDATED_PPO_MAX_GRAD_NORM = .5
CLEANRL_PPO_CLOSER_DREAM_CHUNKS = 4


def train_cleanrl_updated_ppo_dream(
    world_model: DynamicsWorldModel,
    optimizer: Optimizer,
    dream: Experience,
    *,
    learn_from_dream_fn: Callable,
    learn_from_dream_timing: CallTiming | None,
    initial_learning_rate: float,
    iteration: int,
    total_iterations: int,
    global_step: int,
    writer: SummaryWriter | None,
    record: Callable,
    cleanrl_ppo_closer: bool = False,
    shuffle_generator: torch.Generator | None = None,
    metric_step: int | None = None,
    constant_learning_rate: bool = False,
    epochs: int = CLEANRL_UPDATED_PPO_EPOCHS,
    num_minibatches: int = CLEANRL_UPDATED_PPO_NUM_MINIBATCHES,
    value_loss_weight: float = CLEANRL_UPDATED_PPO_VALUE_LOSS_WEIGHT,
    max_grad_norm: float = CLEANRL_UPDATED_PPO_MAX_GRAD_NORM,
):
    """Apply updated CleanRL PPO to one flattened main-model dream batch."""

    if epochs <= 0 or num_minibatches <= 0:
        raise ValueError("CleanRL updated PPO epochs and minibatches must be positive")
    if total_iterations <= 0 or not 0 <= iteration < total_iterations:
        raise ValueError("CleanRL updated PPO iteration values are invalid")

    transitions = flatten_cleanrl_updated_ppo_transitions(world_model, dream)
    batch_size = transitions.latents.shape[0]
    minibatch_size = batch_size // num_minibatches
    if minibatch_size == 0:
        raise ValueError("CleanRL updated PPO has fewer transitions than minibatches")
    if minibatch_size < 2 or batch_size % minibatch_size == 1:
        raise ValueError("CleanRL updated PPO advantage normalization needs two samples per minibatch")

    learning_rate = initial_learning_rate if constant_learning_rate else (1. - iteration / total_iterations) * initial_learning_rate
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    params = optimizer_parameters(optimizer)
    dream_return = dream.episode_return.mean().item()
    dream_length = dream.lens.float().mean().item()
    action_values = safe_cat(action_pair_or_empty(dream.actions), dim = -1)
    action_std = action_values.float().std().item()
    last_metrics = {}
    last_agent_metrics = {}
    closer_clip_fractions = []

    explained_variance = float("nan")
    if cleanrl_ppo_closer:
        flat_returns = transitions.frozen_returns.flatten()
        flat_old_values = transitions.values.flatten()
        target_variance = torch.var(flat_returns, correction = 0)
        if target_variance != 0.:
            explained_variance = 1. - torch.var(
                flat_returns - flat_old_values,
                correction = 0,
            ).div(target_variance).item()

    for epoch in range(epochs):
        if cleanrl_ppo_closer:
            permutation = torch.randperm(
                batch_size,
                device = transitions.latents.device,
                generator = shuffle_generator,
            )
        else:
            permutation = torch.randperm(
                batch_size,
                device = transitions.latents.device,
            )
        for minibatch_index, start in enumerate(range(0, batch_size, minibatch_size)):
            indices = permutation[start:start + minibatch_size]
            minibatch = select_experience_batch(transitions, indices)
            advantages = minibatch.frozen_advantages
            minibatch.frozen_advantages = (
                (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            )

            def forward_backward():
                with record("imagination/model/learn_forward", cuda = True):
                    policy_loss, value_loss, value_diagnostics = call_learn_from_dream(
                        learn_from_dream_fn,
                        minibatch,
                    )
                    loss = policy_loss + value_loss_weight * value_loss

                optimizer.zero_grad(set_to_none = True)
                with record("imagination/model/backward", cuda = True):
                    loss.backward()
                return policy_loss, value_loss, value_diagnostics, loss

            policy_loss, value_loss, value_diagnostics, loss = record_timed_block(
                learn_from_dream_timing,
                world_model.device,
                forward_backward,
            )

            with record("imagination/optimization/grad_clip", cuda = True):
                grad_norm = float(clip_grad_norm_(params, max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                agent_metrics = agent_approx_kl_metrics(
                    world_model,
                    minibatch,
                    include_cleanrl_metrics = cleanrl_ppo_closer,
                )

            imagination_metrics = {
                "imagination/loss": loss.item(),
                "imagination/policy_loss": policy_loss.detach().item(),
                "imagination/value_loss": value_loss.item(),
                "imagination/dream_return": dream_return,
                "imagination/dream_length": dream_length,
                "imagination/action_std": action_std,
                "imagination/grad_norm": grad_norm,
                "imagination/approx_kl_mean": agent_metrics["approx_kl_mean"].item(),
                "imagination/approx_kl_min": agent_metrics["approx_kl_min"].item(),
                "imagination/approx_kl_max": agent_metrics["approx_kl_max"].item(),
                "imagination/cleanrl_updated_ppo_epoch": epoch,
                "imagination/cleanrl_updated_ppo_minibatch": minibatch_index,
                "imagination/cleanrl_updated_ppo_learning_rate": learning_rate,
            }
            imagination_metrics.update({key: value.item() for key, value in value_diagnostics.items()})
            last_metrics = imagination_metrics
            last_agent_metrics = agent_metrics
            if cleanrl_ppo_closer:
                closer_clip_fractions.append(agent_metrics["clip_fraction"])

            if not cleanrl_ppo_closer:
                log_scalars(writer, last_metrics, global_step)
            global_step += 1

            del minibatch, policy_loss, value_loss, value_diagnostics, loss, agent_metrics

    if cleanrl_ppo_closer:
        # The closer mode does not log each minibatch. Log the metrics of the last minibatch once per update,
        # so the imagination/, critic/, and extra/ graphs of the other objectives also exist for this mode.
        log_scalars(writer, last_metrics, global_step - 1)

        entropy = last_metrics["extra/policy_entropy"]
        action_range = world_model.action_embedder.continuous_target_action_range
        if action_range is not None:
            low, high = action_range
            entropy += world_model.action_embedder.num_continuous_action_types * np.log(high - low)

        closer_metrics = {
            "charts/learning_rate": learning_rate,
            "cleanrl_updated_ppo/loss": last_metrics["imagination/loss"],
            "cleanrl_updated_ppo/policy_loss": last_metrics["imagination/policy_loss"],
            "cleanrl_updated_ppo/value_loss": last_metrics["imagination/value_loss"],
            "cleanrl_updated_ppo/entropy": entropy,
            "cleanrl_updated_ppo/old_approx_kl": last_agent_metrics["old_approx_kl"].item(),
            "cleanrl_updated_ppo/approx_kl": last_agent_metrics["approx_kl_mean"].item(),
            "cleanrl_updated_ppo/clip_fraction": torch.stack(closer_clip_fractions).mean().item(),
            "cleanrl_updated_ppo/explained_variance": explained_variance,
            "cleanrl_updated_ppo/grad_norm": last_metrics["imagination/grad_norm"],
        }
        log_scalars(
            writer,
            closer_metrics,
            global_step if metric_step is None else metric_step,
        )
        last_metrics.update(closer_metrics)

    return global_step, last_metrics


def train_reused_ppo_dream(
    world_model: DynamicsWorldModel,
    optimizer: Optimizer,
    dream: Experience,
    *,
    learn_from_dream_fn: Callable,
    learn_from_dream_timing: CallTiming | None,
    max_grad_norm: float,
    epochs: int,
    num_minibatches: int,
    max_exact_kl: float,
    lr_backoff: float,
    max_backoffs: int,
    global_step: int,
    writer: SummaryWriter | None,
    record: Callable,
):
    if epochs <= 0 or num_minibatches <= 0:
        raise ValueError("PPO reuse epochs and minibatches must be positive")
    if dream.latents.shape[0] < num_minibatches:
        raise ValueError("PPO reuse needs at least one environment per minibatch")
    if not 0. < lr_backoff < 1.:
        raise ValueError("ppo_reuse_lr_backoff must be between 0 and 1")
    if max_exact_kl <= 0.:
        raise ValueError("ppo_reuse_max_exact_kl must be positive")

    dream = freeze_ppo_targets(world_model, dream)
    policy_params = ppo_optimizer_role_parameters(optimizer, "policy")
    value_params = ppo_optimizer_role_parameters(optimizer, "value")
    if not policy_params or not value_params:
        raise ValueError("PPO reuse requires policy and value optimizer groups")

    grad_clip_groups = agent_clip_groups(world_model, optimizer)
    actor_updates_stopped = False
    accepted_actor_updates = 0
    completed_value_updates = 0
    last_metrics = {}
    full_policy_inputs = (
        dream.policy_input
        if world_model.policy_all_tokens and exists(dream.policy_input)
        else dream.agent_embed
    )
    full_behavior_unembeds = action_pair_or_empty(dream.old_action_unembeds)

    @torch.no_grad()
    def full_dream_exact_kl():
        return exact_policy_kl(
            world_model,
            full_policy_inputs,
            full_behavior_unembeds,
            lens = dream.lens,
            is_truncated = dream.is_truncated,
        )

    for epoch in range(epochs):
        permutation = torch.randperm(dream.latents.shape[0], device = dream.latents.device)
        minibatch_indices = permutation.tensor_split(num_minibatches)

        for minibatch_index, indices in enumerate(minibatch_indices):
            minibatch = select_experience_batch(dream, indices)

            def forward_backward():
                with record("imagination/model/learn_forward", cuda = True):
                    policy_loss, value_loss, value_diagnostics = call_learn_from_dream(
                        learn_from_dream_fn,
                        minibatch,
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

            policy_gradients = [
                param.grad.detach().clone() if exists(param.grad) else None
                for param in policy_params
            ]
            policy_snapshot = snapshot_optimizer_parameters(optimizer, policy_params)
            accepted_actor_update = False
            backoff_count = 0

            if actor_updates_stopped:
                for param in policy_params:
                    param.grad = None
                optimizer.step()
            else:
                optimizer.step()
                exact_kl = full_dream_exact_kl()

                while exact_kl.item() > max_exact_kl and backoff_count < max_backoffs:
                    restore_optimizer_parameters(optimizer, policy_params, policy_snapshot)
                    backoff_count += 1
                    for group in optimizer.param_groups:
                        if group.get("ppo_role") == "policy":
                            group["lr"] *= lr_backoff

                    for param in value_params:
                        param.grad = None
                    for param, gradient in zip(policy_params, policy_gradients):
                        param.grad = gradient

                    optimizer.step()
                    exact_kl = full_dream_exact_kl()

                if exact_kl.item() <= max_exact_kl:
                    accepted_actor_update = True
                    accepted_actor_updates += 1
                else:
                    restore_optimizer_parameters(optimizer, policy_params, policy_snapshot)
                    actor_updates_stopped = True
                    exact_kl = full_dream_exact_kl()

            completed_value_updates += 1
            optimizer.zero_grad()

            with torch.no_grad():
                agent_metrics = agent_approx_kl_metrics(world_model, minibatch)
                if actor_updates_stopped:
                    exact_kl = full_dream_exact_kl()

            policy_lr = next(
                group["lr"] for group in optimizer.param_groups
                if group.get("ppo_role") == "policy"
            )
            last_metrics = {
                "imagination/loss": loss.item(),
                "imagination/policy_loss": policy_loss.detach().item(),
                "imagination/value_loss": value_loss.item(),
                "imagination/dream_return": dream.episode_return.mean().item(),
                "imagination/dream_length": dream.lens.float().mean().item(),
                "imagination/action_std": dream.actions.continuous.std().item(),
                "imagination/grad_norm": float(norm),
                "imagination/approx_kl_mean": agent_metrics["approx_kl_mean"].item(),
                "imagination/approx_kl_min": agent_metrics["approx_kl_min"].item(),
                "imagination/approx_kl_max": agent_metrics["approx_kl_max"].item(),
                "imagination/exact_policy_kl": exact_kl.item(),
                "imagination/ppo_reuse_epoch": epoch,
                "imagination/ppo_reuse_minibatch": minibatch_index,
                "imagination/ppo_actor_update_accepted": float(accepted_actor_update),
                "imagination/ppo_actor_updates": accepted_actor_updates,
                "imagination/ppo_value_updates": completed_value_updates,
                "imagination/ppo_lr_backoffs": backoff_count,
                "imagination/ppo_policy_learning_rate": policy_lr,
            }
            last_metrics.update({key: value.item() for key, value in value_diagnostics.items()})
            last_metrics.update({f"imagination/{key}": value for key, value in grad_metrics.items()})
            log_scalars(writer, last_metrics, global_step)
            global_step += 1

            del minibatch, policy_loss, value_loss, value_diagnostics, loss
            del agent_metrics, grad_metrics, policy_gradients, policy_snapshot

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
    objective: Literal["ppo", "pmpo", "spo", "reinforce"],
    use_delight_gating: bool,
    global_step: int,
    writer: SummaryWriter | None,
    training_profiler: TrainingProfiler | None,
    old_replay_sampling: bool = False,
    full_transformer_finetuning: bool = False,
    preservation_sequence_length: int | None = None,
    preservation_batch_size: int | None = None,
    align_generated_dream_transitions: bool = False,
    cleanrl_updated_ppo: bool = False,
    cleanrl_ppo_closer: bool = False,
    cleanrl_updated_ppo_iteration_offset: int = 0,
    cleanrl_updated_ppo_total_iterations: int | None = None,
    cleanrl_updated_ppo_initial_learning_rate: float | None = None,
    cleanrl_updated_ppo_constant_learning_rate: bool = False,
    cleanrl_updated_ppo_shuffle_generator: torch.Generator | None = None,
    cleanrl_updated_ppo_metric_step: int | None = None,
    reuse_ppo_dream_batch: bool = False,
    ppo_reuse_epochs: int = 4,
    ppo_reuse_num_minibatches: int = 4,
    ppo_reuse_max_exact_kl: float = 0.02,
    ppo_reuse_lr_backoff: float = 0.5,
    ppo_reuse_max_backoffs: int = 6,
    flip_onset_prompt_fraction: float = 0.,
):
    if steps <= 0:
        return global_step, {}

    last_metrics = {}
    pbar = tqdm(range(steps), desc = "imagination", leave = False)
    grad_clip_groups = agent_clip_groups(world_model, optimizer)

    if reuse_ppo_dream_batch and objective != "ppo":
        raise ValueError("reuse_ppo_dream_batch requires objective='ppo'")
    if reuse_ppo_dream_batch and full_transformer_finetuning:
        raise ValueError("reuse_ppo_dream_batch does not support full_transformer_finetuning")
    if cleanrl_updated_ppo and objective != "ppo":
        raise ValueError("cleanrl_updated_ppo requires objective='ppo'")
    if cleanrl_updated_ppo and full_transformer_finetuning:
        raise ValueError("cleanrl_updated_ppo does not support full_transformer_finetuning")
    if cleanrl_updated_ppo and reuse_ppo_dream_batch:
        raise ValueError("cleanrl_updated_ppo and reuse_ppo_dream_batch are mutually exclusive")
    if cleanrl_ppo_closer and not cleanrl_updated_ppo:
        raise ValueError("cleanrl_ppo_closer requires cleanrl_updated_ppo=true")

    if cleanrl_updated_ppo:
        cleanrl_updated_ppo_total_iterations = (
            steps
            if cleanrl_updated_ppo_total_iterations is None
            else cleanrl_updated_ppo_total_iterations
        )
        cleanrl_updated_ppo_initial_learning_rate = (
            optimizer.param_groups[0]["lr"]
            if cleanrl_updated_ppo_initial_learning_rate is None
            else cleanrl_updated_ppo_initial_learning_rate
        )

    # Flip-onset prompts end upright, 1 to `horizon` rows before a real onset, so the real flip falls in the dream steps.
    prompt_dataset = replay_time_windows(
        replay,
        window_length = prompt_length,
        flip_onset_fraction = flip_onset_prompt_fraction if prompt_length > 0 else 0.,
        min_before = prompt_length,
        max_before = prompt_length + horizon - 1,
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

    def generate_cleanrl_ppo_closer_chunk(record):
        use_prompt = prompt_length > 0 and (static_generate_shape or random.random() < prompt_probability)
        prompts = None
        prompt_batch = None

        if use_prompt and exists(prompt_iterator):
            with record("imagination/data/load_prompt", cuda = True):
                prompt_batch = next(prompt_iterator)
                prompt_batch = tree_map(
                    lambda t: t.to(world_model.device) if torch.is_tensor(t) else t,
                    prompt_batch,
                )

                if static_generate_shape:
                    prompt_batch = repeat_batch_to_size(prompt_batch, batch_size)

                prompts = dict(
                    prompt_latents = prompt_batch.get("latents"),
                    prompt_proprio = prompt_batch.get("proprio"),
                    prompt_discrete_actions = prompt_batch.get("actions_discrete"),
                    prompt_continuous_actions = prompt_batch.get("actions_continuous"),
                    prompt_rewards = prompt_batch.get("rewards")[:, :-1] if exists(prompt_batch.get("rewards")) else None,
                )

        generation_horizon = horizon + prompt_length if exists(prompts) else horizon
        actual_batch_size = prompt_batch.get("latents").shape[0] if exists(prompts) else batch_size
        has_prompt = exists(prompts)
        prompts = prompts or {}

        if exists(getattr(generate_rollout_fn, "timing", None)) and generate_rollout_fn.timing.cuda_graphs:
            cudagraph_mark_step_begin()

        with record("imagination/model/generate", cuda = True):
            chunk = generate_rollout_fn(
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
                chunk = trim_prompt_from_dream(chunk, prompt_length, horizon)

        if align_generated_dream_transitions:
            chunk = align_dream_transition_rewards(chunk)

        return chunk

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

    for imagination_iteration in pbar:
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

            if align_generated_dream_transitions:
                dream = align_dream_transition_rewards(dream)

            if cleanrl_ppo_closer:
                dream_chunks = [dream]
                for _ in range(CLEANRL_PPO_CLOSER_DREAM_CHUNKS - 1):
                    dream_chunks.append(generate_cleanrl_ppo_closer_chunk(record))
                dream = combine_experiences(dream_chunks)

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

        if cleanrl_updated_ppo:
            global_step, last_metrics = train_cleanrl_updated_ppo_dream(
                world_model,
                optimizer,
                dream,
                learn_from_dream_fn = learn_from_dream_fn,
                learn_from_dream_timing = learn_from_dream_timing,
                initial_learning_rate = cleanrl_updated_ppo_initial_learning_rate,
                iteration = cleanrl_updated_ppo_iteration_offset + imagination_iteration,
                total_iterations = cleanrl_updated_ppo_total_iterations,
                global_step = global_step,
                writer = writer,
                record = record,
                cleanrl_ppo_closer = cleanrl_ppo_closer,
                shuffle_generator = cleanrl_updated_ppo_shuffle_generator,
                metric_step = cleanrl_updated_ppo_metric_step,
                constant_learning_rate = cleanrl_updated_ppo_constant_learning_rate,
            )
            pbar.set_postfix(
                dream_return = f"{last_metrics['imagination/dream_return']:.1f}",
                loss = f"{last_metrics['imagination/loss']:.3f}",
            )

            if exists(imagination_step_timing):
                synchronize_if_cuda(world_model.device)
                imagination_step_timing.record(perf_counter() - step_start)

            if exists(profiler):
                profiler.advance()

            del dream
            continue

        if reuse_ppo_dream_batch:
            global_step, last_metrics = train_reused_ppo_dream(
                world_model,
                optimizer,
                dream,
                learn_from_dream_fn = learn_from_dream_fn,
                learn_from_dream_timing = learn_from_dream_timing,
                max_grad_norm = max_grad_norm,
                epochs = ppo_reuse_epochs,
                num_minibatches = ppo_reuse_num_minibatches,
                max_exact_kl = ppo_reuse_max_exact_kl,
                lr_backoff = ppo_reuse_lr_backoff,
                max_backoffs = ppo_reuse_max_backoffs,
                global_step = global_step,
                writer = writer,
                record = record,
            )
            pbar.set_postfix(
                dream_return = f"{last_metrics['imagination/dream_return']:.1f}",
                loss = f"{last_metrics['imagination/loss']:.3f}",
            )

            if exists(imagination_step_timing):
                synchronize_if_cuda(world_model.device)
                imagination_step_timing.record(perf_counter() - step_start)

            if exists(profiler):
                profiler.advance()

            del dream
            continue

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
                    dream.frozen_returns,
                    dream.frozen_advantages,
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
    agent_optimizer: Optimizer | None,
    progress: dict[str, int | float] | None = None,
    optimizer_settings: dict[str, list[dict]] | None = None,
):
    path.parent.mkdir(parents = True, exist_ok = True)
    temp_path = path.with_name(f"{path.name}.tmp")
    torch.save(
        dict(
            loop = loop,
            tokenizer = tokenizer.state_dict(),
            world_model = world_model.state_dict(),
            world_optimizer = optimizer_state_dict_for_checkpoint(world_optimizer),
            # None with fixed_full_transformer_finetuning: world_optimizer holds all parameters.
            agent_optimizer = optimizer_state_dict_for_checkpoint(agent_optimizer) if exists(agent_optimizer) else None,
            progress = progress,
            optimizer_settings = optimizer_settings,
        ),
        str(temp_path),
    )
    os.replace(temp_path, path)


REPLAY_SNAPSHOT_PREFIX = "replay_"


def replay_snapshot_path(checkpoint_path: Path) -> Path:
    """Return the replay snapshot folder of a loop checkpoint, for example replay_loop_25."""
    return checkpoint_path.with_name(f"{REPLAY_SNAPSHOT_PREFIX}{checkpoint_path.stem}")


def sparse_copy(source, destination):
    """Copy one file and keep its zero blocks as holes.

    The replay memmaps hold the full capacity, and the part without episodes is zeros. A plain copy writes these
    zeros to disk (9.9 GB against 3.9 GB for a run of 100 loops). The content of the copy is the same.
    """
    try:
        subprocess.run(["cp", "--sparse=always", "--preserve=mode,timestamps", str(source), str(destination)], check = True)
    except (OSError, subprocess.CalledProcessError) as error:
        warnings.warn(f"sparse copy of {source} failed ({error}); a plain copy writes the zero blocks to disk", RuntimeWarning, stacklevel = 2)
        shutil.copy2(source, destination)
    return destination


def save_resume_checkpoint(path: Path, *, replay: ReplayBuffer, **checkpoint):
    """Save a loop checkpoint and a copy of the replay at the same loop.

    The replay folder of the run receives each rollout. A resume discards it and
    copies the snapshot back, so the model and the replay agree. Only the newest
    snapshot is kept, because each snapshot is a full copy of the replay.
    """
    snapshot = replay_snapshot_path(path)
    temp_snapshot = snapshot.with_name(f"{snapshot.name}.tmp")

    replay.flush()
    shutil.rmtree(temp_snapshot, ignore_errors = True)
    shutil.copytree(replay.folder, temp_snapshot, copy_function = sparse_copy)
    shutil.rmtree(snapshot, ignore_errors = True)
    temp_snapshot.rename(snapshot)

    # Save the checkpoint after its snapshot is complete, and remove older
    # snapshots last. A crash at any point keeps one checkpoint that can resume.
    save_checkpoint(path, **checkpoint)

    for old_snapshot in path.parent.glob(f"{REPLAY_SNAPSHOT_PREFIX}loop_*"):
        if old_snapshot != snapshot and old_snapshot.is_dir():
            shutil.rmtree(old_snapshot)


def restore_replay_snapshot(checkpoint_path: Path, replay_folder: Path) -> ReplayBuffer:
    snapshot = replay_snapshot_path(checkpoint_path)
    if not snapshot.is_dir():
        raise FileNotFoundError(
            f"{checkpoint_path} has no replay snapshot at {snapshot}. Only the newest "
            "checkpoint keeps a snapshot, and checkpoints saved before snapshots existed "
            "cannot resume."
        )

    # A missing field file does not fail ReplayBuffer.from_folder: it makes a new file of zeros. The world model
    # would then train on zero latents for every old episode. (The latents of finished runs can be deleted to free
    # disk space; such a run cannot resume.)
    with open(snapshot / "metadata.pkl", "rb") as file:
        fields = pickle.load(file)["fields"]
    missing = [f"{name}.data.npy" for name in fields if not (snapshot / f"{name}.data.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"the replay snapshot {snapshot} has no {', '.join(missing)}; the run cannot resume")

    shutil.rmtree(replay_folder, ignore_errors = True)
    shutil.copytree(snapshot, replay_folder, copy_function = sparse_copy)
    return ReplayBuffer.from_folder(replay_folder)


# These keys select a kernel, not the update, so a resume on another device can change them.
OPTIMIZER_IMPLEMENTATION_KEYS = frozenset({"fused", "foreach", "capturable", "differentiable"})


def optimizer_group_settings(optimizer: Optimizer) -> list[dict]:
    """Return the hyperparameters of each parameter group, such as lr and weight_decay."""
    return [
        {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in sorted(group.items())
            if key != "params" and key not in OPTIMIZER_IMPLEMENTATION_KEYS and not callable(value)
        }
        for group in optimizer.param_groups
    ]


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


def storage_path_for(path: str | Path, log_storage_dir: str | Path) -> Path | None:
    """Return the path under log_storage_dir that repeats path relative to the working directory.

    For example runs/gym_imagination/x gives <log_storage_dir>/runs/gym_imagination/x.
    Return None for a path outside the working directory, for example a pytest tmp_path.
    """

    try:
        relative = Path(path).resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return None

    return Path(log_storage_dir).expanduser().resolve() / relative


def link_run_dir_to_storage(run_log_dir: Path, log_storage_dir: str | Path) -> Path | None:
    """Create a new run directory under log_storage_dir and symlink it back to run_log_dir.

    Return the storage path. Return None and do nothing for a run_log_dir outside the working directory.
    """

    run_log_dir = Path(run_log_dir)
    if run_log_dir.exists() or run_log_dir.is_symlink():
        raise FileExistsError(f"{run_log_dir} exists (or is a broken symlink). Remove it or use a new run_name.")

    storage = storage_path_for(run_log_dir, log_storage_dir)
    if storage is None:
        return None

    if storage.exists():
        raise FileExistsError(
            f"log storage directory already exists: {storage}. Remove it or use a new run_name."
        )

    storage.mkdir(parents = True)
    run_log_dir.parent.mkdir(parents = True, exist_ok = True)
    run_log_dir.symlink_to(storage, target_is_directory = True)
    return storage


def serialize_hyperparameters(hyperparameters: dict):
    return "```json\n" + json.dumps(hyperparameters, indent = 2, sort_keys = True, default = str) + "\n```"


def configure_matmul_precision(allow_tf32: bool, device: torch.device):
    """Use TF32 and 'high' float32 matmul precision on CUDA. Every trainer calls this."""

    if allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def validate_dynamics_loss_normalization(
    use_dynamics_loss_normalization: bool,
    freeze_dynamics_loss_normalization_after_pretraining: bool,
    group_dynamics_loss_normalization: bool = False,
    use_paper_aligned_shortcut_objective: bool = False,
    dynamics_loss_normalization_start_loop: int = 0,
):
    if dynamics_loss_normalization_start_loop < 0:
        raise ValueError("dynamics_loss_normalization_start_loop must be 0 or more")

    if dynamics_loss_normalization_start_loop > 0 and not use_dynamics_loss_normalization:
        raise ValueError("dynamics_loss_normalization_start_loop requires use_dynamics_loss_normalization=true")

    if group_dynamics_loss_normalization and not use_dynamics_loss_normalization:
        raise ValueError("group_dynamics_loss_normalization requires use_dynamics_loss_normalization=true")

    if group_dynamics_loss_normalization and use_paper_aligned_shortcut_objective:
        raise ValueError(
            "group_dynamics_loss_normalization does not apply with use_paper_aligned_shortcut_objective=true: "
            "that objective already has one dynamics loss normalizer"
        )

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
    total_envs: int,
    parallel_envs: int | None,
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
    continuous_dist_type: Literal["gaussian", "squashed_gaussian", "beta"],
    world_model_batch_size: int,
    world_model_train_steps: int,
    world_model_train_sequence_length: int,
    world_model_learning_rate: float,
    use_dynamics_loss_normalization: bool,
    freeze_dynamics_loss_normalization_after_pretraining: bool,
    use_paper_aligned_shortcut_objective: bool,
    group_dynamics_loss_normalization: bool = False,
    dynamics_loss_normalization_start_loop: int = 0,
    world_model_reward_loss_weight: float = 1.,
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
    objective: Literal["ppo", "pmpo", "spo", "reinforce"],
    reinforce_discount_factor: float,
    reinforce_gae_lambda: float,
    reinforce_return_normalization: Literal["none", "percentile"],
    reinforce_use_discounted_trajectory_weights: bool,
    pmpo_pos_to_neg_weight: float,
    pmpo_kl_div_loss_weight: float,
    use_delight_gating: bool,
    policy_entropy_weight: float,
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
    log_storage_dir: str | None,
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
    world_model_micro_batch_size: int | None = None,
    world_model_autocast_bf16: bool = False,
    use_attn_pool: bool = True,
    attn_pool_einsum_attend: bool = False,
    attn_res: str | None = None,
    attn_res_block_size: int = 2,
    dynamics_ff_expansion_factor: float | None = None,
    stop_after_loops: int | None = None,
    cuda_memory_fraction: float | None = None,
    small_model_only: bool = False,
    small_model: Mapping[str, Any] | None = None,
    align_generated_dream_transitions: bool = False,
    preserve_policy_during_world_model_updates: bool = False,
    world_model_policy_consistency_loss_weight: float = 1.,
    reanchor_heads_after_world_model_updates: bool = False,
    reanchor_heads_steps: int = 200,
    reanchor_heads_learning_rate: float = 3e-4,
    reanchor_heads_contexts: int = 64,
    reanchor_heads_context_length: int = 200,
    cleanrl_updated_ppo: bool = False,
    cleanrl_ppo_closer: bool = False,
    cleanrl_ppo_closer_hl_gauss_value_loss: bool = False,
    cleanrl_updated_ppo_constant_learning_rate: bool = False,
    reuse_ppo_dream_batch: bool = False,
    ppo_reuse_epochs: int = 4,
    ppo_reuse_num_minibatches: int = 4,
    ppo_reuse_max_exact_kl: float = 0.02,
    ppo_reuse_lr_backoff: float = 0.5,
    ppo_reuse_max_backoffs: int = 6,
    scale_reward_gradient_into_transformer: bool = False,
    reward_gradient_into_transformer_scale: float = 0.01,
    freeze_world_model: bool = False,
    behavior_cloning_in_world_model_updates: bool = False,
    world_model_behavior_cloning_loss_weight: float = 1.,
    fixed_full_transformer_finetuning: bool = False,
    fixed_full_transformer_finetuning_steps: int = 10,
    fixed_full_transformer_finetuning_rl_loss_weight: float = 1.,
    fixed_full_transformer_finetuning_dream_micro_batch_size: int | None = None,
    fixed_full_transformer_finetuning_grad_diagnostics: bool = True,
    cleanrl_ppo_entropy_weight: float = 0.,
    world_model_flip_onset_window_fraction: float = 0.,
    imagination_flip_onset_prompt_fraction: float = 0.,
    audit_deeper_dynamics: bool = False,
    audit_arrival_reward_target: bool = False,
    audit_quarter_envs_per_loop: bool = False,
    audit_diagnostics: bool = False,
    audit_diagnostics_every_env_steps: int = 640_000,
    offline_pretraining: bool = False,
    offline_stages: tuple[str, ...] = offline.OFFLINE_STAGES,
    offline_expert_repo: str = "cleanrl/HalfCheetah-v4-td3_continuous_action-seed1",
    offline_expert_revision: str = "85477545d6db77ae3c7ca8e4e0b48f515d4c1432",
    offline_expert_filename: str = "td3_continuous_action.cleanrl_model",
    offline_expert_cache_dir: str = "runs/offline_data/experts",
    offline_expert_path: str | None = None,
    offline_expert_min_return: float = 8000.,
    offline_data_dir: str = "runs/offline_data",
    offline_dataset_name: str = "halfcheetah_v4_mix_1m",
    offline_data_steps: int = 1_000_000,
    offline_data_mix: Mapping[str, float] | None = None,
    offline_expert_noise_stds: tuple[float, ...] = (0.1, 0.2, 0.4, 0.8),
    offline_past_run_replays: tuple[str, ...] = (),
    offline_past_run_return_buckets: tuple[tuple[float, float], ...] = (),
    offline_expert_action_scales: tuple[float, ...] = (),
    offline_expert_scale_change_steps: int = 0,
    offline_expert_scaled_noise_std: float = 0.,
    offline_bang_bang_hold_steps: tuple[float, ...] = (),
    offline_past_run_flip_episode_fraction: float = 0.,
    offline_data_seed: int = 10042,
    offline_data_num_envs: int = 64,
    offline_pretrain_dir: str = "runs/offline_pretrain",
    offline_pretrain_name: str | None = None,
    offline_world_model_steps: int = 20000,
    offline_world_model_finetune_steps: int = 0,
    offline_validation_fraction: float = 0.05,
    offline_pretrain_checkpoint_every: int = 2000,
    offline_pretrained_checkpoint: str | None = None,
    offline_load_world_optimizer: bool = True,
    offline_rehearsal_fraction: float = 0.25,
):
    hyperparameters = dict(locals())

    if offline_data_mix is None:
        offline_data_mix = offline.DEFAULT_DATA_MIX

    # Offline data collection and pretraining (notes/pretraining_impl_plan.md).
    # The stages are empty unless offline_pretraining is true.
    offline_stages = offline.validate_offline_config(
        offline_pretraining = offline_pretraining,
        small_model_only = small_model_only,
        offline_stages = offline_stages,
        offline_pretrained_checkpoint = offline_pretrained_checkpoint,
        checkpoint_path = checkpoint_path,
        offline_rehearsal_fraction = offline_rehearsal_fraction,
        offline_data_mix = offline_data_mix,
        offline_expert_noise_stds = offline_expert_noise_stds,
        offline_data_steps = offline_data_steps,
        offline_data_num_envs = offline_data_num_envs,
        offline_validation_fraction = offline_validation_fraction,
        offline_world_model_steps = offline_world_model_steps,
        offline_world_model_finetune_steps = offline_world_model_finetune_steps,
        offline_pretrain_checkpoint_every = offline_pretrain_checkpoint_every,
        continuous_dist_type = continuous_dist_type,
        offline_past_run_replays = offline_past_run_replays,
        offline_past_run_return_buckets = offline_past_run_return_buckets,
        offline_expert_action_scales = offline_expert_action_scales,
        offline_expert_scale_change_steps = offline_expert_scale_change_steps,
        offline_expert_scaled_noise_std = offline_expert_scaled_noise_std,
        offline_bang_bang_hold_steps = offline_bang_bang_hold_steps,
        offline_past_run_flip_episode_fraction = offline_past_run_flip_episode_fraction,
    )
    offline_pretrain_stage = "pretrain" in offline_stages
    offline_rl_stage = "rl" in offline_stages

    if small_model_only and (audit_diagnostics or any(hyperparameters[flag] for flag in AUDIT_ABLATION_FLAGS)):
        raise ValueError("audit ablation and diagnostics flags require small_model_only=false")

    # Keep the standard Dreamer path untouched unless the explicit benchmark
    # flag selects the state-only trainer before tokenizer setup.
    if small_model_only:
        from .train_model_free import run as run_state_actor_critic
        return run_state_actor_critic(**hyperparameters)

    if not run_details:
        raise ValueError("run_details is required and must be non-empty")

    # Audit ablations (runs/audit_9-24-26/REPORT.md). The overrides are empty
    # unless an audit flag is set. Tokenizer pretraining keeps the configured
    # environment count, so every arm pretrains on the same random data.
    # total_envs sets the episodes per rollout. parallel_envs sets the async worker processes that step them.
    if exists(parallel_envs) and parallel_envs < 1:
        raise ValueError(f"parallel_envs must be positive or null, got {parallel_envs}")
    num_envs = total_envs
    tokenizer_pretrain_num_envs = num_envs
    audit_overrides = resolve_audit_ablations(
        audit_deeper_dynamics = audit_deeper_dynamics,
        audit_arrival_reward_target = audit_arrival_reward_target,
        audit_quarter_envs_per_loop = audit_quarter_envs_per_loop,
        depth = depth,
        time_block_every = time_block_every,
        old_reward_shifting = old_reward_shifting,
        align_generated_dream_transitions = align_generated_dream_transitions,
        num_envs = num_envs,
        replay_size = replay_size,
        num_loops = num_loops,
        checkpoint_every = checkpoint_every,
    )

    if len(audit_overrides) > 0:
        hyperparameters["audit_effective_overrides"] = audit_overrides
        depth = audit_overrides.get("depth", depth)
        old_reward_shifting = audit_overrides.get("old_reward_shifting", old_reward_shifting)
        align_generated_dream_transitions = audit_overrides.get("align_generated_dream_transitions", align_generated_dream_transitions)
        num_envs = audit_overrides.get("num_envs", num_envs)
        replay_size = audit_overrides.get("replay_size", replay_size)
        num_loops = audit_overrides.get("num_loops", num_loops)
        checkpoint_every = audit_overrides.get("checkpoint_every", checkpoint_every)

    if full_transformer_finetuning and objective != "pmpo":
        raise ValueError("full_transformer_finetuning requires objective='pmpo' for the policy-prior loss")

    if full_transformer_finetuning and pmpo_kl_div_loss_weight <= 0.:
        raise ValueError("full_transformer_finetuning requires pmpo_kl_div_loss_weight > 0")

    if align_generated_dream_transitions and not old_reward_shifting:
        raise ValueError("align_generated_dream_transitions requires old_reward_shifting=true")

    if preserve_policy_during_world_model_updates and world_model_policy_consistency_loss_weight <= 0.:
        raise ValueError("world_model_policy_consistency_loss_weight must be positive")

    for name, fraction in (
        ("world_model_flip_onset_window_fraction", world_model_flip_onset_window_fraction),
        ("imagination_flip_onset_prompt_fraction", imagination_flip_onset_prompt_fraction),
    ):
        if not 0. <= fraction < 1.:
            raise ValueError(f"{name} must be in [0, 1), got {fraction}")
        if fraction > 0. and not env_name.startswith("HalfCheetah"):
            raise ValueError(f"{name} finds flips from the HalfCheetah root angle; {env_name} is not HalfCheetah")
        if fraction > 0. and (old_replay_sampling or fixed_full_transformer_finetuning):
            raise ValueError(f"{name} does not support old_replay_sampling or fixed_full_transformer_finetuning")

    if imagination_flip_onset_prompt_fraction > 0. and imagination_prompt_length <= 0:
        raise ValueError("imagination_flip_onset_prompt_fraction needs imagination_prompt_length > 0")

    if reanchor_heads_after_world_model_updates:
        if reanchor_heads_steps <= 0 or reanchor_heads_contexts <= 0 or reanchor_heads_learning_rate <= 0.:
            raise ValueError("reanchor_heads_steps, reanchor_heads_contexts, and reanchor_heads_learning_rate must be positive")
        if not 1 < reanchor_heads_context_length <= max_timesteps:
            raise ValueError("reanchor_heads_context_length must be in (1, max_timesteps]")
        if freeze_world_model:
            raise ValueError("reanchor_heads_after_world_model_updates needs world-model updates, so freeze_world_model must be false")
        if full_transformer_finetuning:
            raise ValueError("reanchor_heads_after_world_model_updates and full_transformer_finetuning exclude each other")
        # BC trains the policy head in the world-model phase on purpose; a re-anchor would undo it.
        if behavior_cloning_in_world_model_updates:
            raise ValueError("reanchor_heads_after_world_model_updates and behavior_cloning_in_world_model_updates exclude each other")

    if reuse_ppo_dream_batch and objective != "ppo":
        raise ValueError("reuse_ppo_dream_batch requires objective='ppo'")

    if reuse_ppo_dream_batch and not normalize_advantages:
        raise ValueError("reuse_ppo_dream_batch requires normalize_advantages=true")

    if cleanrl_updated_ppo and objective != "ppo":
        raise ValueError("cleanrl_updated_ppo requires objective='ppo'")

    if cleanrl_updated_ppo and reuse_ppo_dream_batch:
        raise ValueError("cleanrl_updated_ppo and reuse_ppo_dream_batch are mutually exclusive")

    if cleanrl_updated_ppo and full_transformer_finetuning:
        raise ValueError("cleanrl_updated_ppo does not support full_transformer_finetuning")

    if cleanrl_updated_ppo and continuous_dist_type != "beta":
        raise ValueError("cleanrl_updated_ppo requires continuous_dist_type='beta'")

    if cleanrl_updated_ppo and use_delight_gating:
        raise ValueError("cleanrl_updated_ppo requires use_delight_gating=false")

    if cleanrl_ppo_closer and not cleanrl_updated_ppo:
        raise ValueError("cleanrl_ppo_closer requires cleanrl_updated_ppo=true")

    if cleanrl_ppo_closer_hl_gauss_value_loss and not cleanrl_ppo_closer:
        raise ValueError("cleanrl_ppo_closer_hl_gauss_value_loss requires cleanrl_ppo_closer=true")

    if cleanrl_updated_ppo_constant_learning_rate and not cleanrl_updated_ppo:
        raise ValueError("cleanrl_updated_ppo_constant_learning_rate requires cleanrl_updated_ppo=true")

    validate_dynamics_loss_normalization(
        use_dynamics_loss_normalization,
        freeze_dynamics_loss_normalization_after_pretraining,
        group_dynamics_loss_normalization,
        use_paper_aligned_shortcut_objective,
        dynamics_loss_normalization_start_loop,
    )

    if dynamics_loss_normalization_start_loop > 0 and offline_pretraining:
        raise ValueError("dynamics_loss_normalization_start_loop does not apply to the offline pretraining path")

    if exists(dynamics_ff_expansion_factor) and old_dynamics_ff:
        raise ValueError("old_dynamics_ff sets its own feedforward expansion; leave dynamics_ff_expansion_factor null")

    if exists(checkpoint_path) and clear_log_dir:
        raise ValueError("clear_log_dir must be false when checkpoint_path is set, otherwise resume state would be deleted")

    effective_imagination_train_steps = 1 if cleanrl_ppo_closer else imagination_train_steps

    # Paper phase 3 (imagination training, section 3.3): update only the policy and value heads and keep the
    # pretrained world model fixed. The online loop then skips the world-model training steps.
    if freeze_world_model:
        if not offline_rl_stage:
            raise ValueError("freeze_world_model requires the offline rl stage, which loads a pretrained world model")
        if full_transformer_finetuning:
            raise ValueError("freeze_world_model and full_transformer_finetuning exclude each other")
        if offline_rehearsal_fraction > 0.:
            raise ValueError("freeze_world_model skips world-model training, so set offline_rehearsal_fraction to 0")

    # Paper phase 2 (behavior cloning and reward model, section 3.3) in the online loop: each world-model update
    # also trains the policy head on the replay actions (multi-token prediction, eq. 9). The world-model optimizer
    # updates the transformer and the policy head together, so the policy head follows the change of its input
    # features. The imagination RL update of the policy head does not change.
    # OPEN QUESTION: the BC target is the whole replay, so BC pulls the policy toward the mean of all past policies
    # (the paper's target is the dataset policy). Recent replay only, or a KL to the current policy, are options.
    # OPEN QUESTION: as in the paper, BC reads noisy inputs (the signal levels of world-model training). The acting
    # policy reads clean inputs, so BC follows the feature change only through the noisy inputs.
    # OPEN QUESTION: the world-model optimizer normalizes the gradient scale (AdamAtan2), so the loss weight sets the
    # pull of BC on the transformer, but not the size of the policy-head step (about the world-model lr).
    # OPEN QUESTION: at the bootstrap state of a truncated episode, the replay holds a padded action (0, env action
    # -1). BC also trains on it (at most 8 targets per window that reaches the episode end, about 6e-6 of all
    # targets). The world-model dynamics already read this action. A mask is not implemented.
    if behavior_cloning_in_world_model_updates:
        if world_model_behavior_cloning_loss_weight <= 0.:
            raise ValueError("world_model_behavior_cloning_loss_weight must be positive")
        if freeze_world_model:
            raise ValueError("behavior_cloning_in_world_model_updates needs world-model updates, so freeze_world_model must be false")
        if full_transformer_finetuning:
            raise ValueError("behavior_cloning_in_world_model_updates and full_transformer_finetuning exclude each other")
        if preserve_policy_during_world_model_updates:
            raise ValueError("behavior_cloning_in_world_model_updates and preserve_policy_during_world_model_updates exclude each other")
        # OPEN QUESTION: the offline pretrain stage is the paper's phase 1 and has no policy head. BC on the offline
        # data in that stage is the paper's phase 2, but the offline data holds random-policy episodes. Not supported.
        if offline_pretrain_stage:
            raise ValueError("behavior_cloning_in_world_model_updates applies to the online loop only; remove the pretrain stage")
        # OPEN QUESTION: rehearsal windows come from the offline data (expert, expert + noise, and random policies).
        # BC on them pulls the policy toward that mix, and 30 % of it is random actions. A mask that keeps BC on the
        # online windows only is not implemented, so rehearsal is not supported with this flag.
        if offline_rl_stage and offline_rehearsal_fraction > 0.:
            raise ValueError("behavior_cloning_in_world_model_updates does not support rehearsal; set offline_rehearsal_fraction to 0")

    effective_world_model_train_steps = 0 if freeze_world_model else world_model_train_steps

    # Full-transformer finetuning in the RL loop (paper section 3.3 and its footnote), as the paper's phase 2 with the
    # RL loss in place of BC: each step sums the world-model loss on replay windows and the RL loss on dreams, and one
    # optimizer updates the transformer and the heads. This joint phase replaces the world-model phase and the
    # imagination phase of the loop, so those two phases have 0 steps.
    # The paper gives no optimizer, learning rate, or loss weights for this mode. The joint optimizer is the
    # world-model optimizer, with the policy and value heads as one more group (world-model lr and weight decay).
    joint_finetune_steps = 0
    if fixed_full_transformer_finetuning:
        if fixed_full_transformer_finetuning_steps <= 0:
            raise ValueError("fixed_full_transformer_finetuning_steps must be positive")
        if fixed_full_transformer_finetuning_rl_loss_weight <= 0.:
            raise ValueError("fixed_full_transformer_finetuning_rl_loss_weight must be positive")
        if exists(fixed_full_transformer_finetuning_dream_micro_batch_size) and fixed_full_transformer_finetuning_dream_micro_batch_size <= 0:
            raise ValueError("fixed_full_transformer_finetuning_dream_micro_batch_size must be positive or null")
        # PMPO with the policy-prior KL (the paper's RL loss), or closer CleanRL PPO (the objective of the A2 runs).
        # Closer PPO has no prior. Its 10 epochs of 32 minibatches do not exist here: each joint step does one pass
        # over one new dream batch, so the importance ratio is 1 and the clip has no effect.
        if not ((objective == "pmpo" and pmpo_kl_div_loss_weight > 0.) or cleanrl_ppo_closer):
            raise ValueError(
                "fixed_full_transformer_finetuning requires objective='pmpo' with pmpo_kl_div_loss_weight > 0 "
                "(the policy-prior loss), or cleanrl_ppo_closer=true"
            )
        excluded_flags = dict(
            full_transformer_finetuning = full_transformer_finetuning,
            freeze_world_model = freeze_world_model,
            behavior_cloning_in_world_model_updates = behavior_cloning_in_world_model_updates,
            preserve_policy_during_world_model_updates = preserve_policy_during_world_model_updates,
            reanchor_heads_after_world_model_updates = reanchor_heads_after_world_model_updates,
            old_replay_sampling = old_replay_sampling,
            align_generated_dream_transitions = align_generated_dream_transitions,
            optimize_generate_compilation = optimize_generate_compilation,
        )
        for flag, enabled in excluded_flags.items():
            if enabled:
                raise ValueError(f"fixed_full_transformer_finetuning and {flag} exclude each other")
        # Without the time cache, the rollout reads noised context frames, and one forward cannot give the inputs
        # that the policy read. The learner then trains on other inputs than the rollout.
        if not imagination_use_time_cache:
            raise ValueError("fixed_full_transformer_finetuning requires imagination_use_time_cache=true")
        if imagination_prompt_length > 0 and imagination_prompt_probability != 1.:
            raise ValueError("fixed_full_transformer_finetuning requires imagination_prompt_probability=1.0 when imagination_prompt_length > 0")

        joint_finetune_steps = fixed_full_transformer_finetuning_steps
        effective_world_model_train_steps = 0
        effective_imagination_train_steps = 0

    # The offline rl stage replaces the in-loop world-model pretraining.
    in_loop_world_model_pretrain_steps = (
        0 if offline_rl_stage
        else pretrain_world_model_finetune_steps + pretrain_world_model_combined_steps
    )
    offline_pretrain_steps = (
        offline_world_model_finetune_steps + offline_world_model_steps
        if offline_pretrain_stage
        else 0
    )
    runs_online_loop = not offline_pretraining or offline_rl_stage
    configured_profile_steps = (
        offline_pretrain_steps
        + (
            in_loop_world_model_pretrain_steps
            if not exists(checkpoint_path) and num_loops > 0 and runs_online_loop
            else 0
        )
        + (num_loops * (effective_world_model_train_steps + effective_imagination_train_steps + joint_finetune_steps) if runs_online_loop else 0)
    )
    if profile and configured_profile_steps < PROFILE_TOTAL_STEPS:
        raise ValueError(
            f"profile mode requires at least {PROFILE_TOTAL_STEPS} configured world-model/imagination steps "
            f"(5 warmup + 4 active), but this run has {configured_profile_steps}"
        )

    compile_mode = normalize_compile_mode(compile_mode)
    obs_dim, action_dim, action_range = inspect_env_spaces(env_name, obs_dim)

    # The run directories stay in their normal place if the storage folder does not exist (for example on the cluster).
    if exists(log_storage_dir) and not Path(log_storage_dir).expanduser().is_dir():
        print(f"log_storage_dir {log_storage_dir} does not exist. Run directories stay in their normal place.")
        log_storage_dir = None

    # The collect stage runs before the CUDA check, so a collect-only run works on a CPU node.
    # The expert actor uses the GPU if one is available.
    offline_dataset_dir = offline.offline_dataset_dir(offline_data_dir, env_name, offline_dataset_name)

    if "collect" in offline_stages:
        offline.collect_offline_dataset(
            env_name = env_name,
            obs_dim = obs_dim,
            action_dim = action_dim,
            action_range = action_range,
            data_dir = offline_data_dir,
            dataset_name = offline_dataset_name,
            data_steps = offline_data_steps,
            data_mix = offline_data_mix,
            noise_stds = offline_expert_noise_stds,
            data_seed = offline_data_seed,
            num_envs = offline_data_num_envs,
            expert_repo = offline_expert_repo,
            expert_revision = offline_expert_revision,
            expert_filename = offline_expert_filename,
            expert_cache_dir = offline_expert_cache_dir,
            expert_path = offline_expert_path,
            expert_min_return = offline_expert_min_return,
            expert_device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda"),
            storage_dir = (
                storage_path_for(offline_dataset_dir, log_storage_dir)
                if exists(log_storage_dir) and not offline_dataset_dir.is_symlink()
                else None
            ),
            past_run_replays = offline_past_run_replays,
            past_run_return_buckets = offline_past_run_return_buckets,
            expert_action_scales = offline_expert_action_scales,
            expert_scale_change_steps = offline_expert_scale_change_steps,
            expert_scaled_noise_std = offline_expert_scaled_noise_std,
            bang_bang_hold_steps = offline_bang_bang_hold_steps,
            past_run_flip_episode_fraction = offline_past_run_flip_episode_fraction,
        )

        if offline_stages == ("collect",):
            return None

    offline_signature = None
    offline_dataset = None
    offline_data_split = None
    offline_pretrain_path = None
    offline_package = None
    offline_env_steps = None

    if offline_pretraining:
        offline_signature = offline.pretrain_signature({
            **hyperparameters,
            "obs_dim": obs_dim,
            "depth": depth,
            "old_reward_shifting": old_reward_shifting,
        })
        offline_overrides = {}

        if offline_pretrain_stage:
            offline_dataset = offline.load_offline_dataset(offline_dataset_dir)
            offline_data_split = offline.offline_split(offline_dataset, offline_validation_fraction, offline_data_seed)
            offline_pretrain_path = Path(offline_pretrain_dir) / (offline_pretrain_name or run_name)
            offline_env_steps = offline_dataset.env_steps

            if (offline_pretrain_path / offline.PRETRAINED_FILE).exists():
                raise FileExistsError(
                    f"{offline_pretrain_path / offline.PRETRAINED_FILE} exists. Use offline_stages=[rl] with "
                    "offline_pretrained_checkpoint, or choose a new offline_pretrain_name."
                )

            if exists(log_storage_dir) and not offline_pretrain_path.exists() and not offline_pretrain_path.is_symlink():
                storage = link_run_dir_to_storage(offline_pretrain_path, log_storage_dir)
                if exists(storage):
                    print(f"offline pretrain directory {offline_pretrain_path} -> {storage}")

            offline_overrides["pretrain_tokenizer_observations"] = int(
                (offline_dataset.lens[offline_data_split.train] + 1).sum()
            )
        elif offline_rl_stage:
            # Check the signature before any expensive setup. A resumed run
            # reads only the metadata; its loop checkpoint holds the models.
            offline_package = offline.load_offline_pretrained(offline_pretrained_checkpoint, signature = offline_signature)
            offline_env_steps = int(offline_package["offline_env_steps"])

            if exists(checkpoint_path):
                offline_package = None
            else:
                offline_overrides.update(pretrain_tokenizer_steps = 0, pretrain_tokenizer_observations = 0)

        if offline_rl_stage:
            offline_overrides.update(pretrain_world_model_finetune_steps = 0, pretrain_world_model_combined_steps = 0)

        hyperparameters["offline_effective_overrides"] = offline_overrides
        hyperparameters["offline_env_steps"] = offline_env_steps
        hyperparameters["offline_signature"] = offline_signature

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")

    # A cap on the memory that the caching allocator of this process may hold. Two runs share a 16 GB GPU only if the
    # first run does not cache most of the free memory at its start (entry 50 ran out of memory that way).
    if exists(cuda_memory_fraction):
        if not 0. < cuda_memory_fraction <= 1.:
            raise ValueError(f"cuda_memory_fraction must be in (0, 1], got {cuda_memory_fraction}")
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction)

    if compile is None:
        compile = device.type == "cuda"

    configure_sac(activation_memory_budget)

    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required for this run, but no CUDA device is available")

    if env_vectorization_mode not in ("async", "sync"):
        raise ValueError(f"env_vectorization_mode must be 'async' or 'sync', got {env_vectorization_mode!r}")

    # Sync, async, and grouped async envs give the same data. Env i gets seed + i in each mode.
    allocated_cpus = available_cpu_count()
    env_workers = min(num_envs, parallel_envs) if exists(parallel_envs) else num_envs
    if env_vectorization_mode == "async" and env_workers > allocated_cpus:
        print(
            f"warning: async vectorization requested {env_workers} environment workers, "
            f"but this process can use only {allocated_cpus} CPUs. Set parallel_envs to {allocated_cpus} or less."
        )

    if cleanrl_ppo_closer:
        torch.backends.cudnn.deterministic = True

    configure_matmul_precision(allow_tf32, device)

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

    # A restart of the pretrain stage continues in the same run directory.
    resumes_offline_pretrain = offline_pretrain_stage and offline.pretrain_can_resume(offline_pretrain_path)

    if not exists(checkpoint_path) and run_log_dir.exists() and not resumes_offline_pretrain:
        raise FileExistsError(
            f"run directory already exists for run_name={run_name!r}: {run_log_dir}. "
            "Use a new run_name or pass checkpoint_path to continue the existing run."
        )

    if clear_log_dir:
        if run_log_dir.is_symlink():
            shutil.rmtree(run_log_dir.resolve(), ignore_errors = True)
            run_log_dir.unlink()
        else:
            shutil.rmtree(run_log_dir, ignore_errors = True)

    # A new run directory can live on another disk. The symlink keeps the normal path.
    if exists(log_storage_dir) and not run_log_dir.exists():
        storage = link_run_dir_to_storage(run_log_dir, log_storage_dir)
        if exists(storage):
            print(f"run directory {run_log_dir} -> {storage}")

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

    if offline_pretrain_stage:
        # Fit the normalization and train the tokenizer on the offline data.
        tokenizer_loss = offline.prepare_offline_tokenizer(
            tokenizer,
            dataset = offline_dataset,
            split = offline_data_split,
            pretrain_dir = offline_pretrain_path,
            signature = offline_signature,
            train_tokenizer_fn = lambda model, observations: train_tokenizer(
                model,
                observations,
                steps = pretrain_tokenizer_steps,
                batch_size = tokenizer_batch_size,
                learning_rate = tokenizer_learning_rate,
                max_grad_norm = max_grad_norm,
                device = device,
                writer = writer,
                training_profiler = training_profiler,
            ),
            writer = writer,
            device = device,
        )
    elif offline_rl_stage:
        # The pretrained checkpoint, or the loop checkpoint on resume, holds the tokenizer.
        pass
    elif not exists(checkpoint_path):
        pretrain_obs = collect_random_observations(
            env_name,
            num_steps = max(1, pretrain_tokenizer_observations // tokenizer_pretrain_num_envs),
            num_envs = tokenizer_pretrain_num_envs,
            parallel_envs = parallel_envs,
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

    env = None
    if runs_online_loop:
        env = make_env(
            env_name,
            seed,
            vectorized = True,
            num_envs = num_envs,
            parallel_envs = parallel_envs,
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
            use_attn_pool = use_attn_pool,
            attn_pool_einsum_attend = attn_pool_einsum_attend,
            attn_res = attn_res,
            attn_res_block_size = attn_res_block_size,
        ),
        final_special_cross_attn = final_special_cross_attn,
        num_discrete_actions = 0,
        num_continuous_actions = action_dim,
        continuous_dist_type = continuous_dist_type,
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
        # The model sums the BC loss over the 8 multi-token-prediction distances, as it sums the reward loss.
        continuous_action_loss_weight = world_model_behavior_cloning_loss_weight if behavior_cloning_in_world_model_updates else 0.,
        discrete_action_loss_weight = 0.,
        agent_predicts_state = agent_predicts_state,
        agent_state_pred_loss_weight = agent_state_pred_loss_weight,
        prob_shortcut_train = prob_shortcut_train,
        policy_all_tokens = policy_all_tokens,
        gae_discount_factor = 0.99,
        reinforce_discount_factor = reinforce_discount_factor,
        reinforce_gae_lambda = reinforce_gae_lambda,
        reinforce_return_normalization = reinforce_return_normalization,
        reinforce_use_discounted_trajectory_weights = reinforce_use_discounted_trajectory_weights,
        pmpo_pos_to_neg_weight = pmpo_pos_to_neg_weight,
        pmpo_kl_div_loss_weight = pmpo_kl_div_loss_weight,
        ppo_eps_clip = 0.2,
        value_clip = 0.2 if cleanrl_updated_ppo else 0.4,
        clip_values = cleanrl_updated_ppo,
        cleanrl_ppo_closer = cleanrl_ppo_closer,
        cleanrl_ppo_closer_hl_gauss_value_loss = cleanrl_ppo_closer_hl_gauss_value_loss,
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
        # CleanRL PPO uses its own entropy coefficient (ent_coef); 0 as in CleanRL continuous PPO.
        policy_entropy_weight = cleanrl_ppo_entropy_weight if cleanrl_updated_ppo else policy_entropy_weight,
        scale_reward_gradient_into_transformer = scale_reward_gradient_into_transformer,
        reward_gradient_into_transformer_scale = reward_gradient_into_transformer_scale,
        use_loss_normalization = use_dynamics_loss_normalization,
        use_paper_aligned_shortcut_objective = use_paper_aligned_shortcut_objective,
        **(dict(group_dynamics_loss_normalization = True) if group_dynamics_loss_normalization else dict()),
        **(dict(reward_loss_weight = world_model_reward_loss_weight) if world_model_reward_loss_weight != 1. else dict()),
        **(dict(ff_kwargs = dict(expansion_factor = dynamics_ff_expansion_factor)) if exists(dynamics_ff_expansion_factor) else dict()),
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

    def add_policy_head_to_world_optimizer():
        # As in the paper's phase 2, one optimizer updates the transformer and the policy head with one loss.
        # The policy head is an extra group of the world-model optimizer, with the world-model lr and weight decay.
        # OPEN QUESTION: the paper gives no optimizer or lr. The agent optimizer keeps its own state for the same
        # policy-head parameters, so the policy head has two optimizer states, one for BC and one for RL. The paper
        # never updates the policy head with BC and RL in the same phase, so it gives no answer here.
        # A separate group (added after the other groups) keeps the saved group order of a pretrained optimizer.
        # policy_head_parameters() includes a set, so the group takes the module order: a saved state then maps to
        # the same parameters in each process (see make_optimizer).
        policy_param_set = set(world_model.policy_head_parameters())
        world_optimizer.add_param_group(dict(params = [param for param in world_model.parameters() if param in policy_param_set]))

    # A pretrained optimizer state has only the world-model groups. Its load (below) comes first, then the group.
    if behavior_cloning_in_world_model_updates and not exists(offline_package):
        add_policy_head_to_world_optimizer()

    def add_agent_heads_to_world_optimizer():
        # fixed_full_transformer_finetuning: the world-model optimizer becomes the one optimizer of the run. The
        # policy and value heads are one more group, added after the other groups, in the module order (see
        # add_policy_head_to_world_optimizer). The heads thus get the world-model lr, weight decay, and update rule.
        # With closer PPO, the group keeps the agent lr of that objective (the loop sets its linear decay).
        agent_param_set = set(world_model.agent_parameters())
        world_optimizer.add_param_group(dict(
            params = [param for param in world_model.parameters() if param in agent_param_set],
            **(dict(lr = agent_learning_rate) if cleanrl_ppo_closer else dict()),
        ))

        owners = [param for group in world_optimizer.param_groups for param in group["params"]]
        assert len(owners) == len(set(owners)) and set(owners) == set(world_model.parameters()), (
            "fixed_full_transformer_finetuning: the optimizer must hold each model parameter one time"
        )

    if fixed_full_transformer_finetuning and not exists(offline_package):
        add_agent_heads_to_world_optimizer()

    if fixed_full_transformer_finetuning:
        # One optimizer only. The agent parameters have no second optimizer state.
        agent_optimizer = None
    elif cleanrl_updated_ppo:
        agent_optimizer = make_cleanrl_updated_ppo_optimizer(
            world_model,
            lr = agent_learning_rate,
        )
    elif reuse_ppo_dream_batch:
        agent_optimizer = make_ppo_reuse_optimizer(
            world_model,
            lr = agent_learning_rate,
            weight_decay = optimizer_weight_decay,
        )
    else:
        agent_optimizer = make_optimizer(
            agent_params,
            lr = agent_learning_rate,
            weight_decay = optimizer_weight_decay,
            use_muon = use_muon_optimizer,
            muon_params = world_model.muon_parameters(),
        )

    cleanrl_updated_ppo_shuffle_generator = (
        torch.Generator(device = device).manual_seed(seed)
        if cleanrl_ppo_closer
        else None
    )

    # Checkpoints keep the settings from the config, not the settings at save
    # time. Schedules and PPO backoff change lr during training.
    optimizer_settings = dict(
        world = optimizer_group_settings(world_optimizer),
        agent = optimizer_group_settings(agent_optimizer) if exists(agent_optimizer) else [],
    )

    start_loop = 0
    resume_progress = None
    replay = None
    memmap_path = run_log_dir / "replay_buffer"
    if exists(checkpoint_path):
        pkg = torch.load(checkpoint_path, map_location = device, weights_only = True)
        resume_progress = pkg.get("progress")
        if resume_progress is None:
            raise ValueError(
                f"{checkpoint_path} has no step counters. Checkpoints saved before resume "
                "support existed cannot resume."
            )

        # load_state_dict replaces lr, weight_decay, and the other group settings
        # with the saved values, so a changed config value has no effect.
        assert pkg["optimizer_settings"] == optimizer_settings, (
            "the optimizer settings of the config differ from the checkpoint. A resume "
            "keeps the checkpoint settings, so restore the config values. "
            f"checkpoint: {pkg['optimizer_settings']}, config: {optimizer_settings}"
        )

        tokenizer.load_state_dict(pkg["tokenizer"])
        world_model.load_state_dict(pkg["world_model"])
        # Muon optimizer states in checkpoints saved before the fixed parameter
        # order are not compatible (see make_optimizer).
        load_optimizer_state_if_compatible(world_optimizer, pkg["world_optimizer"], "world")
        if exists(agent_optimizer):
            load_optimizer_state_if_compatible(agent_optimizer, pkg["agent_optimizer"], "agent")
        start_loop = int(pkg.get("loop", 0)) + 1

        # Episodes collected after the checkpoint were not seen by the saved
        # model, so the replay goes back to the snapshot from the same loop.
        replay = restore_replay_snapshot(Path(checkpoint_path), memmap_path)

    if exists(offline_package):
        # The world-model keys and the tokenizer come from the pretrained checkpoint.
        # The policy and value heads keep their initialization.
        offline.apply_offline_pretrained(
            offline_package,
            tokenizer = tokenizer,
            world_model = world_model,
            world_optimizer = world_optimizer,
            load_world_optimizer = offline_load_world_optimizer,
        )
        print(f"loaded offline pretrained checkpoint: {offline_pretrained_checkpoint}")
        del offline_package

        if behavior_cloning_in_world_model_updates:
            add_policy_head_to_world_optimizer()
            # Loop checkpoints and their resume check must have the settings of all groups.
            optimizer_settings["world"] = optimizer_group_settings(world_optimizer)

        if fixed_full_transformer_finetuning:
            add_agent_heads_to_world_optimizer()
            optimizer_settings["world"] = optimizer_group_settings(world_optimizer)

    if behavior_cloning_in_world_model_updates:
        # The loss weights are model buffers. A loaded state dict (a pretrained package has weight 0, and a resume
        # checkpoint has its own weight) replaces the configured weight, and a weight of 0 removes the BC gradient.
        world_model.continuous_action_loss_weight.fill_(world_model_behavior_cloning_loss_weight)

    # The agent loss reads detached world-model outputs, so only the policy and value heads train.
    if freeze_world_model:
        for param in world_model.world_model_parameters():
            param.requires_grad_(False)

    remaining_profile_steps = (
        offline_pretrain_steps
        + (
            in_loop_world_model_pretrain_steps
            if start_loop == 0 and num_loops > 0 and runs_online_loop
            else 0
        )
        + (
            max(0, num_loops - start_loop) * (effective_world_model_train_steps + effective_imagination_train_steps + joint_finetune_steps)
            if runs_online_loop
            else 0
        )
    )
    if profile and remaining_profile_steps < PROFILE_TOTAL_STEPS:
        raise ValueError(
            f"profile mode requires {PROFILE_TOTAL_STEPS} remaining world-model/imagination steps, "
            f"but the loaded run has {remaining_profile_steps}"
        )

    policy_prior = FrozenPolicyPrior(world_model).to(device) if objective == "pmpo" else None
    if fixed_full_transformer_finetuning and objective == "pmpo":
        # The prior is the policy at the start of each loop, as a frozen copy of the transformer and the policy head.
        policy_prior = FrozenModelPolicyPrior(world_model).to(device)

    head_reanchor = None
    if reanchor_heads_after_world_model_updates:
        head_reanchor = HeadReanchor(
            world_model,
            steps = reanchor_heads_steps,
            learning_rate = reanchor_heads_learning_rate,
            # The step size of interact_with_env and of the policy-consistency contexts.
            step_size = world_model.max_steps // imagination_generate_steps,
            chunk_size = default(world_model_micro_batch_size, reanchor_heads_contexts),
        )
        # A separate stream, so the windows do not change the other random streams of the run.
        head_reanchor_rng = np.random.default_rng([seed, 11])
    compile_interact = compile or compile_interact
    compile_world_model = compile or compile_world_model
    compile_generate = compile or compile_generate
    compile_learn = False if cleanrl_ppo_closer else compile or compile_learn
    static_compile_shapes = compile_dynamic is False

    if compile_generate and static_compile_shapes and imagination_prompt_length > 0 and imagination_prompt_probability != 1.:
        raise ValueError("compile_generate requires imagination_prompt_probability=1.0 when imagination_prompt_length > 0")

    if fixed_full_transformer_finetuning and compile_generate and not compile_generate_with_time_cache:
        # A compiled rollout (not only its inner forward) gives other first-frame inputs than the eager learner.
        raise ValueError("fixed_full_transformer_finetuning with compile_generate requires compile_generate_with_time_cache=true")

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
        store_old_action_unembeds = (
            (objective == "pmpo" and not exists(policy_prior)) or
            (objective == "ppo" and reuse_ppo_dream_batch)
        ),
        full_transformer_finetuning = full_transformer_finetuning,
        # The joint learner reads the latents that the rollout read, so it gets them before the clamp to (-1, 1).
        clamp_generated_latents = not fixed_full_transformer_finetuning,
        # The joint learner uses the eager forward. The compiled forward gives other outputs for the first generated
        # frame (EagerWithoutTimeCacheCallable), so the rollout uses the eager forward for that frame too.
        eager_generate_without_time_cache = fixed_full_transformer_finetuning,
    )

    if offline_pretrain_stage:
        # Use the world optimizer, compiled loss, and batch shape of the online
        # loop, so the rl stage reuses the same compiled graphs.
        offline_pretrained_checkpoint = offline.pretrain_offline(
            tokenizer = tokenizer,
            world_model = world_model,
            world_optimizer = world_optimizer,
            dataset = offline_dataset,
            split = offline_data_split,
            pretrain_dir = offline_pretrain_path,
            signature = offline_signature,
            action_range = action_range,
            device = device,
            train_steps = partial(
                train_world_model,
                world_model,
                world_optimizer,
                world_model_loss_fn = compile_runtime.world_model_loss,
                world_model_loss_timing = compile_runtime.world_model_loss_timing,
                world_model_step_timing = compile_runtime.world_model_step_timing,
                batch_size = world_model_batch_size,
                sequence_length = world_model_train_sequence_length,
                static_batch_shape = compile_world_model and static_compile_shapes,
                max_grad_norm = max_grad_norm,
                micro_batch_size = world_model_micro_batch_size,
                autocast_bf16 = world_model_autocast_bf16,
                # A pretrain-only run logs to the normal world_model/ and extra/ graphs. A run that also has the rl
                # stage keeps the prefix, because the online world-model steps start again at 0.
                writer = (
                    offline.PrefixedScalarWriter(writer, "offline_pretrain/train/")
                    if exists(writer) and offline_rl_stage
                    else writer
                ),
                training_profiler = training_profiler,
                old_replay_sampling = old_replay_sampling,
                update_loss_ema = True,
                flip_onset_window_fraction = world_model_flip_onset_window_fraction,
            ),
            validate = partial(
                world_model_validation_metrics,
                world_model,
                batch_size = world_model_batch_size,
                sequence_length = world_model_train_sequence_length,
                num_batches = offline.WORLD_MODEL_VALIDATION_BATCHES,
                micro_batch_size = world_model_micro_batch_size,
            ),
            world_model_steps = offline_world_model_steps,
            finetune_steps = offline_world_model_finetune_steps,
            checkpoint_every = offline_pretrain_checkpoint_every,
            writer = writer,
        )

        if not offline_rl_stage:
            training_profiler.flush_timings(flush_writer = True)
            print_compile_timing_report(compile_runtime.timings)

            if exists(event_writer):
                event_writer.flush()

            return compile_runtime.timings if return_compile_timings else None

    # Rehearsal keeps offline windows in each online world-model batch.
    # Parallel runs share the offline replay, so it opens read-only.
    offline_rehearsal_replay = None
    if offline_rl_stage and offline_rehearsal_fraction > 0.:
        offline_rehearsal_replay = offline.open_rehearsal_replay(offline_pretrained_checkpoint)

    if start_loop == 0:
        shutil.rmtree(memmap_path, ignore_errors=True)

    # A resume continues the counters, so the TensorBoard steps, the audit
    # schedule, and the closer-PPO metric step do not restart at zero.
    resume_progress = resume_progress or {}
    wm_step = int(resume_progress.get("wm_step", 0))
    imagination_step = int(resume_progress.get("imagination_step", 0))
    env_step = int(resume_progress.get("env_step", 0))
    # The offline rl stage starts from a pretrained world model, so the
    # in-loop pretraining on the first rollout does not run.
    did_world_model_pretrain = start_loop > 0 or offline_rl_stage

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

    audit_runner = None
    if audit_diagnostics:
        audit_runner = AuditDiagnostics(
            env_name = env_name,
            obs_dim = obs_dim,
            action_range = action_range,
            prompt_length = imagination_prompt_length,
            horizon = imagination_horizon,
            generate_steps = imagination_generate_steps,
            use_time_cache = imagination_use_time_cache,
            old_reward_shifting = old_reward_shifting,
            every_env_steps = audit_diagnostics_every_env_steps,
            seed = seed,
            device = device,
        )
        register_runtime_cleanup(audit_runner.close)
        if start_loop == 0:
            log_scalars(
                writer,
                {"audit/world_model/parameters": sum(param.numel() for param in world_params)},
                0,
            )
        audit_runner.next_env_step = int(resume_progress.get("audit_next_env_step", audit_runner.next_env_step))
        audit_start_time = perf_counter() - float(resume_progress.get("audit_elapsed_seconds", 0.))

    pbar = tqdm(range(start_loop, num_loops), desc = "loops")

    for loop in pbar:
        # the plain loss sum before dynamics_loss_normalization_start_loop (the loop 0 pretrain phase included)
        world_model.apply_loss_normalization = loop >= dynamics_loss_normalization_start_loop

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

                if exists(audit_runner):
                    log_scalars(writer, audit_runner.rollout_metrics(cpu_exp, rollout_lengths), env_step)

                for episode_return, episode_length in zip(rollout_returns, rollout_lengths):
                    log_scalars(
                        writer,
                        {
                            "charts/episodic_return": episode_return,
                            "charts/episodic_length": episode_length,
                        },
                        env_step,
                    )

                    # Sample-efficiency comparisons with model-free runs count the offline env steps too.
                    if exists(offline_env_steps):
                        log_scalars(
                            writer,
                            {"charts/episodic_return_offline_inclusive": episode_return},
                            env_step + offline_env_steps,
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

        # The world model has not trained on this rollout yet, so these
        # metrics use held-out current-policy data.
        if exists(audit_runner) and did_world_model_pretrain and audit_runner.is_due(env_step):
            with training_profiler.record("audit/diagnostics"):
                audit_metrics = audit_runner.model_metrics(
                    world_model,
                    tokenizer,
                    replay,
                    cpu_exp,
                    rollout_lengths,
                )
            log_scalars(writer, audit_metrics, env_step)

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
                micro_batch_size = world_model_micro_batch_size,
                autocast_bf16 = world_model_autocast_bf16,
                global_step = wm_step,
                writer = writer,
                training_profiler = training_profiler,
                shortcut_train_mode = "finetune",
                desc = "world model pretrain finetune",
                old_replay_sampling = old_replay_sampling,
                update_loss_ema = True,
                flip_onset_window_fraction = world_model_flip_onset_window_fraction,
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
                micro_batch_size = world_model_micro_batch_size,
                autocast_bf16 = world_model_autocast_bf16,
                global_step = wm_step,
                writer = writer,
                training_profiler = training_profiler,
                shortcut_train_mode = "combined",
                desc = "world model pretrain combined",
                old_replay_sampling = old_replay_sampling,
                update_loss_ema = True,
                flip_onset_window_fraction = world_model_flip_onset_window_fraction,
            )

            did_world_model_pretrain = True

        if exists(head_reanchor) and effective_world_model_train_steps > 0:
            with training_profiler.record("world_model/reanchor_capture", cuda = True):
                head_reanchor.capture(rollout_context_windows(
                    cpu_exp,
                    num_windows = reanchor_heads_contexts,
                    window_length = reanchor_heads_context_length,
                    rng = head_reanchor_rng,
                ))

        wm_step, wm_metrics = train_world_model(
            world_model,
            world_optimizer,
            replay,
            world_model_loss_fn = compile_runtime.world_model_loss,
            world_model_loss_timing = compile_runtime.world_model_loss_timing,
            world_model_step_timing = compile_runtime.world_model_step_timing,
            steps = effective_world_model_train_steps,
            batch_size = world_model_batch_size,
            sequence_length = world_model_train_sequence_length,
            static_batch_shape = compile_world_model and static_compile_shapes,
            max_grad_norm = max_grad_norm,
            micro_batch_size = world_model_micro_batch_size,
            autocast_bf16 = world_model_autocast_bf16,
            global_step = wm_step,
            writer = writer,
            training_profiler = training_profiler,
            shortcut_train_mode = "combined",
            desc = "world model",
            old_replay_sampling = old_replay_sampling,
            update_loss_ema = not freeze_dynamics_loss_normalization_after_pretraining,
            preserve_policy_during_world_model_updates = preserve_policy_during_world_model_updates,
            world_model_policy_consistency_loss_weight = world_model_policy_consistency_loss_weight,
            policy_consistency_step_size = world_model.max_steps // imagination_generate_steps,
            rehearsal_replay = offline_rehearsal_replay,
            rehearsal_fraction = offline_rehearsal_fraction,
            flip_onset_window_fraction = world_model_flip_onset_window_fraction,
        )

        if exists(head_reanchor):
            with training_profiler.record("world_model/reanchor_fit", cuda = True):
                reanchor_metrics = head_reanchor.fit()
            log_scalars(writer, reanchor_metrics, env_step)

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
            steps = effective_imagination_train_steps,
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
            align_generated_dream_transitions = align_generated_dream_transitions,
            cleanrl_updated_ppo = cleanrl_updated_ppo,
            cleanrl_ppo_closer = cleanrl_ppo_closer,
            cleanrl_updated_ppo_iteration_offset = loop if cleanrl_ppo_closer else loop * imagination_train_steps,
            cleanrl_updated_ppo_total_iterations = num_loops if cleanrl_ppo_closer else num_loops * imagination_train_steps,
            cleanrl_updated_ppo_initial_learning_rate = agent_learning_rate,
            cleanrl_updated_ppo_constant_learning_rate = cleanrl_updated_ppo_constant_learning_rate,
            cleanrl_updated_ppo_shuffle_generator = cleanrl_updated_ppo_shuffle_generator,
            cleanrl_updated_ppo_metric_step = env_step if cleanrl_ppo_closer else None,
            reuse_ppo_dream_batch = reuse_ppo_dream_batch,
            ppo_reuse_epochs = ppo_reuse_epochs,
            ppo_reuse_num_minibatches = ppo_reuse_num_minibatches,
            ppo_reuse_max_exact_kl = ppo_reuse_max_exact_kl,
            ppo_reuse_lr_backoff = ppo_reuse_lr_backoff,
            ppo_reuse_max_backoffs = ppo_reuse_max_backoffs,
            flip_onset_prompt_fraction = imagination_flip_onset_prompt_fraction,
        )

        if fixed_full_transformer_finetuning:
            # The world-model phase and the imagination phase above have 0 steps. Each step of this phase sums the
            # world-model loss and the RL loss, and world_optimizer (the one optimizer) updates all parameters.
            joint_rl_loss = JointImaginationLoss(
                world_model,
                policy_prior,
                replay,
                objective = objective,
                value_loss_weight = CLEANRL_UPDATED_PPO_VALUE_LOSS_WEIGHT if cleanrl_ppo_closer else 1.,
                generate_rollout_fn = compile_runtime.generate_rollout,
                batch_size = imagination_batch_size,
                horizon = imagination_horizon,
                prompt_length = imagination_prompt_length,
                static_generate_shape = compile_generate and static_compile_shapes,
                loss_weight = fixed_full_transformer_finetuning_rl_loss_weight,
                micro_batch_size = fixed_full_transformer_finetuning_dream_micro_batch_size,
                use_delight_gating = use_delight_gating,
                grad_diagnostics = fixed_full_transformer_finetuning_grad_diagnostics,
                autocast_bf16 = world_model_autocast_bf16,
                global_step = imagination_step,
                writer = writer,
                training_profiler = training_profiler,
            )
            world_optimizer.zero_grad(set_to_none = True)

            if cleanrl_ppo_closer:
                # The linear lr decay of closer PPO, on the group of the policy and value heads (the last group).
                world_optimizer.param_groups[-1]["lr"] = (1. if cleanrl_updated_ppo_constant_learning_rate else 1. - loop / num_loops) * agent_learning_rate

            wm_step, wm_metrics = train_world_model(
                world_model,
                world_optimizer,
                replay,
                world_model_loss_fn = compile_runtime.world_model_loss,
                world_model_loss_timing = compile_runtime.world_model_loss_timing,
                world_model_step_timing = compile_runtime.world_model_step_timing,
                steps = joint_finetune_steps,
                batch_size = world_model_batch_size,
                sequence_length = world_model_train_sequence_length,
                static_batch_shape = compile_world_model and static_compile_shapes,
                max_grad_norm = max_grad_norm,
                micro_batch_size = world_model_micro_batch_size,
                autocast_bf16 = world_model_autocast_bf16,
                global_step = wm_step,
                writer = writer,
                training_profiler = training_profiler,
                shortcut_train_mode = "combined",
                desc = "world model + imagination",
                update_loss_ema = not freeze_dynamics_loss_normalization_after_pretraining,
                rehearsal_replay = offline_rehearsal_replay,
                rehearsal_fraction = offline_rehearsal_fraction,
                extra_loss_fn = joint_rl_loss,
                grad_clip_groups = joint_finetune_clip_groups(world_model, world_optimizer),
            )
            imagination_step, imagination_metrics = joint_rl_loss.global_step, joint_rl_loss.last_metrics

        if exists(audit_runner):
            log_scalars(
                writer,
                audit_runner.progress_metrics(
                    loop = loop,
                    elapsed_seconds = perf_counter() - audit_start_time,
                    world_model_updates = wm_step,
                    imagination_updates = imagination_step,
                    world_model_metrics = wm_metrics,
                    imagination_metrics = imagination_metrics,
                ),
                env_step,
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
                progress = dict(env_step = env_step, wm_step = wm_step, imagination_step = imagination_step)
                if exists(audit_runner):
                    progress.update(
                        audit_next_env_step = audit_runner.next_env_step,
                        audit_elapsed_seconds = perf_counter() - audit_start_time,
                    )

                save_resume_checkpoint(
                    checkpoint_root / f"loop_{loop + 1}.pt",
                    replay = replay,
                    loop = loop,
                    tokenizer = tokenizer,
                    world_model = world_model,
                    world_optimizer = world_optimizer,
                    agent_optimizer = agent_optimizer,
                    progress = progress,
                    optimizer_settings = optimizer_settings,
                )

        if device.type == "cuda":
            # The peak GPU memory of each loop gives the memory fraction for runs that share the GPU.
            training_profiler.log_timing("memory/cuda_max_allocated", torch.cuda.max_memory_allocated(device) / 2 ** 20, metric = "mebibytes")
            training_profiler.log_timing("memory/cuda_max_reserved", torch.cuda.max_memory_reserved(device) / 2 ** 20, metric = "mebibytes")
            torch.cuda.reset_peak_memory_stats(device)

        training_profiler.flush_timings(flush_writer = True)

        # A budget stop that keeps num_loops, so the learning-rate schedules stay the same.
        if exists(stop_after_loops) and loop + 1 >= stop_after_loops:
            break

    print_compile_timing_report(compile_runtime.timings)

    if exists(event_writer):
        event_writer.flush()

    if return_compile_timings:
        return compile_runtime.timings

if __name__ == "__main__":
    fire.Fire(main)
