# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "tqdm",
#     "dreamer4",
#     "adam-atan2-pytorch>=0.3.6",
#     "torch-einops-utils",
#     "fire",
#     "gymnasium[mujoco]",
#     "tensorboard",
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

# contributed by @CarsonBurke in https://github.com/lucidrains/dreamer4/pull/29

from __future__ import annotations

import random
import shutil
from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal
from math import sqrt

import fire
import gymnasium as gym
import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils._pytree import tree_map

from einops import rearrange
from adam_atan2_pytorch import MuonAdamAtan2
from torch_einops_utils import safe_cat, temp_eval, lens_to_mask

from memmap_replay_buffer.replay_buffer import ReplayBuffer, ReplayDatasetTimeWindow, collate_var_time

from dreamer4.dreamer4 import (
    Actions,
    DynamicsWorldModel,
    Experience,
    StateTokenizer,
    combine_experiences,
    divisible_by,
    exists,
    tree_map_tensor
)

def cycle(dl):
    while True:
        for data in dl:
            yield data


# compile helpers

@dataclass
class CallTiming:
    name: str
    compiled: bool
    compile_mode: str | None = None
    cuda_graphs: bool = False
    calls: int = 0
    total_seconds: float = 0.
    first_call_seconds: float | None = None

    def record(self, seconds: float):
        self.calls += 1
        self.total_seconds += seconds

        if self.first_call_seconds is None:
            self.first_call_seconds = seconds

    @property
    def warm_calls(self):
        return max(self.calls - 1, 0)

    @property
    def warm_seconds(self):
        return self.total_seconds - (self.first_call_seconds or 0.)

    @property
    def warm_average_seconds(self):
        return self.warm_seconds / self.warm_calls if self.warm_calls > 0 else None


class TimedCallable:
    def __init__(
        self,
        fn: Callable,
        timing: CallTiming,
        device: torch.device,
    ):
        self.fn = fn
        self.timing = timing
        self.device = device

    def __call__(self, *args, **kwargs):
        synchronize_if_cuda(self.device)
        start = perf_counter()
        out = self.fn(*args, **kwargs)
        synchronize_if_cuda(self.device)
        self.timing.record(perf_counter() - start)
        return out


def normalize_compile_mode(mode: str | None):
    if not exists(mode):
        return None

    mode = str(mode)

    if mode in ("", "none", "None", "null"):
        return None

    if mode == "default":
        return "default"

    if mode in ("reduce_overhead", "reduce-overhead"):
        return "reduce_overhead"

    raise ValueError("compile_mode must be one of None, 'default', or 'reduce_overhead'")


def torch_compile_mode(mode: str | None):
    mode = normalize_compile_mode(mode)
    return "reduce-overhead" if mode == "reduce_overhead" else mode


def compile_mode_uses_cuda_graphs(mode: str | None):
    return normalize_compile_mode(mode) == "reduce_overhead"


def cudagraph_mark_step_begin():
    mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)

    if callable(mark_step_begin):
        mark_step_begin()


class CompiledCallable:
    def __init__(
        self,
        fn: Callable,
        timing: CallTiming,
    ):
        self.fn = fn
        self.timing = timing

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class WorldModelTrainingLoss(nn.Module):
    def __init__(self, world_model: DynamicsWorldModel):
        super().__init__()
        self.world_model = world_model

    def forward(
        self,
        latents: Tensor,
        rewards: Tensor | None,
        terminals: Tensor | None,
        continuous_actions: Tensor | None,
        lens: Tensor | None,
    ):
        return self.world_model(
            latents = latents,
            rewards = rewards,
            terminals = terminals,
            continuous_actions = continuous_actions,
            lens = lens,
            latent_has_view_dim = latents.ndim == 5,
            return_all_losses = True,
            update_loss_ema = True,
        )


class ImaginationGenerateRollout(nn.Module):
    def __init__(
        self,
        world_model: DynamicsWorldModel,
        *,
        generate_steps: int,
        store_old_action_unembeds: bool,
        use_time_cache: bool,
    ):
        super().__init__()
        self.world_model = world_model
        self.generate_steps = generate_steps
        self.store_old_action_unembeds = store_old_action_unembeds
        self.use_time_cache = use_time_cache

    def forward(
        self,
        generation_horizon: int,
        batch_size: int,
        prompt_latents: Tensor | None = None,
        prompt_proprio: Tensor | None = None,
        prompt_discrete_actions: Tensor | None = None,
        prompt_continuous_actions: Tensor | None = None,
        prompt_rewards: Tensor | None = None,
    ):
        return self.world_model.generate(
            generation_horizon,
            num_steps = self.generate_steps,
            batch_size = batch_size,
            return_decoded_video = False,
            return_agent_actions = True,
            return_log_probs_and_values = True,
            return_rewards_per_frame = True,
            return_terminals = False,
            use_time_cache = self.use_time_cache,
            store_agent_embed = True,
            store_old_action_unembeds = self.store_old_action_unembeds,
            prompt_latents = prompt_latents,
            prompt_proprio = prompt_proprio,
            prompt_discrete_actions = prompt_discrete_actions,
            prompt_continuous_actions = prompt_continuous_actions,
            prompt_rewards = prompt_rewards,
        )


class ImaginationLearningLoss(nn.Module):
    def __init__(
        self,
        world_model: DynamicsWorldModel,
        *,
        objective: Literal["ppo", "pmpo", "spo"],
        use_delight_gating: bool,
    ):
        super().__init__()
        self.world_model = world_model
        self.objective = objective
        self.use_delight_gating = use_delight_gating

    def forward(
        self,
        latents: Tensor,
        proprio: Tensor | None,
        critic_state: Tensor | None,
        agent_embed: Tensor | None,
        rewards: Tensor | None,
        terminals: Tensor | None,
        actions_discrete: Tensor | None,
        actions_continuous: Tensor | None,
        log_probs_discrete: Tensor | None,
        log_probs_continuous: Tensor | None,
        old_action_unembeds_discrete: Tensor | None,
        old_action_unembeds_continuous: Tensor | None,
        values: Tensor | None,
        step_size: int | Tensor | None,
        lens: Tensor | None,
        is_truncated: Tensor | None,
        is_from_world_model: bool | Tensor,
        episode_return: Tensor | None,
    ):
        experience = Experience(
            latents = latents,
            proprio = proprio,
            critic_state = critic_state,
            agent_embed = agent_embed,
            rewards = rewards,
            terminals = terminals,
            actions = Actions(actions_discrete, actions_continuous),
            log_probs = Actions(log_probs_discrete, log_probs_continuous),
            old_action_unembeds = Actions(old_action_unembeds_discrete, old_action_unembeds_continuous),
            values = values,
            step_size = step_size,
            lens = lens,
            is_truncated = is_truncated,
            is_from_world_model = is_from_world_model,
            episode_return = episode_return,
        )

        return self.world_model.learn_from_experience(
            experience,
            only_learn_policy_value_heads = True,
            objective = self.objective,
            use_delight_gating = self.use_delight_gating,
            return_diagnostics = True,
        )


@dataclass
class CompileRuntime:
    world_model_loss: Callable
    world_model_loss_timing: CallTiming | None
    generate_rollout: Callable
    generate_rollout_timing: CallTiming | None
    learn_from_dream: Callable
    learn_from_dream_timing: CallTiming | None
    world_model_step_timing: CallTiming | None
    imagination_step_timing: CallTiming | None
    timings: list[CallTiming]


@dataclass
class BenchmarkResult:
    name: str
    wall_seconds: float
    hot_path_seconds: float
    first_call_seconds: float
    warm_seconds: float
    warm_average_seconds: float | None
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None


def synchronize_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_torch_compile_state(device: torch.device):
    torch.compiler.reset()

    try:
        import torch._inductor.codecache as codecache

        codecache.FxGraphCache.clear()
        codecache.PyCodeCache.cache_clear()
        codecache.CppCodeCache.cache_clear()
    except Exception:
        pass

    if device.type == "cuda":
        torch.cuda.empty_cache()


def torch_compile_kwargs(
    *,
    backend: str,
    mode: str | None,
    fullgraph: bool,
    dynamic: bool | None,
):
    kwargs = dict(backend = backend, fullgraph = fullgraph)
    mode = torch_compile_mode(mode)

    if exists(mode) and mode != "default":
        kwargs["mode"] = mode

    if dynamic is not None:
        kwargs["dynamic"] = dynamic

    return kwargs


def maybe_compile_and_time(
    fn: Callable,
    *,
    name: str,
    enabled: bool,
    track_timing: bool,
    device: torch.device,
    compile_backend: str,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool | None,
    compile_cudagraphs = True,
    wrap_timing = True,
):
    compile_mode = normalize_compile_mode(compile_mode)
    uses_cuda_graphs = (
        enabled and
        device.type == "cuda" and
        compile_cudagraphs and
        compile_mode_uses_cuda_graphs(compile_mode)
    )
    effective_compile_mode = compile_mode

    if enabled and compile_mode_uses_cuda_graphs(compile_mode) and not compile_cudagraphs:
        # CUDA Graph capture is unsafe for these autograd loss wrappers: AOTAutograd
        # keeps forward intermediates live for backward, and graph replay can overwrite them.
        effective_compile_mode = "default"

    timing = CallTiming(
        name = name,
        compiled = enabled,
        compile_mode = effective_compile_mode,
        cuda_graphs = uses_cuda_graphs,
    )

    if enabled:
        fn = torch.compile(
            fn,
            **torch_compile_kwargs(
                backend = compile_backend,
                mode = effective_compile_mode,
                fullgraph = compile_fullgraph,
                dynamic = compile_dynamic,
            )
        )
        fn = CompiledCallable(fn, timing)

    if not enabled and not track_timing:
        return fn, timing

    if not wrap_timing:
        return fn, timing

    return TimedCallable(fn, timing, device), timing


def build_compile_runtime(
    world_model: DynamicsWorldModel,
    *,
    device: torch.device,
    compile_world_model: bool,
    compile_generate: bool,
    compile_learn: bool,
    compile_backend: str,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool | None,
    track_compile_performance: bool,
    imagination_generate_steps: int,
    imagination_use_time_cache: bool,
    objective: Literal["ppo", "pmpo", "spo"],
    use_delight_gating: bool,
    store_old_action_unembeds: bool,
) -> CompileRuntime:
    timings = []

    world_model_loss, timing = maybe_compile_and_time(
        WorldModelTrainingLoss(world_model),
        name = "world_model_loss",
        enabled = compile_world_model,
        track_timing = track_compile_performance,
        device = device,
        compile_backend = compile_backend,
        compile_mode = compile_mode,
        compile_fullgraph = compile_fullgraph,
        compile_dynamic = compile_dynamic,
        compile_cudagraphs = False,
        wrap_timing = False,
    )
    world_model_loss_timing = timing if compile_world_model or track_compile_performance else None
    timings.append(timing)

    generate_rollout, timing = maybe_compile_and_time(
        ImaginationGenerateRollout(
            world_model,
            generate_steps = imagination_generate_steps,
            store_old_action_unembeds = store_old_action_unembeds,
            use_time_cache = imagination_use_time_cache,
        ),
        name = "imagination_generate",
        enabled = compile_generate,
        track_timing = track_compile_performance,
        device = device,
        compile_backend = compile_backend,
        compile_mode = compile_mode,
        compile_fullgraph = compile_fullgraph,
        compile_dynamic = compile_dynamic,
        compile_cudagraphs = True,
    )
    generate_rollout_timing = timing if compile_generate or track_compile_performance else None
    timings.append(timing)

    learn_from_dream, timing = maybe_compile_and_time(
        ImaginationLearningLoss(
            world_model,
            objective = objective,
            use_delight_gating = use_delight_gating,
        ),
        name = "imagination_learn",
        enabled = compile_learn,
        track_timing = track_compile_performance,
        device = device,
        compile_backend = compile_backend,
        compile_mode = compile_mode,
        compile_fullgraph = compile_fullgraph,
        compile_dynamic = compile_dynamic,
        compile_cudagraphs = False,
        wrap_timing = False,
    )
    learn_from_dream_timing = timing if compile_learn or track_compile_performance else None
    timings.append(timing)

    world_model_step_timing = CallTiming(name = "world_model_step", compiled = compile_world_model)
    imagination_step_timing = CallTiming(name = "imagination_step", compiled = compile_learn or compile_generate)

    if track_compile_performance:
        timings.extend([world_model_step_timing, imagination_step_timing])
    else:
        world_model_step_timing = None
        imagination_step_timing = None

    return CompileRuntime(
        world_model_loss = world_model_loss,
        world_model_loss_timing = world_model_loss_timing,
        generate_rollout = generate_rollout,
        generate_rollout_timing = generate_rollout_timing,
        learn_from_dream = learn_from_dream,
        learn_from_dream_timing = learn_from_dream_timing,
        world_model_step_timing = world_model_step_timing,
        imagination_step_timing = imagination_step_timing,
        timings = timings,
    )


def print_compile_timing_report(timings: list[CallTiming]):
    active_timings = [timing for timing in timings if timing.calls > 0]

    if len(active_timings) == 0:
        return

    print("\ncompile performance accounting:")

    for timing in active_timings:
        first = timing.first_call_seconds or 0.
        warm_avg = timing.warm_average_seconds
        warm_avg_text = f"{warm_avg:.4f}s" if exists(warm_avg) else "n/a"
        compiled_text = "compiled" if timing.compiled else "eager"
        mode_text = timing.compile_mode or ("compiled" if timing.compiled else "eager")

        print(
            f"  {timing.name} ({compiled_text}): "
            f"calls={timing.calls}, first_call={first:.4f}s, "
            f"warm_total={timing.warm_seconds:.4f}s, warm_avg={warm_avg_text}, "
            f"total={timing.total_seconds:.4f}s, "
            f"mode={mode_text}, cuda_graphs={timing.cuda_graphs}"
        )


def record_timed_block(timing: CallTiming | None, device: torch.device, fn: Callable):
    if not exists(timing):
        return fn()

    synchronize_if_cuda(device)
    start = perf_counter()
    out = fn()
    synchronize_if_cuda(device)
    timing.record(perf_counter() - start)
    return out


def action_pair_or_empty(actions):
    if not exists(actions):
        return Actions(None, None)

    if isinstance(actions, Actions):
        return actions

    return Actions(*actions)

# tokenizer

class ObservationTokenizer(nn.Module):
    def __init__(
        self,
        obs_dim: int = 17,
        num_latent_tokens: int = 4,
        dim_latent: int = 32,
        hidden_dim: int = 256,
        depth: int = 2,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_latent_tokens = num_latent_tokens
        self.dim_latent = dim_latent
        self.eps = eps

        self.tokenizer = StateTokenizer(
            dim_state = obs_dim,
            num_latent_tokens = num_latent_tokens,
            dim_latent = dim_latent,
            dim = hidden_dim,
            depth = depth,
            attn_every = 0,
            attn_dim_head = 64
        )

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

def make_env(
    env_name: str,
    seed: int | None,
    *,
    vectorized = True,
    num_envs = 8,
    episode_stats_buffer_length = 100,
):
    if vectorized:
        env = gym.make_vec(env_name, num_envs = num_envs)
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
):
    env = make_env(env_name, seed, vectorized = True, num_envs = num_envs)
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


# training helpers

def log_scalars(writer: SummaryWriter | None, scalars: dict[str, float], step: int):
    if not exists(writer):
        return

    for key, value in scalars.items():
        if value is None:
            continue
        writer.add_scalar(key, float(value), step)

def get_episode_count(env) -> int:
    return int(getattr(env, "episode_count", 0))

def get_completed_episode_stats(env, start_episode_count: int):
    return_queue = getattr(env, "return_queue", None)
    length_queue = getattr(env, "length_queue", None)

    if return_queue is None or length_queue is None:
        return [], []

    num_new_episodes = max(get_episode_count(env) - start_episode_count, 0)

    if num_new_episodes == 0:
        return [], []

    if num_new_episodes > len(return_queue) or num_new_episodes > len(length_queue):
        raise RuntimeError(
            f"episode statistics buffer only retained {len(return_queue)} returns and {len(length_queue)} lengths, "
            f"but {num_new_episodes} episodes completed since the last read"
        )

    returns = list(return_queue)[-num_new_episodes:]
    lengths = list(length_queue)[-num_new_episodes:]

    return returns, lengths

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

    if not use_muon:
        return AdamW(params, lr = lr, weight_decay = weight_decay)

    optimizer_param_set = set(params)
    muon_params = [
        param for param in dict.fromkeys(muon_params)
        if param in optimizer_param_set
    ]

    if len(muon_params) == 0:
        return AdamW(params, lr = lr, weight_decay = weight_decay)

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
    return disjoint_optimizer_param_groups(
        optimizer,
        [
            ("reward_head", module_parameters(getattr(world_model, "to_reward_pred", None))),
            ("action_head", module_parameters(getattr(world_model, "action_embedder", None))),
            ("terminal_head", module_parameters(getattr(world_model, "to_state_terminal_pred", None))),
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
        self.policy_head = deepcopy(world_model.policy_head)
        self.action_embedder = deepcopy(world_model.action_embedder)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def refresh_from(self, world_model: DynamicsWorldModel):
        self.policy_head.load_state_dict(world_model.policy_head.state_dict())
        self.action_embedder.load_state_dict(world_model.action_embedder.state_dict())
        self.to(world_model.device)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def action_unembeds(self, agent_embeds: Tensor):
        policy_embed = self.policy_head(agent_embeds.detach())
        return self.action_embedder.unembed(policy_embed, pred_head_index = 0)


@torch.no_grad()
def agent_approx_kl_metrics(world_model: DynamicsWorldModel, dream: Experience):
    assert exists(dream.agent_embed)
    assert exists(dream.actions)
    assert exists(dream.log_probs)

    agent_embeds = dream.agent_embed.detach()
    policy_embed = world_model.policy_head(agent_embeds)
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

    agent_embeds = dream.agent_embed.detach()
    policy_embed = world_model.policy_head(agent_embeds)
    current_unembeds = world_model.action_embedder.unembed(policy_embed, pred_head_index = 0)
    prior_unembeds = policy_prior.action_unembeds(agent_embeds)

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
        kl_values = agent_embeds.new_zeros((1,))

    return {
        "prior_kl_mean": kl_values.mean(),
        "prior_kl_min": kl_values.min(),
        "prior_kl_max": kl_values.max(),
    }


def attach_policy_prior_unembeds(dream: Experience, policy_prior: FrozenPolicyPrior):
    assert exists(dream.agent_embed)
    dream.old_action_unembeds = policy_prior.action_unembeds(dream.agent_embed)
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
):
    tokenizer.to(device)
    observations = observations.to(device)
    tokenizer.set_normalization(observations)

    if steps <= 0:
        return None

    dataset = TensorDataset(observations)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = len(dataset) >= batch_size)
    optimizer = AdamW(tokenizer.parameters(), lr = learning_rate)
    iterator = cycle(dataloader) if len(dataloader) > 0 else None

    last_loss = None
    pbar = tqdm(range(steps), desc = "tokenizer")

    for step in pbar:
        batch = next(iterator)[0]

        loss = tokenizer(batch)
        loss.backward()

        clip_grad_norm_(tokenizer.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        last_loss = loss.item()
        pbar.set_postfix(loss = f"{last_loss:.4f}")
        log_scalars(writer, {"tokenizer/recon_loss": last_loss}, step)

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
    max_grad_norm: float,
    global_step: int,
    writer: SummaryWriter | None,
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

    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = len(dataset) >= batch_size
    )

    dataloader_iter = cycle(dataloader) if len(dataloader) > 0 else None

    pbar = tqdm(range(steps), desc="world model", dynamic_ncols=True)

    for step in pbar:
        if exists(world_model_step_timing):
            synchronize_if_cuda(world_model.device)
            step_start = perf_counter()

        batch = next(dataloader_iter)

        exp = Experience.from_buffer_dict(batch)
        exp.lens = batch.get('_lens')
        exp = exp.to(world_model.device)

        # Mask terminals and is_truncated based on whether the window reaches the episode end
        reaches_episode_end = batch.get('_reaches_episode_end')
        if exists(reaches_episode_end):
            reaches_episode_end = reaches_episode_end.to(world_model.device)
            if exists(exp.terminals): exp.terminals = exp.terminals & reaches_episode_end
            if exists(exp.is_truncated): exp.is_truncated = exp.is_truncated & reaches_episode_end.to(world_model.device)

        def forward_backward():
            loss, losses = world_model_loss_fn(
                exp.latents,
                exp.rewards,
                exp.terminals,
                exp.actions.continuous if exists(exp.actions) else None,
                exp.lens,
            )

            loss.backward()
            return loss, losses

        loss, losses = record_timed_block(world_model_loss_timing, world_model.device, forward_backward)

        norm, grad_metrics = clip_grad_norm_by_group(grad_clip_groups, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        last_metrics = {
            "world_model/loss": loss.item(),
            "world_model/flow_loss": losses.flow.item(),
            "world_model/shortcut_loss": losses.shortcut.item(),
            "world_model/reward_loss": (losses.agent_embed_reward.mean() + losses.latent_state_reward.mean()).item(),
            "world_model/terminal_loss": (losses.agent_embed_terminal.mean() + losses.latent_state_terminal.mean()).item(),
            "world_model/agent_state_pred_loss": losses.agent_state_pred.item(),
            "world_model/grad_norm": float(norm),
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
    generate_steps: int,
    max_grad_norm: float,
    objective: Literal["ppo", "pmpo", "spo"],
    use_delight_gating: bool,
    global_step: int,
    writer: SummaryWriter | None,
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

    prompt_dataloader = DataLoader(
        prompt_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = len(prompt_dataset) >= batch_size
    )

    prompt_iterator = cycle(prompt_dataloader) if prompt_length > 0 and len(prompt_dataloader) > 0 else None

    for _ in pbar:
        if exists(imagination_step_timing):
            synchronize_if_cuda(world_model.device)
            step_start = perf_counter()

        with torch.no_grad():
            use_prompt = random.random() < prompt_probability
            prompts = None

            if use_prompt and exists(prompt_iterator):
                prompt_batch = next(prompt_iterator)

                prompt_batch = tree_map(lambda t: t.to(world_model.device) if torch.is_tensor(t) else t, prompt_batch)

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
                dream = trim_prompt_from_dream(dream, prompt_length, horizon)

            if objective == "pmpo" and exists(policy_prior):
                dream = attach_policy_prior_unembeds(dream, policy_prior)

        dream_actions = action_pair_or_empty(dream.actions)
        dream_log_probs = action_pair_or_empty(dream.log_probs)
        dream_old_action_unembeds = action_pair_or_empty(dream.old_action_unembeds)

        def forward_backward():
            policy_loss, value_loss, value_diagnostics = learn_from_dream_fn(
                dream.latents,
                dream.proprio,
                dream.critic_state,
                dream.agent_embed,
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
            )

            loss = policy_loss + value_loss
            loss.backward()
            return policy_loss, value_loss, value_diagnostics, loss

        policy_loss, value_loss, value_diagnostics, loss = record_timed_block(
            learn_from_dream_timing,
            world_model.device,
            forward_backward,
        )

        norm, grad_metrics = clip_grad_norm_by_group(grad_clip_groups, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        agent_metrics = agent_approx_kl_metrics(world_model, dream)
        if objective == "pmpo" and exists(policy_prior):
            agent_metrics.update(agent_prior_kl_metrics(world_model, dream, policy_prior))

        raw_reward_sum_mean = dream.episode_return.mean().item() if exists(dream.episode_return) else 0.
        dream_len = dream.lens.float().mean().item() if exists(dream.lens) else horizon
        action_std = dream.actions.continuous.std().item() if exists(dream.actions) and exists(dream.actions.continuous) else 0.

        last_metrics = {
            "imagination/loss": loss.item(),
            "imagination/policy_loss": policy_loss.item(),
            "imagination/value_loss": value_loss.item(),
            "imagination/raw_reward_sum_mean": raw_reward_sum_mean,
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
        pbar.set_postfix(raw_reward = f"{raw_reward_sum_mean:.1f}", loss = f"{loss.item():.3f}")
        global_step += 1

        if exists(imagination_step_timing):
            synchronize_if_cuda(world_model.device)
            imagination_step_timing.record(perf_counter() - step_start)

        del dream
        del agent_metrics, grad_metrics
        del dream_actions, dream_log_probs, dream_old_action_unembeds
        del policy_loss, value_loss, value_diagnostics, loss

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
            world_optimizer = world_optimizer.state_dict(),
            agent_optimizer = agent_optimizer.state_dict(),
        ),
        str(path),
    )


def load_optimizer_state_if_compatible(optimizer: Optimizer, state_dict, name: str):
    try:
        optimizer.load_state_dict(state_dict)
        return True
    except (KeyError, RuntimeError, ValueError) as exc:
        print(f"skipping {name} optimizer state: {exc}")
        return False


def resolve_log_dir(log_dir: str, checkpoint_path: str | None):
    log_root = Path(log_dir)

    if exists(checkpoint_path):
        return log_root / f"resume_{Path(checkpoint_path).stem}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_root / timestamp


def run_compile_benchmark(
    main_kwargs: dict,
    *,
    benchmark_num_loops: int,
    benchmark_num_envs: int,
    benchmark_max_timesteps: int,
    benchmark_require_cuda: bool,
    benchmark_preset: Literal["smoke", "perf"],
):
    benchmark_root = Path(main_kwargs["log_dir"]) / "compile_benchmark"
    common_kwargs = {
        key: value
        for key, value in main_kwargs.items()
        if not key.startswith("benchmark_")
    }

    smoke = benchmark_preset == "smoke"

    common_kwargs.update(
        num_loops = benchmark_num_loops,
        rollouts_per_loop = 1,
        num_envs = benchmark_num_envs,
        max_timesteps = benchmark_max_timesteps,
        replay_size = 4,
        model_dim = min(int(main_kwargs["model_dim"]), 32) if smoke else main_kwargs["model_dim"],
        depth = min(int(main_kwargs["depth"]), 1) if smoke else main_kwargs["depth"],
        time_block_every = 1,
        world_model_batch_size = 1 if smoke else main_kwargs["world_model_batch_size"],
        world_model_train_steps = max(2, min(int(main_kwargs["world_model_train_steps"]), 2)),
        world_model_train_sequence_length = min(int(main_kwargs["world_model_train_sequence_length"]), benchmark_max_timesteps) if smoke else main_kwargs["world_model_train_sequence_length"],
        imagination_batch_size = 1 if smoke else main_kwargs["imagination_batch_size"],
        imagination_horizon = min(int(main_kwargs["imagination_horizon"]), 4) if smoke else main_kwargs["imagination_horizon"],
        imagination_prompt_length = min(int(main_kwargs["imagination_prompt_length"]), 2),
        imagination_train_steps = max(2, min(int(main_kwargs["imagination_train_steps"]), 2)),
        imagination_generate_steps = 2,
        imagination_use_time_cache = False,
        pretrain_tokenizer_steps = max(1, min(int(main_kwargs["pretrain_tokenizer_steps"]), 1)) if smoke else main_kwargs["pretrain_tokenizer_steps"],
        pretrain_tokenizer_observations = max(32, benchmark_num_envs * benchmark_max_timesteps) if smoke else main_kwargs["pretrain_tokenizer_observations"],
        tokenizer_batch_size = min(int(main_kwargs["tokenizer_batch_size"]), 16) if smoke else main_kwargs["tokenizer_batch_size"],
        tokenizer_eval_every = 0,
        use_muon_optimizer = False,
        use_tensorboard = False,
        checkpoint_every = 0,
        checkpoint_path = None,
        clear_log_dir = True,
        unique_log_dir = False,
        benchmark_compile = False,
        track_compile_performance = True,
        compile_fullgraph = main_kwargs["compile_fullgraph"],
        compile_dynamic = False,
        require_cuda = benchmark_require_cuda,
        prob_shortcut_train = 1.,
    )

    results = []

    for name, compile_enabled, compile_mode in (
        ("eager", False, None),
        ("compiled_default", True, "default"),
        ("compiled_reduce_overhead_generated_graphs", True, "reduce_overhead"),
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reset_torch_compile_state(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        run_kwargs = dict(common_kwargs)
        run_kwargs.update(
            compile = compile_enabled,
            compile_world_model = False,
            compile_generate = False,
            compile_learn = False,
            compile_mode = compile_mode,
            log_dir = str(benchmark_root / name),
            return_compile_timings = True,
        )

        print(f"\nbenchmark {name}:")
        start = perf_counter()
        timings = main(**run_kwargs)
        synchronize_if_cuda(device)
        wall_seconds = perf_counter() - start
        step_timings = [timing for timing in timings if timing.name in ("world_model_step", "imagination_step")]
        hot_path_seconds = sum(timing.total_seconds for timing in step_timings)
        first_call_seconds = sum(timing.first_call_seconds or 0. for timing in step_timings)
        warm_seconds = sum(timing.warm_seconds for timing in step_timings)
        warm_calls = sum(timing.warm_calls for timing in step_timings)
        warm_average_seconds = warm_seconds / warm_calls if warm_calls > 0 else None

        peak_allocated_mib = peak_reserved_mib = None
        if device.type == "cuda":
            peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 2 ** 20
            peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2 ** 20

        results.append(BenchmarkResult(
            name = name,
            wall_seconds = wall_seconds,
            hot_path_seconds = hot_path_seconds,
            first_call_seconds = first_call_seconds,
            warm_seconds = warm_seconds,
            warm_average_seconds = warm_average_seconds,
            peak_allocated_mib = peak_allocated_mib,
            peak_reserved_mib = peak_reserved_mib,
        ))
        print(f"benchmark {name} wall time: {wall_seconds:.3f}s")
        print(f"benchmark {name} measured hot-path time: {hot_path_seconds:.3f}s")
        print(f"benchmark {name} measured first-call time: {first_call_seconds:.3f}s")
        if exists(warm_average_seconds):
            print(f"benchmark {name} measured warm time: {warm_seconds:.3f}s total, {warm_average_seconds:.3f}s avg")
        if exists(peak_allocated_mib) and exists(peak_reserved_mib):
            print(f"benchmark {name} peak CUDA memory: {peak_allocated_mib:.1f} MiB allocated, {peak_reserved_mib:.1f} MiB reserved")

    eager = next(result for result in results if result.name == "eager")
    eager_seconds = eager.wall_seconds
    eager_hot_path = eager.hot_path_seconds
    eager_first = eager.first_call_seconds
    eager_warm = eager.warm_seconds
    eager_warm_average = eager.warm_average_seconds

    print("\nbenchmark summary:")
    print(f"  eager warm runtime: {eager_warm:.3f}s total, {eager_warm_average:.3f}s avg")
    if exists(eager.peak_allocated_mib) and exists(eager.peak_reserved_mib):
        print(f"  eager peak CUDA memory: {eager.peak_allocated_mib:.1f} MiB allocated, {eager.peak_reserved_mib:.1f} MiB reserved")

    for compiled in (result for result in results if result.name != "eager"):
        name = compiled.name
        compiled_seconds = compiled.wall_seconds
        compiled_hot_path = compiled.hot_path_seconds
        compiled_first = compiled.first_call_seconds
        compiled_warm = compiled.warm_seconds
        compiled_warm_average = compiled.warm_average_seconds
        warm_speedup = eager_warm / compiled_warm if compiled_warm > 0 else float("inf")
        compile_overhead = max(compiled_first - eager_first, 0.)

        break_even_calls = None
        if (
            exists(eager_warm_average) and
            exists(compiled_warm_average) and
            compiled_warm_average < eager_warm_average
        ):
            break_even_calls = compile_overhead / (eager_warm_average - compiled_warm_average)

        print(f"\n  {name}:")
        print(f"    steady-state warm runtime speedup: {warm_speedup:.3f}x")
        print(f"    compiled warm runtime: {compiled_warm:.3f}s total, {compiled_warm_average:.3f}s avg")
        print(f"    compiled first-call time: {compiled_first:.3f}s")
        print(f"    estimated compile overhead vs eager first call: {compile_overhead:.3f}s")
        if exists(compiled.peak_allocated_mib) and exists(compiled.peak_reserved_mib):
            print(f"    peak CUDA memory: {compiled.peak_allocated_mib:.1f} MiB allocated, {compiled.peak_reserved_mib:.1f} MiB reserved")

        if exists(break_even_calls):
            print(f"    estimated break-even warm calls: {break_even_calls:.1f}")
        else:
            print("    estimated break-even warm calls: never (compiled warm runtime is not faster)")

        print(f"    total elapsed, including setup and compile: {compiled_seconds:.3f}s")
        print(f"    measured hot-path elapsed, including compile: {compiled_hot_path:.3f}s")

    print(f"\n  total eager elapsed, including setup: {eager_seconds:.3f}s")
    print(f"  measured hot-path elapsed, eager: {eager_hot_path:.3f}s")


def main(
    env_name = "HalfCheetah-v4",
    num_loops = 500,
    rollouts_per_loop = 1,
    num_envs = 32,
    max_timesteps = 1000,
    replay_size = 256,
    seed = 42,
    cpu = False,
    obs_dim = 17,
    num_latent_tokens = 4,
    dim_latent = 32,
    model_dim = 128,
    attn_heads = 4,
    attn_dim_head = 32,
    depth = 3,
    time_block_every = 2,
    final_special_cross_attn = False,
    reward_encoder_type: Literal["symexp_two_hot", "hl_gauss"] = "hl_gauss",
    prob_shortcut_train: float | None = None,
    world_model_batch_size = 32,
    world_model_train_steps = 13,
    world_model_train_sequence_length = 200,
    world_model_learning_rate = 3e-4,
    imagination_batch_size = 128,
    imagination_horizon = 32,
    imagination_prompt_length = 8,
    imagination_prompt_probability = 1.,
    imagination_train_steps = 3,
    imagination_generate_steps = 4,
    imagination_use_time_cache = True,
    agent_learning_rate = 3e-4,
    use_muon_optimizer = True,
    optimizer_weight_decay = 0.01,
    objective: Literal["ppo", "pmpo", "spo"] = "pmpo",
    pmpo_pos_to_neg_weight = 0.5,
    pmpo_kl_div_loss_weight = 0.3,
    use_delight_gating = False,
    agent_predicts_state = True,
    agent_state_pred_loss_weight = 0.1,
    pretrain_tokenizer_steps = 1000,
    pretrain_tokenizer_observations = 8192,
    tokenizer_batch_size = 256,
    tokenizer_learning_rate = 3e-4,
    tokenizer_eval_every = 10,
    tokenizer_eval_batch_size = 2048,
    max_grad_norm = 0.5,
    use_tensorboard = True,
    log_dir = "runs/halfcheetah_imagination",
    checkpoint_folder = "checkpoints_halfcheetah_imagination",
    checkpoint_every = 25,
    checkpoint_path: str | None = None,
    clear_log_dir = False,
    unique_log_dir = True,
    compile = False,
    compile_world_model = False,
    compile_generate = False,
    compile_learn = False,
    compile_backend = "inductor",
    compile_mode: str | None = "reduce_overhead",
    compile_fullgraph = False,
    compile_dynamic: bool | None = None,
    track_compile_performance = False,
    allow_tf32 = True,
    benchmark_compile = False,
    benchmark_num_loops = 2,
    benchmark_num_envs = 1,
    benchmark_max_timesteps = 8,
    benchmark_require_cuda = True,
    benchmark_preset: Literal["smoke", "perf"] = "smoke",
    require_cuda = False,
    return_compile_timings = False,
):
    compile_mode = normalize_compile_mode(compile_mode)

    if benchmark_compile:
        return run_compile_benchmark(
            locals(),
            benchmark_num_loops = benchmark_num_loops,
            benchmark_num_envs = benchmark_num_envs,
            benchmark_max_timesteps = benchmark_max_timesteps,
            benchmark_require_cuda = benchmark_require_cuda,
            benchmark_preset = benchmark_preset,
        )

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")

    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required for this run, but no CUDA device is available")

    if allow_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    run_log_dir = resolve_log_dir(log_dir, checkpoint_path) if unique_log_dir else Path(log_dir)

    if clear_log_dir:
        shutil.rmtree(run_log_dir, ignore_errors = True)

    writer = SummaryWriter(str(run_log_dir)) if use_tensorboard else None

    tokenizer = ObservationTokenizer(
        obs_dim = obs_dim,
        num_latent_tokens = num_latent_tokens,
        dim_latent = dim_latent,
    ).to(device)

    tokenizer_loss = None

    if not exists(checkpoint_path):
        pretrain_obs = collect_random_observations(
            env_name,
            num_steps = max(1, pretrain_tokenizer_observations // num_envs),
            num_envs = num_envs,
            seed = seed,
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
    )

    single_action_space = getattr(env, "single_action_space", env.action_space)
    action_dim = int(np.prod(single_action_space.shape))
    action_lows = np.asarray(single_action_space.low)
    action_highs = np.asarray(single_action_space.high)

    assert np.allclose(action_lows, -1.) and np.allclose(action_highs, 1.), (
        "this script expects the native policy target range to match the env action range [-1, 1]"
    )

    reward_range = (-4., 4.)
    value_range = (-10., 10.)

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
        num_discrete_actions = 0,
        num_continuous_actions = action_dim,
        continuous_dist_type = "beta",
        continuous_target_action_range = (-1., 1.),
        reward_encoder_type = reward_encoder_type,
        reward_encoder_kwargs = dict(reward_range = reward_range, num_bins = 51, sigma_to_bin_ratio = 0.75, min_max_value_on_bin_center = True, use_symlog = True),
        value_encoder_kwargs = dict(reward_range = value_range, num_bins = 51, sigma_to_bin_ratio = 0.75, min_max_value_on_bin_center = True, use_symlog = True),
        predict_terminals = True,
        continuous_action_loss_weight = 0.,
        discrete_action_loss_weight = 0.,
        agent_predicts_state = agent_predicts_state,
        agent_state_pred_loss_weight = agent_state_pred_loss_weight,
        prob_shortcut_train = prob_shortcut_train,
        gae_discount_factor = 0.99,
        pmpo_pos_to_neg_weight = pmpo_pos_to_neg_weight,
        pmpo_kl_div_loss_weight = pmpo_kl_div_loss_weight,
        ppo_eps_clip = 0.2,
        normalize_advantages = True,
        policy_entropy_weight = 0.01,
        use_loss_normalization = False,
        attn_heads = attn_heads,
        attn_dim_head = attn_dim_head,
        policy_head_mlp_depth = 2,
        value_head_mlp_depth = 2,
    ).to(device)

    world_params = world_model.world_model_parameters()
    agent_params = world_model.agent_parameters()

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

    policy_prior = FrozenPolicyPrior(world_model).to(device) if objective == "pmpo" else None
    compile_world_model = compile or compile_world_model
    compile_generate = compile or compile_generate
    compile_learn = compile or compile_learn
    track_compile_performance = track_compile_performance or compile_world_model or compile_generate or compile_learn

    compile_runtime = build_compile_runtime(
        world_model,
        device = device,
        compile_world_model = compile_world_model,
        compile_generate = compile_generate,
        compile_learn = compile_learn,
        compile_backend = compile_backend,
        compile_mode = compile_mode,
        compile_fullgraph = compile_fullgraph,
        compile_dynamic = compile_dynamic,
        track_compile_performance = track_compile_performance,
        imagination_generate_steps = imagination_generate_steps,
        imagination_use_time_cache = imagination_use_time_cache,
        objective = objective,
        use_delight_gating = use_delight_gating,
        store_old_action_unembeds = objective == "pmpo" and not exists(policy_prior),
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

    print(f"training {env_name} from {obs_dim} raw observations on {device}")
    print(f"tensorboard log dir: {run_log_dir.absolute()}" if use_tensorboard else "tensorboard disabled")
    if compile_world_model or compile_generate or compile_learn:
        compiled_paths = [
            name
            for name, enabled in (
                ("world_model_loss", compile_world_model),
                ("imagination_generate", compile_generate),
                ("imagination_learn", compile_learn),
            )
            if enabled
        ]
        print(f"torch.compile enabled for: {', '.join(compiled_paths)}")
        if device.type == "cuda" and compile_mode_uses_cuda_graphs(compile_mode):
            cuda_graph_paths = [
                name
                for name, enabled in (
                    ("imagination_generate", compile_generate),
                )
                if enabled
            ]

            if len(cuda_graph_paths) > 0:
                print(f"CUDA Graph capture enabled for: {', '.join(cuda_graph_paths)}")
                print("CUDA Graph capture disabled for autograd training losses")
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
                start_episode_count = get_episode_count(env)

                exp = world_model.interact_with_env(
                    env,
                    seed = seed if loop == 0 and rollout_idx == 0 else None,
                    max_timesteps = max_timesteps,
                    env_is_vectorized = True,
                    store_agent_embed = False,
                    store_old_action_unembeds = False,
                    obs_to_latents_fn = obs_to_latents_fn(tokenizer),
                )

                if tokenizer_eval_every > 0 and divisible_by(loop, tokenizer_eval_every):
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
                        max_timesteps = max_timesteps + 10
                    )

                for single_exp in exp.unbind():
                    data, meta = single_exp.cpu().to_buffer_dict()
                    data, meta = tree_map_tensor(lambda t: rearrange(t, '1 ... -> ...'), (data, meta))
                    replay.store_episode(**data, **meta)

                rollout_horizon_returns.extend(exp.episode_return.detach().cpu().tolist())

                rollout_steps = exp.rewards.shape[1]
                has_bootstrap_padding = exists(exp.is_truncated) and exists(exp.terminals) and (exp.is_truncated & ~exp.terminals).any()
                if has_bootstrap_padding:
                    rollout_steps -= 1

                env_step += rollout_steps * exp.rewards.shape[0]

                completed_returns, completed_lengths = get_completed_episode_stats(env, start_episode_count)

                for episode_return, episode_length in zip(completed_returns, completed_lengths):
                    episodic_returns.append(float(episode_return))
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
            max_grad_norm = max_grad_norm,
            global_step = wm_step,
            writer = writer,
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
            generate_steps = imagination_generate_steps,
            max_grad_norm = max_grad_norm,
            objective = objective,
            use_delight_gating = use_delight_gating,
            global_step = imagination_step,
            writer = writer,
        )

        postfix_return = avg_return if exists(avg_return) else avg_horizon_return
        postfix = {"return": f"{postfix_return:.1f}", "replay": len(replay)}
        if wm_metrics:
            postfix["wm"] = f"{wm_metrics['world_model/loss']:.2f}"
        if imagination_metrics:
            postfix["dream"] = f"{imagination_metrics['imagination/raw_reward_sum_mean']:.1f}"
        pbar.set_postfix(postfix)

        if checkpoint_every > 0 and divisible_by(loop + 1, checkpoint_every):
            save_checkpoint(
                Path(checkpoint_folder) / f"loop_{loop + 1}.pt",
                loop = loop,
                tokenizer = tokenizer,
                world_model = world_model,
                world_optimizer = world_optimizer,
                agent_optimizer = agent_optimizer,
            )

    env.close()
    print_compile_timing_report(compile_runtime.timings)

    if exists(writer):
        writer.flush()
        writer.close()

    if return_compile_timings:
        return compile_runtime.timings

if __name__ == "__main__":
    fire.Fire(main)
