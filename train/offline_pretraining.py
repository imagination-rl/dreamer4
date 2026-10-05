"""Offline data collection and pretraining (notes/pretraining_impl_plan.md).

The path is inactive unless offline_pretraining is true. It has three stages:

- collect: download a pinned CleanRL TD3 expert. Use it to collect expert,
  expert + noise, and random episodes. Save the raw episodes to disk.
- pretrain: train the tokenizer and the world model on the offline data only.
  Save a pretrained checkpoint and a latent replay for rehearsal.
- rl: load the pretrained checkpoint and run the online loop of train_gym.py.

Each stage writes a complete artifact. A later stage reads only that artifact,
so the stages can run in separate jobs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_einops_utils import temp_eval

from einops import rearrange
from memmap_replay_buffer.replay_buffer import ReplayBuffer

from dreamer4.dreamer4 import Actions, Experience, tree_map_tensor

from .audit_ablations import (
    HALF_CHEETAH_ROOT_ANGLE_INDEX,
    HALF_CHEETAH_X_VELOCITY_INDEX,
    _wrap_angle,
    env_to_native_actions,
    native_to_env_actions,
)

OFFLINE_STAGES = ("collect", "pretrain", "rl")
# past_runs: episodes from the replays of earlier online runs (the policies of those runs, from loop 0 to the end).
# expert_scaled: the expert with its actions multiplied by one scale for each episode (closed loop). It gives actions of
# moderate size at speed (scale 0.4 to 0.7: mean |a| 0.33 to 0.58, 1.7 to 5.9 m/s,
# report/9-30-26-past-runs/scripts/probe_scaled_expert.py), which the other modes do not give. The action shape stays
# the expert's bang-bang shape (about 62 % of the dims at 0.95 * scale or more), and each scale gives one steady gait.
# bang_bang: random actions at the bounds of the action range. Each action dim keeps its sign for a random number of
# steps (geometric, mean = the hold steps of the episode). It gives saturated actions at low speed (|vx| p90 0.5-1.8 m/s)
# and flips from upright states (first flip at step 50-120, then 70-90 % flipped steps). The other modes give
# saturated actions only at speed (expert, expert_noise), which teaches "saturated means fast"
# (report/10-1-26-realistic-data-analysis).
OFFLINE_MODES = ("expert", "expert_noise", "random", "past_runs", "expert_scaled", "bang_bang")
OFFLINE_MODE_IDS = {mode: index for index, mode in enumerate(OFFLINE_MODES)}
# The modes of the first datasets. A collection config lists a later mode only if its fraction is above 0,
# so the datasets collected before that mode existed still agree with their configs.
FIRST_OFFLINE_MODES = ("expert", "expert_noise", "random")
PAST_RUN_SOURCE_FILE = "past_run_source.npy"
ACTION_SCALE_FILE = "action_scale.npy"
DEFAULT_DATA_MIX = {"expert": 0.3, "expert_noise": 0.4, "random": 0.3}

OFFLINE_DATASET_FORMAT_VERSION = 1
EXPERT_VALIDATION_EPISODES = 5
# Expert validation episodes use seeds far from the dataset seeds.
EXPERT_VALIDATION_SEED_OFFSET = 1_000_000
X_VELOCITY_PERCENTILES = (5, 25, 50, 75, 95)

# Beta actions at exactly 0 or 1 have an infinite log density.
NATIVE_ACTION_EPS = 1e-4

ENCODE_CHUNK_EPISODES = 64
TOKENIZER_EVAL_CHUNK_ROWS = 65536
WORLD_MODEL_VALIDATION_BATCHES = 4

TOKENIZER_FILE = "tokenizer.pt"
PRETRAINED_FILE = "pretrained.pt"
STEP_CHECKPOINT_PREFIX = "pretrained_step_"
SPLIT_FILE = "split.json"
REPLAY_DIR = "offline_replay"
VALIDATION_REPLAY_DIR = "offline_replay_val"
REPLAY_MARKER_FILE = "offline_replay.json"

# sha256 of the pinned expert files. The check detects a corrupt download.
KNOWN_EXPERT_SHA256 = {
    (
        "cleanrl/HalfCheetah-v4-td3_continuous_action-seed1",
        "85477545d6db77ae3c7ca8e4e0b48f515d4c1432",
        "td3_continuous_action.cleanrl_model",
    ): "0000ee726a5b36f40bb75f48bd9cdb0d0869bcdc77ecc1150edc3f85b26c5488",
}

# The config values that decide the tokenizer shape, the world-model shape,
# and the world-model targets. A pretrained checkpoint serves only runs with
# the same values.
PRETRAIN_SIGNATURE_KEYS = (
    "env_name",
    "obs_dim",
    "num_latent_tokens",
    "dim_latent",
    "model_dim",
    "attn_heads",
    "attn_dim_head",
    "depth",
    "time_block_every",
    "final_special_cross_attn",
    "old_state_tokenizer",
    "old_dynamics_ff",
    "old_embedding_init",
    "old_continuous_action_encoder",
    "add_action_embed_to_agent_token",
    "continuous_dist_type",
    "policy_all_tokens",
    "agent_predicts_state",
    "reward_encoder_type",
    "reward_encoder_num_bins",
    "reward_range",
    "reward_encoder_sigma_to_bin_ratio",
    "reward_min_max_value_on_bin_center",
    "reward_use_symlog",
    "reward_decode_symlog_expected_raw",
    "reward_gradient_into_transformer_scale",
    "old_reward_head",
    "old_terminal_head",
    "old_reward_shifting",
    "prob_shortcut_train",
    "use_paper_aligned_shortcut_objective",
    "use_dynamics_loss_normalization",
    "scale_reward_gradient_into_transformer",
)

# Signature keys added later. A key with the value None (the default) is left out of the signature,
# so pretrained checkpoints saved before the key existed still agree with default configs.
OPTIONAL_PRETRAIN_SIGNATURE_KEYS = (
    "terminal_head_mlp_dim",
)

# World-model state that only the agent uses. These keys are not part of a
# pretrained world model, so one pretrained world model serves RL runs with
# different value heads. critic_state_embedder feeds only the value head.
AGENT_STATE_MODULES = ("value_encoder", "actor_critic_value_encoder", "critic_state_embedder")
AGENT_STATE_BUFFERS = ("ema_returns_mean", "ema_returns_var")


def exists(v):
    return v is not None


def _train_gym():
    # train_gym imports this module at load time, so import it at call time.
    from . import train_gym
    return train_gym


def round_half_up(x: float) -> int:
    # Python round() rounds 0.5 to the even integer. Batch splits need a stable rule.
    return int(math.floor(x + 0.5))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def json_normalize(value):
    # Tuples become lists, so YAML, Fire, and saved values compare equal.
    return json.loads(json.dumps(value))


def save_torch_atomic(obj, path: Path):
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    try:
        torch.save(obj, str(tmp_path))
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok = True)


class PrefixedScalarWriter:
    """Add a tag prefix, so pretrain metrics do not share tags with the online loop."""

    def __init__(self, writer, prefix: str):
        self.writer = writer
        self.prefix = prefix

    def add_scalar(self, tag: str, value, step: int):
        self.writer.add_scalar(self.prefix + tag, value, step)


# config validation

def normalize_offline_stages(stages) -> tuple[str, ...]:
    if isinstance(stages, str):
        stages = [stages]

    if not isinstance(stages, (list, tuple)) or len(stages) == 0:
        raise ValueError(f"offline_stages must be a non-empty ordered subset of {list(OFFLINE_STAGES)}, got {stages!r}")

    unknown = [stage for stage in stages if stage not in OFFLINE_STAGES]
    if unknown:
        raise ValueError(f"offline_stages contains unknown stages {unknown}; choose from {list(OFFLINE_STAGES)}")

    positions = [OFFLINE_STAGES.index(stage) for stage in stages]
    if any(later <= earlier for earlier, later in zip(positions, positions[1:])):
        raise ValueError(
            f"offline_stages must be an ordered subset of {list(OFFLINE_STAGES)} without repeats, got {list(stages)}"
        )

    return tuple(stages)


def normalize_data_mix(data_mix: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(data_mix, Mapping):
        raise ValueError(f"offline_data_mix must be a mapping, got {type(data_mix).__name__}")

    unknown = sorted(set(data_mix) - set(OFFLINE_MODES))
    if unknown:
        raise ValueError(f"offline_data_mix contains unknown modes {unknown}; choose from {list(OFFLINE_MODES)}")

    mix = {mode: float(data_mix.get(mode, 0.)) for mode in OFFLINE_MODES}

    if any(fraction < 0. for fraction in mix.values()):
        raise ValueError(f"offline_data_mix fractions must not be negative, got {mix}")

    if not math.isclose(sum(mix.values()), 1., abs_tol = 1e-6):
        raise ValueError(f"offline_data_mix fractions must add up to 1, got {mix} (sum {sum(mix.values())})")

    return mix


def validate_offline_config(
    *,
    offline_pretraining: bool,
    small_model_only: bool,
    offline_stages,
    offline_pretrained_checkpoint: str | None,
    checkpoint_path: str | None,
    offline_rehearsal_fraction: float,
    offline_data_mix: Mapping[str, float],
    offline_expert_noise_stds: Sequence[float],
    offline_data_steps: int,
    offline_data_num_envs: int,
    offline_validation_fraction: float,
    offline_world_model_steps: int,
    offline_world_model_finetune_steps: int,
    offline_pretrain_checkpoint_every: int,
    continuous_dist_type: str,
    offline_past_run_replays: Sequence[str] = (),
    offline_past_run_return_buckets: Sequence[Sequence[float]] = (),
    offline_expert_action_scales: Sequence[float] = (),
    offline_expert_scale_change_steps: int = 0,
    offline_expert_scaled_noise_std: float = 0.,
    offline_past_run_flip_episode_fraction: float = 0.,
    offline_bang_bang_hold_steps: Sequence[float] = (),
) -> tuple[str, ...]:
    """Return the selected stages, or an empty tuple if the offline path is off."""

    if not offline_pretraining:
        return ()

    if small_model_only:
        raise ValueError("offline_pretraining requires small_model_only=false")

    stages = normalize_offline_stages(offline_stages)

    if "rl" in stages and "pretrain" not in stages and not exists(offline_pretrained_checkpoint):
        raise ValueError("the rl offline stage requires the pretrain stage or offline_pretrained_checkpoint")

    if exists(offline_pretrained_checkpoint) and "pretrain" in stages:
        raise ValueError("offline_pretrained_checkpoint cannot be set together with the pretrain offline stage")

    if "pretrain" in stages and exists(checkpoint_path):
        raise ValueError("the pretrain offline stage cannot be combined with checkpoint_path")

    if not 0. <= offline_rehearsal_fraction < 1.:
        raise ValueError(f"offline_rehearsal_fraction must be in [0, 1), got {offline_rehearsal_fraction}")

    mix = normalize_data_mix(offline_data_mix)

    if mix["expert_noise"] > 0.:
        stds = list(offline_expert_noise_stds or ())
        if len(stds) == 0 or any(float(std) <= 0. for std in stds):
            raise ValueError(f"offline_expert_noise_stds must be a non-empty list of positive values, got {offline_expert_noise_stds!r}")

    if mix["past_runs"] > 0. and len(list(offline_past_run_replays or ())) == 0:
        raise ValueError("offline_data_mix has past_runs episodes, so offline_past_run_replays must list replay folders")

    if mix["expert_scaled"] > 0.:
        scales = list(offline_expert_action_scales or ())
        if len(scales) == 0 or any(not 0. < float(scale) <= 1. for scale in scales):
            raise ValueError(f"offline_expert_action_scales must be a non-empty list of values in (0, 1], got {offline_expert_action_scales!r}")
        if int(offline_expert_scale_change_steps) < 0 or float(offline_expert_scaled_noise_std) < 0.:
            raise ValueError("offline_expert_scale_change_steps and offline_expert_scaled_noise_std must not be negative")
        if int(offline_expert_scale_change_steps) > 0 and min(scales) == max(scales):
            raise ValueError("offline_expert_scale_change_steps > 0 draws scales from [min, max] of offline_expert_action_scales; give two different values")

    if mix["bang_bang"] > 0.:
        holds = list(offline_bang_bang_hold_steps or ())
        if len(holds) == 0 or any(float(hold) < 1. for hold in holds):
            raise ValueError(f"offline_bang_bang_hold_steps must be a non-empty list of values of 1 or more, got {offline_bang_bang_hold_steps!r}")

    check_return_buckets(offline_past_run_return_buckets or ())

    if not 0. <= float(offline_past_run_flip_episode_fraction) < 1.:
        raise ValueError(f"offline_past_run_flip_episode_fraction must be in [0, 1), got {offline_past_run_flip_episode_fraction}")

    if float(offline_past_run_flip_episode_fraction) > 0. and (mix["past_runs"] <= 0. or len(list(offline_past_run_return_buckets or ())) == 0):
        raise ValueError("offline_past_run_flip_episode_fraction > 0 needs past_runs episodes and offline_past_run_return_buckets")

    if offline_data_steps <= 0:
        raise ValueError(f"offline_data_steps must be positive, got {offline_data_steps}")

    if offline_data_num_envs <= 0:
        raise ValueError(f"offline_data_num_envs must be positive, got {offline_data_num_envs}")

    if not 0. <= offline_validation_fraction < 1.:
        raise ValueError(f"offline_validation_fraction must be in [0, 1), got {offline_validation_fraction}")

    if offline_world_model_steps < 0 or offline_world_model_finetune_steps < 0:
        raise ValueError("offline_world_model_steps and offline_world_model_finetune_steps must not be negative")

    if offline_pretrain_checkpoint_every < 0:
        raise ValueError(f"offline_pretrain_checkpoint_every must not be negative, got {offline_pretrain_checkpoint_every}")

    # The offline actions are converted to the native [0, 1] Beta range.
    if continuous_dist_type != "beta":
        raise ValueError(f"offline_pretraining requires continuous_dist_type='beta', got {continuous_dist_type!r}")

    return stages


# expert

class CleanRLTD3Actor(nn.Module):
    """The actor of CleanRL td3_continuous_action.py."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("action_scale", torch.ones(1, action_dim))
        self.register_buffer("action_bias", torch.zeros(1, action_dim))

    def forward(self, obs: Tensor) -> Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def load_expert_actor(path: str | Path, *, obs_dim: int, action_dim: int) -> CleanRLTD3Actor:
    """Load the actor from a CleanRL `torch.save((actor, qf1, qf2))` file."""

    package = torch.load(str(path), map_location = "cpu", weights_only = True)

    if not isinstance(package, (tuple, list)) or len(package) == 0 or not isinstance(package[0], Mapping):
        raise ValueError(f"{path} is not a CleanRL TD3 model file: expected a tuple of state dicts")

    state = dict(package[0])
    required = {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc_mu.weight", "fc_mu.bias", "action_scale", "action_bias"}
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError(f"{path} actor state dict does not have the keys {missing}")

    hidden_dim, file_obs_dim = state["fc1.weight"].shape
    file_action_dim = state["fc_mu.weight"].shape[0]

    if (file_obs_dim, file_action_dim) != (obs_dim, action_dim):
        raise ValueError(
            f"{path} expert has obs dim {file_obs_dim} and action dim {file_action_dim}; "
            f"the env has obs dim {obs_dim} and action dim {action_dim}"
        )

    for key in ("action_scale", "action_bias"):
        state[key] = state[key].reshape(1, action_dim)

    actor = CleanRLTD3Actor(obs_dim, action_dim, hidden_dim)
    actor.load_state_dict(state)
    return actor.eval().requires_grad_(False)


def download_expert(
    *,
    repo: str,
    revision: str,
    filename: str,
    cache_dir: str | Path,
    expert_path: str | None = None,
) -> tuple[Path, str]:
    """Return the path and the sha256 of the expert file. Download it if it is not cached."""

    if exists(expert_path):
        path = Path(expert_path)
        if not path.is_file():
            raise FileNotFoundError(f"offline_expert_path does not exist: {path}")
        return path, sha256_file(path)

    path = Path(cache_dir) / repo / revision / filename
    sha_path = path.with_name(path.name + ".sha256")
    expected = KNOWN_EXPERT_SHA256.get((repo, revision, filename))

    if path.is_file() and sha_path.is_file():
        digest = sha256_file(path)
        if digest == sha_path.read_text().strip() and (not exists(expected) or digest == expected):
            return path, digest
        print(f"expert cache does not agree with its sha256 file, downloading again: {path}")

    path.parent.mkdir(parents = True, exist_ok = True)
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    print(f"downloading expert {url}")

    try:
        with urllib.request.urlopen(url, timeout = 60) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)

        digest = sha256_file(tmp_path)
        if exists(expected) and digest != expected:
            raise ValueError(f"expert download {url} has sha256 {digest}, expected {expected}")

        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok = True)

    sha_path.write_text(digest + "\n")
    return path, digest


@torch.no_grad()
def expert_actions(actor: CleanRLTD3Actor | None, obs: np.ndarray, action_dim: int) -> np.ndarray:
    if not exists(actor):
        return np.zeros((obs.shape[0], action_dim), dtype = np.float32)

    # The actor runs on its own device. The env data stays in numpy on the CPU.
    device = actor.fc1.weight.device
    return actor(torch.from_numpy(obs).to(device)).cpu().numpy()


def run_episode_batch(
    env,
    actor: CleanRLTD3Actor | None,
    *,
    modes: np.ndarray,
    noise_stds: np.ndarray,
    rng: np.random.Generator,
    seed: int,
    action_range: tuple[float, float],
    max_episode_steps: int,
    action_dim: int,
    action_scales: np.ndarray | None = None,
    scale_change_steps: int = 0,
    scale_range: tuple[float, float] | None = None,
    scale_rng: np.random.Generator | None = None,
    hold_steps: np.ndarray | None = None,
    bang_rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Run one episode in each env of a vector env. Record each env until its first episode ends.

    modes holds one OFFLINE_MODE_IDS value per env. noise_stds holds the
    noise std of each env in units of half the action range. action_scales
    holds the factor of the expert actions of each expert_scaled env.
    With scale_change_steps > 0, each expert_scaled env draws a new factor from
    scale_range (uniform, with scale_rng) every scale_change_steps steps. An
    expert_scaled env with a noise std above 0 also gets the Gaussian noise
    (after the factor). hold_steps holds the mean hold of each bang_bang env:
    each step, each action dim of the env draws a new sign (with bang_rng) with
    probability 1 / hold_steps.
    """

    num_envs = env.num_envs
    low, high = action_range
    half_range = (high - low) / 2.

    obs, _ = env.reset(seed = seed)
    obs = np.asarray(obs, dtype = np.float32)
    obs_dim = obs.shape[-1]

    observations = np.zeros((num_envs, max_episode_steps + 1, obs_dim), dtype = np.float32)
    actions = np.zeros((num_envs, max_episode_steps, action_dim), dtype = np.float32)
    rewards = np.zeros((num_envs, max_episode_steps), dtype = np.float32)
    terminated = np.zeros(num_envs, dtype = np.bool_)
    truncated = np.zeros(num_envs, dtype = np.bool_)
    lens = np.zeros(num_envs, dtype = np.int32)
    done = np.zeros(num_envs, dtype = np.bool_)

    observations[:, 0] = obs
    is_noise = modes == OFFLINE_MODE_IDS["expert_noise"]
    is_random = modes == OFFLINE_MODE_IDS["random"]
    is_scaled = modes == OFFLINE_MODE_IDS["expert_scaled"]
    # Noise for the expert_scaled envs only if their std is above 0, so the older datasets do not change.
    gets_noise = is_noise | (is_scaled & (noise_stds > 0.))
    if is_scaled.any():
        action_scales = np.array(action_scales, dtype = np.float32)
    is_bang = modes == OFFLINE_MODE_IDS["bang_bang"]
    if is_bang.any():
        # A separate generator, so the other modes get the same values with or without bang_bang envs.
        signs = bang_rng.choice(np.array([-1., 1.], dtype = np.float32), size = (num_envs, action_dim))
        sign_change = 1. / np.asarray(hold_steps, dtype = np.float64)[:, None]

    for t in range(max_episode_steps):
        if is_scaled.any() and scale_change_steps > 0 and t > 0 and t % scale_change_steps == 0:
            # A separate generator: the main generator gives the same values with or without these draws.
            new_scales = scale_rng.uniform(*scale_range, size = num_envs).astype(np.float32)
            action_scales = np.where(is_scaled, new_scales, action_scales)

        action = expert_actions(actor, obs, action_dim)
        # The random generator gives the same values with or without expert_scaled envs.
        if is_scaled.any():
            action = np.where(is_scaled[:, None], action * action_scales[:, None], action)
        noise = rng.normal(0., 1., size = action.shape) * (noise_stds[:, None] * half_range)
        action = np.where(gets_noise[:, None], action + noise, action)
        action = np.where(is_random[:, None], rng.uniform(low, high, size = action.shape), action)
        if is_bang.any():
            if t > 0:
                change = bang_rng.random((num_envs, action_dim)) < sign_change
                signs = np.where(change, bang_rng.choice(np.array([-1., 1.], dtype = np.float32), size = signs.shape), signs)
            action = np.where(is_bang[:, None], np.where(signs > 0., high, low), action)
        action = np.clip(action, low, high).astype(np.float32)

        next_obs, reward, term, trunc, _ = env.step(action)
        next_obs = np.asarray(next_obs, dtype = np.float32)
        term = np.asarray(term, dtype = np.bool_)
        trunc = np.asarray(trunc, dtype = np.bool_)

        # The vector env resets a finished env on the next step. Its later data is not recorded.
        active = ~done
        observations[active, t + 1] = next_obs[active]
        actions[active, t] = action[active]
        rewards[active, t] = np.asarray(reward, dtype = np.float32)[active]
        lens[active] += 1
        terminated |= term & active
        truncated |= trunc & active
        done |= term | trunc

        obs = next_obs

        if done.all():
            break

    # An env without a time limit stops at max_episode_steps, as interact_with_env does.
    truncated |= ~done

    return dict(
        obs = observations,
        actions = actions,
        rewards = rewards,
        terminated = terminated,
        truncated = truncated,
        lens = lens,
    )


def env_max_episode_steps(env_name: str) -> int:
    max_episode_steps = gym.spec(env_name).max_episode_steps

    if not exists(max_episode_steps) or max_episode_steps <= 0:
        raise ValueError(f"{env_name} has no time limit; the offline collector needs max_episode_steps")

    return int(max_episode_steps)


def validate_expert(
    actor: CleanRLTD3Actor,
    *,
    env_name: str,
    action_range: tuple[float, float],
    action_dim: int,
    min_return: float,
    seed: int,
    num_episodes: int = EXPERT_VALIDATION_EPISODES,
) -> float:
    """Run deterministic expert episodes. Raise an error if the mean return is too low."""

    env = _train_gym().make_env(
        env_name,
        seed,
        vectorized = True,
        num_envs = num_episodes,
        env_vectorization_mode = "sync",
    )

    try:
        out = run_episode_batch(
            env,
            actor,
            modes = np.full(num_episodes, OFFLINE_MODE_IDS["expert"], dtype = np.int8),
            noise_stds = np.zeros(num_episodes, dtype = np.float32),
            rng = np.random.default_rng(seed),
            seed = seed,
            action_range = action_range,
            max_episode_steps = env_max_episode_steps(env_name),
            action_dim = action_dim,
        )
    finally:
        env.close()

    returns = out["rewards"].sum(axis = 1)
    mean_return = float(returns.mean())

    print(f"expert validation: mean return {mean_return:.1f} over {num_episodes} episodes ({np.round(returns, 1).tolist()})")

    if mean_return < min_return:
        raise ValueError(
            f"expert mean return {mean_return:.1f} is below offline_expert_min_return={min_return}. "
            "Check the expert file, the observation order, and the env version."
        )

    return mean_return


# collect

def offline_dataset_dir(data_dir: str | Path, env_name: str, dataset_name: str) -> Path:
    return Path(data_dir) / env_name / dataset_name


def collection_config(
    *,
    env_name: str,
    dataset_name: str,
    data_steps: int,
    data_mix: Mapping[str, float],
    noise_stds: Sequence[float],
    data_seed: int,
    num_envs: int,
    expert_repo: str,
    expert_revision: str,
    expert_filename: str,
    expert_path: str | None,
    past_run_replays: Sequence[str] = (),
    past_run_return_buckets: Sequence[Sequence[float]] = (),
    expert_action_scales: Sequence[float] = (),
    expert_scale_change_steps: int = 0,
    expert_scaled_noise_std: float = 0.,
    past_run_flip_episode_fraction: float = 0.,
    bang_bang_hold_steps: Sequence[float] = (),
) -> dict[str, Any]:
    """The config values that decide the collected data."""

    mix = normalize_data_mix(data_mix)
    uses_expert = mix["expert"] + mix["expert_noise"] + mix["expert_scaled"] > 0.
    uses_past_runs = mix["past_runs"] > 0.
    listed_mix = {mode: fraction for mode, fraction in mix.items() if mode in FIRST_OFFLINE_MODES or fraction > 0.}
    past_runs = dict(past_run_replays = [str(path) for path in past_run_replays]) if uses_past_runs else {}
    # Only a config with return buckets lists them, so a dataset collected before the buckets keeps its config.
    if uses_past_runs and len(past_run_return_buckets) > 0:
        past_runs["past_run_return_buckets"] = [[float(low), float(high)] for low, high in past_run_return_buckets]
    # Only a config that uses it lists the flip share, so the earlier datasets keep their configs.
    if uses_past_runs and float(past_run_flip_episode_fraction) > 0.:
        past_runs["past_run_flip_episode_fraction"] = float(past_run_flip_episode_fraction)
    # As for past_runs: only a config with expert_scaled episodes lists the scales.
    if mix["expert_scaled"] > 0.:
        past_runs["expert_action_scales"] = [float(scale) for scale in expert_action_scales]
        # Only a config that uses them lists these, so the earlier expert_scaled datasets keep their configs.
        if int(expert_scale_change_steps) > 0:
            past_runs["expert_scale_change_steps"] = int(expert_scale_change_steps)
        if float(expert_scaled_noise_std) > 0.:
            past_runs["expert_scaled_noise_std"] = float(expert_scaled_noise_std)
    # Only a config with bang_bang episodes lists the hold steps.
    if mix["bang_bang"] > 0.:
        past_runs["bang_bang_hold_steps"] = [float(hold) for hold in bang_bang_hold_steps]

    return json_normalize(dict(
        format_version = OFFLINE_DATASET_FORMAT_VERSION,
        env_name = env_name,
        dataset_name = dataset_name,
        data_steps = int(data_steps),
        data_mix = listed_mix,
        **past_runs,
        noise_stds = [float(std) for std in noise_stds] if mix["expert_noise"] > 0. else [],
        data_seed = int(data_seed),
        num_envs = int(num_envs),
        expert = dict(
            repo = expert_repo if not exists(expert_path) else None,
            revision = expert_revision if not exists(expert_path) else None,
            filename = expert_filename if not exists(expert_path) else None,
            path_sha256 = sha256_file(expert_path) if exists(expert_path) else None,
        ) if uses_expert else None,
    ))


def mode_statistics(
    env_name: str,
    *,
    obs: np.ndarray,
    rewards: np.ndarray,
    lens: np.ndarray,
    modes: np.ndarray,
    noise_stds: np.ndarray,
    action_scales: np.ndarray | None = None,
    hold_steps: np.ndarray | None = None,
) -> dict[str, Any]:
    statistics = {}

    for mode, mode_id in OFFLINE_MODE_IDS.items():
        indices = np.flatnonzero(modes == mode_id)
        if len(indices) == 0:
            continue

        returns = rewards[indices].sum(axis = 1)
        entry = dict(
            episodes = int(len(indices)),
            env_steps = int(lens[indices].sum()),
            return_mean = float(returns.mean()),
            return_std = float(returns.std()),
        )

        if mode == "expert_noise":
            values, counts = np.unique(noise_stds[indices], return_counts = True)
            entry["noise_std_episodes"] = {f"{value:g}": int(count) for value, count in zip(values, counts)}

        if mode == "expert_scaled":
            values, counts = np.unique(action_scales[indices], return_counts = True)
            if len(values) <= 16:
                entry["action_scale_episodes"] = {f"{value:g}": int(count) for value, count in zip(values, counts)}
            else:
                # Scales from a range: the first scale of each episode.
                entry["first_action_scale"] = dict(
                    min = float(values.min()), max = float(values.max()), mean = float(action_scales[indices].mean()),
                )

        if mode == "bang_bang":
            values, counts = np.unique(hold_steps[indices], return_counts = True)
            entry["hold_steps_episodes"] = {f"{value:g}": int(count) for value, count in zip(values, counts)}

        if env_name.startswith("HalfCheetah"):
            states = np.concatenate([obs[index, :lens[index] + 1] for index in indices])
            x_velocity = states[:, HALF_CHEETAH_X_VELOCITY_INDEX]
            entry["x_velocity_percentiles"] = {
                f"p{q}": float(value)
                for q, value in zip(X_VELOCITY_PERCENTILES, np.percentile(x_velocity, X_VELOCITY_PERCENTILES))
            }
            entry["flipped_frac"] = float((_wrap_angle(states[:, HALF_CHEETAH_ROOT_ANGLE_INDEX]) > np.pi / 2).mean())

        statistics[mode] = entry

    return statistics


def package_versions() -> dict[str, str | None]:
    try:
        import mujoco
        mujoco_version = mujoco.__version__
    except ImportError:
        mujoco_version = None

    return dict(
        gymnasium = gym.__version__,
        mujoco = mujoco_version,
        numpy = np.__version__,
        torch = torch.__version__,
    )


def check_return_buckets(buckets: Sequence[Sequence[float]]):
    """Each bucket is [low, high) with low < high. The buckets must not overlap: an episode must be in one bucket only."""

    for bucket in buckets:
        if len(bucket) != 2 or not float(bucket[0]) < float(bucket[1]):
            raise ValueError(f"each offline_past_run_return_buckets entry must be [low, high] with low < high, got {bucket!r}")

    ordered = sorted((float(low), float(high)) for low, high in buckets)
    for (_, high), (low, _) in zip(ordered, ordered[1:]):
        if low < high:
            raise ValueError(f"offline_past_run_return_buckets must not overlap, got {[list(bucket) for bucket in buckets]!r}")


def balanced_counts(total: int, capacity: np.ndarray) -> np.ndarray:
    """Split total into counts as equal as possible, each at most its capacity. The first entries get the remainder."""

    capacity = np.asarray(capacity, dtype = np.int64)
    if total > capacity.sum():
        raise ValueError(f"{total} episodes are necessary, but only {capacity.sum()} are available")

    counts = np.zeros_like(capacity)
    remaining = total

    while remaining > 0:
        # Each pass gives the open entries an equal share. An entry at its capacity gives its share to the others.
        open_entries = np.flatnonzero(counts < capacity)
        share = np.full(len(open_entries), remaining // len(open_entries))
        share[:remaining % len(open_entries)] += 1
        share = np.minimum(share, capacity[open_entries] - counts[open_entries])
        counts[open_entries] += share
        remaining -= int(share.sum())

    return counts


PAST_RUN_FIELDS = ("critic_state", "actions_continuous", "rewards")
PAST_RUN_META_FIELDS = ("episode_lens", "is_truncated", "terminals", "is_from_world_model")


def open_past_run_replay(folder: Path) -> SimpleNamespace:
    """Open the fields of a replay that past_run_episodes reads, read-only.

    ReplayBuffer.from_folder opens each field file with mode 'r+', and creates a field file that does not exist. The
    queues delete latents.data.npy of finished runs, so from_folder would write a zero latents file into the run folder.
    """

    missing = [
        name for name in (
            *(f"{field}.data.npy" for field in PAST_RUN_FIELDS),
            *(f"{field}.data.meta.npy" for field in PAST_RUN_META_FIELDS),
            "num_episodes.state.npy", "episode_index.state.npy",
        )
        if not (folder / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{folder} has no {missing}")

    return SimpleNamespace(
        data = {field: np.load(folder / f"{field}.data.npy", mmap_mode = "r") for field in PAST_RUN_FIELDS},
        meta_data = {field: np.load(folder / f"{field}.data.meta.npy", mmap_mode = "r") for field in PAST_RUN_META_FIELDS},
        num_episodes = int(np.load(folder / "num_episodes.state.npy")),
        episode_index = int(np.load(folder / "episode_index.state.npy")),
    )


def past_run_episodes(
    replay_folders: Sequence[str | Path],
    count: int,
    *,
    rng: np.random.Generator,
    max_episode_steps: int,
    obs_dim: int,
    action_dim: int,
    action_range: tuple[float, float],
    return_buckets: Sequence[Sequence[float]] = (),
    flip_episode_fraction: float = 0.,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, Any]]]:
    """Take count episodes from the replays of earlier online runs.

    Without return buckets, each folder gives count / len(folders) episodes, drawn uniformly from all loops of
    that run: the first random policy, the learned policies, and the collapsed gait of a run that collapsed.
    With return buckets [low, high), each bucket gets count / len(buckets) episodes with a return in it. The
    folders share the episodes of a bucket equally, and a folder with too few gives its share to the others.
    The first buckets and folders get the remainder.
    A replay row t holds obs_t, the native action a_t, and r_t. The last row of a truncated episode is the
    bootstrap row (final observation, zero action). Only full real episodes of max_episode_steps transitions
    are used. Returns the episodes in the dataset layout, the (folder index, replay episode index) of each
    episode, and one summary for each folder.
    With flip_episode_fraction > 0 (return buckets only), that share of each bucket comes from the episodes of the
    bucket in which the cheetah flips (an upright-to-flipped transition of the root angle), shared equally by the
    folders as above. If a bucket has too few of them, the other episodes of the bucket fill its count. The rest of
    the bucket comes from all its other episodes.
    """

    folders = [Path(folder) for folder in replay_folders]
    if len(folders) == 0:
        raise ValueError("past_runs episodes need at least one replay folder")

    check_return_buckets(return_buckets)
    buckets = [(float(low), float(high)) for low, high in return_buckets]
    if flip_episode_fraction > 0. and len(buckets) == 0:
        raise ValueError("flip_episode_fraction > 0 needs return buckets")

    replays, candidates, returns, has_flip = [], [], [], []

    for folder in folders:
        replay = open_past_run_replay(folder)
        meta = replay.meta_data

        if replay.data["critic_state"].shape[-1] != obs_dim or replay.data["actions_continuous"].shape[-1] != action_dim:
            raise ValueError(f"{folder} has observations or actions of another size than the env")

        # _initted is not a stored flag: the replay clears it on the next slot to write, which can hold a full
        # episode. A slot that was never written has episode length 0.
        usable = (
            (np.asarray(meta["episode_lens"]) == max_episode_steps + 1) &
            np.asarray(meta["is_truncated"], dtype = np.bool_) &
            ~np.asarray(meta["terminals"], dtype = np.bool_) &
            ~np.asarray(meta["is_from_world_model"], dtype = np.bool_)
        )
        episodes = np.flatnonzero(usable)
        replays.append(replay)
        candidates.append(episodes)
        returns.append(np.asarray(replay.data["rewards"][episodes, :max_episode_steps], dtype = np.float64).sum(-1) if len(buckets) else None)
        if flip_episode_fraction > 0.:
            angle = np.asarray(replay.data["critic_state"][episodes, :max_episode_steps + 1, HALF_CHEETAH_ROOT_ANGLE_INDEX])
            flipped = np.abs(_wrap_angle(angle)) > np.pi / 2
            has_flip.append((~flipped[:, :-1] & flipped[:, 1:]).any(axis = 1))

    # chosen[f]: the replay episodes of folder f. bucket_of[f]: the bucket index of each (-1 without buckets).
    chosen = [np.zeros(0, dtype = np.int64) for _ in folders]
    bucket_of = [np.zeros(0, dtype = np.int64) for _ in folders]

    if len(buckets) == 0:
        per_folder = [count // len(folders) + int(index < count % len(folders)) for index in range(len(folders))]
        for index, (folder, take) in enumerate(zip(folders, per_folder)):
            if len(candidates[index]) < take:
                raise ValueError(f"{folder} has {len(candidates[index])} full episodes; {take} are necessary")
            chosen[index] = rng.choice(candidates[index], size = take, replace = False)
            bucket_of[index] = np.full(take, -1)
    elif flip_episode_fraction <= 0.:
        per_bucket = [count // len(buckets) + int(index < count % len(buckets)) for index in range(len(buckets))]
        for bucket_index, ((low, high), total) in enumerate(zip(buckets, per_bucket)):
            in_bucket = [episodes[(folder_returns >= low) & (folder_returns < high)] for episodes, folder_returns in zip(candidates, returns)]
            try:
                takes = balanced_counts(total, np.array([len(episodes) for episodes in in_bucket]))
            except ValueError as error:
                raise ValueError(f"return bucket [{low:g}, {high:g}): {error}") from None
            for index, take in enumerate(takes):
                picked = rng.choice(in_bucket[index], size = int(take), replace = False)
                chosen[index] = np.concatenate([chosen[index], picked])
                bucket_of[index] = np.concatenate([bucket_of[index], np.full(int(take), bucket_index)])
    else:
        per_bucket = [count // len(buckets) + int(index < count % len(buckets)) for index in range(len(buckets))]
        flip_chosen = [np.zeros(0, dtype = np.int64) for _ in folders]
        for bucket_index, ((low, high), total) in enumerate(zip(buckets, per_bucket)):
            masks = [(folder_returns >= low) & (folder_returns < high) for folder_returns in returns]
            flip_pools = [episodes[mask & flips] for episodes, mask, flips in zip(candidates, masks, has_flip)]
            flip_total = min(int(round(flip_episode_fraction * total)), sum(len(pool) for pool in flip_pools))
            flip_takes = balanced_counts(flip_total, np.array([len(pool) for pool in flip_pools])) if flip_total > 0 else np.zeros(len(folders), dtype = np.int64)
            for index, take in enumerate(flip_takes):
                picked = rng.choice(flip_pools[index], size = int(take), replace = False)
                chosen[index] = np.concatenate([chosen[index], picked])
                flip_chosen[index] = np.concatenate([flip_chosen[index], picked])
                bucket_of[index] = np.concatenate([bucket_of[index], np.full(int(take), bucket_index)])
            # The rest of the bucket: all its episodes that are not chosen yet, with or without a flip.
            rest = [np.setdiff1d(episodes[mask], chosen[index]) for index, (episodes, mask) in enumerate(zip(candidates, masks))]
            try:
                takes = balanced_counts(total - flip_total, np.array([len(episodes) for episodes in rest]))
            except ValueError as error:
                raise ValueError(f"return bucket [{low:g}, {high:g}): {error}") from None
            for index, take in enumerate(takes):
                picked = rng.choice(rest[index], size = int(take), replace = False)
                chosen[index] = np.concatenate([chosen[index], picked])
                bucket_of[index] = np.concatenate([bucket_of[index], np.full(int(take), bucket_index)])

    arrays = dict(
        obs = np.zeros((count, max_episode_steps + 1, obs_dim), dtype = np.float32),
        actions = np.zeros((count, max_episode_steps, action_dim), dtype = np.float32),
        rewards = np.zeros((count, max_episode_steps), dtype = np.float32),
        terminated = np.zeros(count, dtype = np.bool_),
        truncated = np.ones(count, dtype = np.bool_),
        lens = np.full(count, max_episode_steps, dtype = np.int32),
    )
    source = np.zeros((count, 2), dtype = np.int64)
    summaries = []
    row = 0

    for folder_index, (folder, replay) in enumerate(zip(folders, replays)):
        # Sorted, so the reads go forward through the memmap.
        order = np.argsort(chosen[folder_index], kind = "stable")
        episodes, episode_buckets = chosen[folder_index][order], bucket_of[folder_index][order]
        take = len(episodes)

        for episode in episodes:
            arrays["obs"][row] = replay.data["critic_state"][episode, :max_episode_steps + 1]
            native = np.asarray(replay.data["actions_continuous"][episode, :max_episode_steps], dtype = np.float32)
            arrays["actions"][row] = np.clip(native_to_env_actions(native, action_range), *action_range)
            arrays["rewards"][row] = replay.data["rewards"][episode, :max_episode_steps]
            source[row] = (folder_index, episode)
            row += 1

        episode_returns = arrays["rewards"][row - take:row].sum(axis = 1)
        summary = dict(
            folder = str(folder),
            stored_episodes = int((np.asarray(replay.meta_data["episode_lens"]) > 0).sum()),
            # The replay state that the episode indices of past_run_source.npy refer to. A resumed run adds episodes.
            replay_num_episodes = int(replay.num_episodes),
            replay_episode_index = int(replay.episode_index),
            full_episodes = int(len(candidates[folder_index])),
            episodes = int(take),
            return_mean = float(episode_returns.mean()) if take else None,
            return_min = float(episode_returns.min()) if take else None,
            return_max = float(episode_returns.max()) if take else None,
        )
        if len(buckets):
            summary["episodes_per_bucket"] = {
                f"[{low:g}, {high:g})": int((episode_buckets == index).sum()) for index, (low, high) in enumerate(buckets)
            }
            summary["available_per_bucket"] = {
                f"[{low:g}, {high:g})": int(((returns[folder_index] >= low) & (returns[folder_index] < high)).sum())
                for low, high in buckets
            }
        if flip_episode_fraction > 0.:
            in_flip_share = np.isin(episodes, flip_chosen[folder_index])
            summary["flip_share_episodes_per_bucket"] = {
                f"[{low:g}, {high:g})": int((in_flip_share & (episode_buckets == index)).sum()) for index, (low, high) in enumerate(buckets)
            }
        summaries.append(summary)

    return arrays, source, summaries


def collect_offline_dataset(
    *,
    env_name: str,
    obs_dim: int,
    action_dim: int,
    action_range: tuple[float, float],
    data_dir: str | Path,
    dataset_name: str,
    data_steps: int,
    data_mix: Mapping[str, float],
    noise_stds: Sequence[float],
    data_seed: int,
    num_envs: int,
    expert_repo: str,
    expert_revision: str,
    expert_filename: str,
    expert_cache_dir: str | Path,
    expert_path: str | None,
    expert_min_return: float,
    expert_device: str | torch.device = "cpu",
    storage_dir: str | Path | None = None,
    past_run_replays: Sequence[str] = (),
    past_run_return_buckets: Sequence[Sequence[float]] = (),
    expert_action_scales: Sequence[float] = (),
    expert_scale_change_steps: int = 0,
    expert_scaled_noise_std: float = 0.,
    past_run_flip_episode_fraction: float = 0.,
    bang_bang_hold_steps: Sequence[float] = (),
) -> Path:
    """Collect the offline dataset. Skip the stage if an agreeing dataset exists.

    The expert actor runs on expert_device. The envs step on the CPU.
    With storage_dir, write the new dataset to storage_dir and symlink dataset_dir to it.
    """

    dataset_dir = offline_dataset_dir(data_dir, env_name, dataset_name)
    config = collection_config(
        env_name = env_name,
        dataset_name = dataset_name,
        data_steps = data_steps,
        data_mix = data_mix,
        noise_stds = noise_stds,
        data_seed = data_seed,
        num_envs = num_envs,
        expert_repo = expert_repo,
        expert_revision = expert_revision,
        expert_filename = expert_filename,
        expert_path = expert_path,
        past_run_replays = past_run_replays,
        past_run_return_buckets = past_run_return_buckets,
        expert_action_scales = expert_action_scales,
        expert_scale_change_steps = expert_scale_change_steps,
        expert_scaled_noise_std = expert_scaled_noise_std,
        past_run_flip_episode_fraction = past_run_flip_episode_fraction,
        bang_bang_hold_steps = bang_bang_hold_steps,
    )

    if dataset_dir.exists():
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"{dataset_dir} exists without manifest.json. Remove it or choose a new offline_dataset_name.")

        saved_config = json.loads(manifest_path.read_text())["config"]
        mismatched = sorted(key for key in config.keys() | saved_config.keys() if config.get(key) != saved_config.get(key))

        if mismatched:
            raise ValueError(
                f"offline dataset {dataset_dir} exists with a different config for {mismatched}. "
                "Choose a new offline_dataset_name; the collector does not overwrite data."
            )

        print(f"offline dataset exists and agrees with the config, skipping collection: {dataset_dir}")
        return dataset_dir

    mix = normalize_data_mix(data_mix)
    max_episode_steps = env_max_episode_steps(env_name)
    counts = {mode: round_half_up(data_steps * mix[mode] / max_episode_steps) for mode in OFFLINE_MODES}
    num_episodes = sum(counts.values())

    if num_episodes == 0:
        raise ValueError(f"offline_data_steps={data_steps} gives no episodes of {max_episode_steps} steps")

    rng = np.random.default_rng(data_seed)

    # Each expert_noise episode samples one noise std. The permutation mixes
    # the modes in each env batch.
    modes = np.concatenate([
        np.full(counts[mode], OFFLINE_MODE_IDS[mode], dtype = np.int8)
        for mode in OFFLINE_MODES
    ])
    episode_noise_stds = np.zeros(num_episodes, dtype = np.float32)
    is_noise = modes == OFFLINE_MODE_IDS["expert_noise"]
    if is_noise.any():
        episode_noise_stds[is_noise] = rng.choice(np.asarray(noise_stds, dtype = np.float32), size = int(is_noise.sum()))

    # Each expert_scaled episode samples one action scale. Without these episodes, the generator is not used here.
    episode_action_scales = np.ones(num_episodes, dtype = np.float32)
    is_scaled = modes == OFFLINE_MODE_IDS["expert_scaled"]
    scale_range = (float(min(expert_action_scales, default = 1.)), float(max(expert_action_scales, default = 1.)))
    if is_scaled.any() and expert_scale_change_steps > 0:
        # Scales from the range [min, max] of the list; the scale changes every expert_scale_change_steps steps.
        episode_action_scales[is_scaled] = rng.uniform(*scale_range, size = int(is_scaled.sum())).astype(np.float32)
    elif is_scaled.any():
        episode_action_scales[is_scaled] = rng.choice(np.asarray(expert_action_scales, dtype = np.float32), size = int(is_scaled.sum()))
    if is_scaled.any() and expert_scaled_noise_std > 0.:
        episode_noise_stds[is_scaled] = expert_scaled_noise_std

    # Each bang_bang episode samples one mean hold, with a separate generator: without these episodes, the main
    # generator gives the same values as before this mode existed.
    episode_hold_steps = np.ones(num_episodes, dtype = np.float32)
    is_bang = modes == OFFLINE_MODE_IDS["bang_bang"]
    if is_bang.any():
        hold_rng = np.random.default_rng([data_seed, OFFLINE_MODE_IDS["bang_bang"]])
        episode_hold_steps[is_bang] = hold_rng.choice(np.asarray(bang_bang_hold_steps, dtype = np.float32), size = int(is_bang.sum()))

    permutation = rng.permutation(num_episodes)
    modes, episode_noise_stds = modes[permutation], episode_noise_stds[permutation]
    episode_action_scales = episode_action_scales[permutation]
    episode_hold_steps = episode_hold_steps[permutation]

    actor = None
    expert_info = None

    if counts["expert"] + counts["expert_noise"] + counts["expert_scaled"] > 0:
        path, digest = download_expert(
            repo = expert_repo,
            revision = expert_revision,
            filename = expert_filename,
            cache_dir = expert_cache_dir,
            expert_path = expert_path,
        )
        actor = load_expert_actor(path, obs_dim = obs_dim, action_dim = action_dim).to(expert_device)
        validation_return = validate_expert(
            actor,
            env_name = env_name,
            action_range = action_range,
            action_dim = action_dim,
            min_return = expert_min_return,
            seed = data_seed + EXPERT_VALIDATION_SEED_OFFSET,
        )
        expert_info = dict(
            repo = expert_repo if not exists(expert_path) else None,
            revision = expert_revision if not exists(expert_path) else None,
            filename = expert_filename if not exists(expert_path) else None,
            path = str(path),
            sha256 = digest,
            validation_mean_return = validation_return,
            device = str(expert_device),
        )

    arrays = dict(
        obs = np.zeros((num_episodes, max_episode_steps + 1, obs_dim), dtype = np.float32),
        actions = np.zeros((num_episodes, max_episode_steps, action_dim), dtype = np.float32),
        rewards = np.zeros((num_episodes, max_episode_steps), dtype = np.float32),
        terminated = np.zeros(num_episodes, dtype = np.bool_),
        truncated = np.zeros(num_episodes, dtype = np.bool_),
        lens = np.zeros(num_episodes, dtype = np.int32),
    )

    env = _train_gym().make_env(
        env_name,
        data_seed,
        vectorized = True,
        num_envs = num_envs,
        env_vectorization_mode = "sync",
    )

    # The env steps only the expert, expert_noise, and random episodes. Without past_runs episodes, the env
    # episodes are all episodes, in the same order and with the same seeds as before that mode existed.
    env_episodes = np.flatnonzero(modes != OFFLINE_MODE_IDS["past_runs"])

    try:
        for start in tqdm(range(0, len(env_episodes), num_envs), desc = "offline collect"):
            stop = min(start + num_envs, len(env_episodes))
            count = stop - start
            batch_episodes = env_episodes[start:stop]

            # A partial last batch fills the unused envs with random episodes and drops them.
            batch_modes = np.full(num_envs, OFFLINE_MODE_IDS["random"], dtype = np.int8)
            batch_noise_stds = np.zeros(num_envs, dtype = np.float32)
            batch_action_scales = np.ones(num_envs, dtype = np.float32)
            batch_hold_steps = np.ones(num_envs, dtype = np.float32)
            batch_modes[:count] = modes[batch_episodes]
            batch_hold_steps[:count] = episode_hold_steps[batch_episodes]
            batch_noise_stds[:count] = episode_noise_stds[batch_episodes]
            batch_action_scales[:count] = episode_action_scales[batch_episodes]

            # The vector env seeds env i with seed + i, so each episode has its own seed.
            out = run_episode_batch(
                env,
                actor,
                modes = batch_modes,
                noise_stds = batch_noise_stds,
                rng = rng,
                seed = data_seed + start,
                action_range = action_range,
                max_episode_steps = max_episode_steps,
                action_dim = action_dim,
                action_scales = batch_action_scales,
                scale_change_steps = expert_scale_change_steps,
                scale_range = scale_range,
                scale_rng = np.random.default_rng([data_seed, OFFLINE_MODE_IDS["expert_scaled"], start]),
                hold_steps = batch_hold_steps,
                bang_rng = np.random.default_rng([data_seed, OFFLINE_MODE_IDS["bang_bang"], start]),
            )

            for key, value in out.items():
                arrays[key][batch_episodes] = value[:count]
    finally:
        env.close()

    past_run_source = None
    past_run_summaries = None
    if counts["past_runs"] > 0:
        # A separate generator, so the env episodes do not depend on the past_runs episodes.
        past_episodes = np.flatnonzero(modes == OFFLINE_MODE_IDS["past_runs"])
        past_arrays, past_source, past_run_summaries = past_run_episodes(
            past_run_replays,
            len(past_episodes),
            rng = np.random.default_rng([data_seed, OFFLINE_MODE_IDS["past_runs"]]),
            max_episode_steps = max_episode_steps,
            obs_dim = obs_dim,
            action_dim = action_dim,
            action_range = action_range,
            return_buckets = past_run_return_buckets,
            flip_episode_fraction = past_run_flip_episode_fraction,
        )
        for key, value in past_arrays.items():
            arrays[key][past_episodes] = value
        past_run_source = np.full((num_episodes, 2), -1, dtype = np.int64)
        past_run_source[past_episodes] = past_source

    statistics = mode_statistics(
        env_name,
        obs = arrays["obs"],
        rewards = arrays["rewards"],
        lens = arrays["lens"],
        modes = modes,
        noise_stds = episode_noise_stds,
        action_scales = episode_action_scales,
        hold_steps = episode_hold_steps,
    )
    manifest = dict(
        config = config,
        max_episode_steps = max_episode_steps,
        num_episodes = num_episodes,
        env_steps = int(arrays["lens"].sum()),
        episodes_per_mode = {mode: count for mode, count in counts.items() if mode in FIRST_OFFLINE_MODES or count > 0},
        obs_dim = obs_dim,
        action_dim = action_dim,
        action_range = list(action_range),
        versions = package_versions(),
        expert = expert_info,
        statistics = statistics,
        **(dict(past_runs = past_run_summaries) if exists(past_run_summaries) else {}),
    )

    final_dir = dataset_dir if storage_dir is None else Path(storage_dir)
    tmp_dir = final_dir.with_name(final_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors = True)
    tmp_dir.mkdir(parents = True)

    for key, value in arrays.items():
        np.save(tmp_dir / f"{key}.npy", value)

    np.save(tmp_dir / "mode.npy", modes)
    np.save(tmp_dir / "noise_std.npy", episode_noise_stds)
    if exists(past_run_source):
        # (replay folder index in manifest["past_runs"], replay episode index); -1 for the other modes.
        np.save(tmp_dir / PAST_RUN_SOURCE_FILE, past_run_source)
    if counts["expert_scaled"] > 0:
        # The action scale of each episode (the first scale if the scale changes); 1 for the other modes.
        np.save(tmp_dir / ACTION_SCALE_FILE, episode_action_scales)
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent = 2, sort_keys = True) + "\n")
    os.replace(tmp_dir, final_dir)

    if final_dir != dataset_dir:
        dataset_dir.parent.mkdir(parents = True, exist_ok = True)
        dataset_dir.symlink_to(final_dir, target_is_directory = True)

    print(f"offline dataset: {num_episodes} episodes, {manifest['env_steps']} env steps -> {dataset_dir}")
    for mode, entry in statistics.items():
        print(f"  {mode}: {entry['episodes']} episodes, return {entry['return_mean']:.1f} +- {entry['return_std']:.1f}")

    return dataset_dir


# load

@dataclass
class OfflineDataset:
    path: Path
    manifest: dict[str, Any]
    obs: np.ndarray          # (E, T + 1, obs_dim)
    actions: np.ndarray      # (E, T, action_dim), env range
    rewards: np.ndarray      # (E, T)
    terminated: np.ndarray   # (E,)
    truncated: np.ndarray    # (E,)
    lens: np.ndarray         # (E,) transitions per episode
    mode: np.ndarray         # (E,) OFFLINE_MODE_IDS values
    noise_std: np.ndarray    # (E,)

    @property
    def num_episodes(self) -> int:
        return int(self.lens.shape[0])

    @property
    def env_steps(self) -> int:
        return int(self.lens.sum())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.manifest, sort_keys = True).encode()).hexdigest()

    def observations(self, indices: np.ndarray) -> np.ndarray:
        """All valid observations of the episodes, including each final observation."""

        if len(indices) == 0:
            return np.zeros((0, self.obs.shape[-1]), dtype = np.float32)

        return np.concatenate([self.obs[index, :self.lens[index] + 1] for index in indices])


def load_offline_dataset(dataset_dir: str | Path) -> OfflineDataset:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"offline dataset not found: {manifest_path}. Run the collect offline stage first."
        )

    def load(name):
        return np.load(dataset_dir / f"{name}.npy", mmap_mode = "r")

    return OfflineDataset(
        path = dataset_dir,
        manifest = json.loads(manifest_path.read_text()),
        obs = load("obs"),
        actions = load("actions"),
        rewards = load("rewards"),
        terminated = np.asarray(load("terminated")),
        truncated = np.asarray(load("truncated")),
        lens = np.asarray(load("lens")),
        mode = np.asarray(load("mode")),
        noise_std = np.asarray(load("noise_std")),
    )


@dataclass
class OfflineSplit:
    train: np.ndarray
    validation: np.ndarray


def offline_split(dataset: OfflineDataset, validation_fraction: float, seed: int) -> OfflineSplit:
    """Hold out episodes per mode. Each mode with two or more episodes keeps at least one."""

    rng = np.random.default_rng(seed)
    validation = []

    for mode_id in OFFLINE_MODE_IDS.values():
        indices = np.flatnonzero(dataset.mode == mode_id)
        if validation_fraction <= 0. or len(indices) < 2:
            continue

        count = min(len(indices) - 1, max(1, round_half_up(len(indices) * validation_fraction)))
        validation.extend(rng.choice(indices, size = count, replace = False).tolist())

    validation = np.sort(np.asarray(validation, dtype = np.int64))
    train = np.setdiff1d(np.arange(dataset.num_episodes), validation)
    return OfflineSplit(train = train, validation = validation)


# latent replay

def interact_step_size(world_model) -> int:
    """The step_size that interact_with_env() stores with its default num_steps."""

    num_steps = inspect.signature(type(world_model).interact_with_env).parameters["num_steps"].default
    return world_model.max_steps // num_steps


def offline_episode_experience(
    *,
    latents: Tensor,
    observations: np.ndarray,
    env_actions: np.ndarray,
    rewards: np.ndarray,
    terminated: bool,
    truncated: bool,
    action_range: tuple[float, float],
    step_size: int,
) -> Experience:
    """One offline episode in the row layout of interact_with_env().

    latents and observations hold one row per observation (at least L + 1).
    env_actions and rewards hold the L transitions. Row t pairs obs_t with
    a_t and r_t. A truncated episode gets one bootstrap row with the final
    observation, a zero action, and a zero reward.
    """

    num_transitions = int(env_actions.shape[0])
    bootstrap = truncated and not terminated
    rows = num_transitions + int(bootstrap)

    native_actions = env_to_native_actions(np.asarray(env_actions, dtype = np.float32), action_range)
    native_actions = np.clip(native_actions, NATIVE_ACTION_EPS, 1. - NATIVE_ACTION_EPS)

    actions = torch.zeros(rows, env_actions.shape[-1])
    actions[:num_transitions] = torch.from_numpy(native_actions)
    episode_rewards = torch.zeros(rows)
    episode_rewards[:num_transitions] = torch.from_numpy(np.array(rewards, dtype = np.float32))

    return Experience(
        latents = rearrange(latents[:rows].float(), "t n d -> 1 t 1 n d"),
        critic_state = torch.from_numpy(np.array(observations[:rows], dtype = np.float32))[None],
        rewards = episode_rewards[None],
        actions = Actions(None, actions[None]),
        # The world-model loss does not read log probs or values. The zeros
        # keep the schema of the online replay.
        log_probs = Actions(None, torch.zeros(1, rows, env_actions.shape[-1])),
        values = torch.zeros(1, rows),
        step_size = step_size,
        lens = torch.tensor([rows]),
        is_truncated = torch.tensor([bool(truncated)]),
        terminals = torch.tensor([bool(terminated)]),
        is_from_world_model = False,
        episode_return = episode_rewards.sum()[None],
    )


def store_experience(replay: ReplayBuffer, experience: Experience):
    # The online loop stores each episode this way.
    data, meta = experience.to_buffer_dict()
    data, meta = tree_map_tensor(lambda t: rearrange(t, "1 ... -> ..."), (data, meta))
    replay.store_episode(**data, **meta)


@torch.no_grad()
def encode_observations(tokenizer, observations: np.ndarray, device: torch.device) -> Tensor:
    """Encode (..., obs_dim) observations to (..., n, d) latents on the CPU."""

    obs = torch.from_numpy(np.ascontiguousarray(observations, dtype = np.float32))
    flat = rearrange(obs, "... d -> (...) d").to(device)
    latents = tokenizer(flat, return_latents = True).cpu()
    return latents.reshape(*obs.shape[:-1], *latents.shape[1:])


def build_offline_replay(
    tokenizer,
    dataset: OfflineDataset,
    indices: np.ndarray,
    folder: Path,
    *,
    step_size: int,
    action_range: tuple[float, float],
    device: torch.device,
    marker: dict[str, Any],
) -> ReplayBuffer:
    """Encode the episodes into a memmap latent replay with the online replay layout."""

    tmp_folder = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp_folder, ignore_errors = True)
    replay = None

    with temp_eval(tokenizer):
        for start in tqdm(range(0, len(indices), ENCODE_CHUNK_EPISODES), desc = f"encode {folder.name}"):
            chunk = indices[start:start + ENCODE_CHUNK_EPISODES]
            latents = encode_observations(tokenizer, dataset.obs[chunk], device)

            for offset, index in enumerate(chunk):
                num_transitions = int(dataset.lens[index])
                experience = offline_episode_experience(
                    latents = latents[offset],
                    observations = dataset.obs[index],
                    env_actions = dataset.actions[index, :num_transitions],
                    rewards = dataset.rewards[index, :num_transitions],
                    terminated = bool(dataset.terminated[index]),
                    truncated = bool(dataset.truncated[index]),
                    action_range = action_range,
                    step_size = step_size,
                )

                if not exists(replay):
                    replay = Experience.create_memmap_replay_buffer(
                        experience,
                        tmp_folder,
                        max_episodes = len(indices),
                        max_timesteps = dataset.obs.shape[1],
                        circular = False,
                    )

                store_experience(replay, experience)

    replay.flush()
    del replay

    # The marker is written before the rename, so a folder without it is incomplete.
    (tmp_folder / REPLAY_MARKER_FILE).write_text(json.dumps(marker, indent = 2, sort_keys = True) + "\n")
    shutil.rmtree(folder, ignore_errors = True)
    os.replace(tmp_folder, folder)
    return ReplayBuffer.from_folder(folder, read_only = True)


def open_offline_replay(folder: Path, marker: dict[str, Any] | None = None) -> ReplayBuffer | None:
    """Open a complete replay folder. Return None if it is missing, incomplete, or stale."""

    marker_path = folder / REPLAY_MARKER_FILE
    if not marker_path.is_file():
        return None

    if exists(marker) and json.loads(marker_path.read_text()) != json_normalize(marker):
        return None

    return ReplayBuffer.from_folder(folder, read_only = True)


def open_rehearsal_replay(pretrained_checkpoint: str | Path) -> ReplayBuffer:
    folder = Path(pretrained_checkpoint).parent / REPLAY_DIR
    replay = open_offline_replay(folder)

    if not exists(replay):
        raise FileNotFoundError(f"offline rehearsal replay is missing or incomplete: {folder}")

    return replay


def rehearsal_batch_sizes(batch_size: int, rehearsal_fraction: float) -> tuple[int, int]:
    """Return (online, offline) window counts for one world-model batch."""

    offline = round_half_up(batch_size * rehearsal_fraction)
    online = batch_size - offline

    if offline < 0 or online <= 0:
        raise ValueError(
            f"offline_rehearsal_fraction={rehearsal_fraction} leaves no online windows in a batch of {batch_size}"
        )

    return online, offline


def concat_replay_batches(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Join two replay window batches key by key along the batch dimension."""

    if first.keys() != second.keys():
        raise ValueError(
            "replay batches have different keys: "
            f"only first {sorted(first.keys() - second.keys())}, only second {sorted(second.keys() - first.keys())}"
        )

    joined = {}

    for key, value in first.items():
        other = second[key]

        if not (torch.is_tensor(value) and torch.is_tensor(other)):
            raise ValueError(f"replay batch key {key!r} is not a tensor in both batches")

        if value.dtype != other.dtype or value.shape[1:] != other.shape[1:]:
            raise ValueError(
                f"replay batch key {key!r} differs: {value.dtype} {tuple(value.shape)} and {other.dtype} {tuple(other.shape)}"
            )

        joined[key] = torch.cat((value, other), dim = 0)

    return joined


# checkpoints

def pretrain_signature(values: Mapping[str, Any]) -> dict[str, Any]:
    signature = {key: json_normalize(values[key]) for key in PRETRAIN_SIGNATURE_KEYS}
    signature.update({
        key: json_normalize(values[key])
        for key in OPTIONAL_PRETRAIN_SIGNATURE_KEYS
        if values.get(key) is not None
    })
    return signature


def check_pretrain_signature(saved: Mapping[str, Any], current: Mapping[str, Any], *, source: str):
    missing = object()
    differences = [
        f"{key} (pretrained {saved.get(key, '<missing>')!r}, current {current.get(key, '<missing>')!r})"
        for key in sorted(saved.keys() | current.keys())
        if saved.get(key, missing) != current.get(key, missing)
    ]

    if differences:
        raise ValueError(f"{source} does not agree with the current config: {'; '.join(differences)}")


def agent_state_keys(world_model: nn.Module) -> set[str]:
    """State-dict keys of the policy head, the value heads, and their encoders."""

    agent_params = {id(param) for param in world_model.agent_parameters()}
    keys = {
        name
        for name, param in world_model.named_parameters(remove_duplicate = False)
        if id(param) in agent_params
    }

    # A module whose parameters are all agent parameters also owns its buffers.
    prefixes = list(AGENT_STATE_MODULES)
    for name, module in world_model.named_modules():
        params = list(module.parameters())
        if name and len(params) > 0 and all(id(param) in agent_params for param in params):
            prefixes.append(name)

    keys.update(
        key
        for key in world_model.state_dict().keys()
        if key in AGENT_STATE_BUFFERS or any(key.startswith(prefix + ".") for prefix in prefixes)
    )
    return keys


def world_model_state_without_agent_keys(world_model: nn.Module) -> dict[str, Tensor]:
    agent_keys = agent_state_keys(world_model)
    return {key: value for key, value in world_model.state_dict().items() if key not in agent_keys}


def load_world_model_without_agent_keys(world_model: nn.Module, state_dict: Mapping[str, Tensor]):
    """Load the world-model keys. Keep the agent heads of world_model unchanged."""

    target_keys = set(world_model.state_dict().keys())
    world_keys = target_keys - agent_state_keys(world_model)
    missing = sorted(world_keys - state_dict.keys())
    unexpected = sorted(state_dict.keys() - target_keys)

    if missing or unexpected:
        raise ValueError(
            "pretrained world model does not agree with the current world model: "
            f"missing non-agent keys {missing}, unexpected keys {unexpected}"
        )

    world_model.load_state_dict({key: state_dict[key] for key in world_keys}, strict = False)


def pretrained_package(
    *,
    tokenizer: nn.Module,
    world_model: nn.Module,
    world_optimizer,
    signature: Mapping[str, Any],
    dataset: OfflineDataset,
    world_model_steps: int,
) -> dict[str, Any]:
    return dict(
        tokenizer = tokenizer.state_dict(),
        world_model = world_model_state_without_agent_keys(world_model),
        world_optimizer = _train_gym().optimizer_state_dict_for_checkpoint(world_optimizer),
        signature = dict(signature),
        dataset_manifest = json.dumps(dataset.manifest, sort_keys = True),
        dataset_fingerprint = dataset.fingerprint,
        world_model_steps = int(world_model_steps),
        offline_env_steps = dataset.env_steps,
    )


def load_offline_pretrained(path: str | Path, *, signature: Mapping[str, Any]) -> dict[str, Any]:
    """Load a pretrained checkpoint to the CPU and check its signature."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"offline pretrained checkpoint not found: {path}")

    package = torch.load(str(path), map_location = "cpu", weights_only = True)
    missing = sorted({"tokenizer", "world_model", "world_optimizer", "signature", "offline_env_steps"} - package.keys())
    if missing:
        raise ValueError(f"{path} is not an offline pretrained checkpoint; it does not have {missing}")

    check_pretrain_signature(package["signature"], signature, source = str(path))
    return package


def load_optimizer_moments(optimizer, state_dict, name: str) -> bool:
    """Load the per-parameter optimizer state. Keep the configured group hyperparameters.

    Optimizer.load_state_dict() also replaces lr, weight decay, and betas with
    the saved values. The current config sets those values.
    """

    group_settings = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    loaded = _train_gym().load_optimizer_state_if_compatible(optimizer, state_dict, name)

    for group, settings in zip(optimizer.param_groups, group_settings):
        group.update(settings)

    return loaded


def apply_offline_pretrained(
    package: Mapping[str, Any],
    *,
    tokenizer: nn.Module,
    world_model: nn.Module,
    world_optimizer,
    load_world_optimizer: bool,
):
    tokenizer.load_state_dict(package["tokenizer"])
    load_world_model_without_agent_keys(world_model, package["world_model"])

    if load_world_optimizer:
        load_optimizer_moments(world_optimizer, package["world_optimizer"], "offline pretrained world")


def step_checkpoint_path(pretrain_dir: Path, step: int) -> Path:
    return pretrain_dir / f"{STEP_CHECKPOINT_PREFIX}{step}.pt"


def latest_step_checkpoint(pretrain_dir: Path) -> Path | None:
    steps = []

    for path in pretrain_dir.glob(f"{STEP_CHECKPOINT_PREFIX}*.pt"):
        suffix = path.stem[len(STEP_CHECKPOINT_PREFIX):]
        if suffix.isdigit():
            steps.append((int(suffix), path))

    return max(steps)[1] if steps else None


def pretrain_can_resume(pretrain_dir: Path) -> bool:
    # split.json is the first file of a pretrain stage. A stage that stops
    # before tokenizer.pt exists trains the tokenizer again.
    return (pretrain_dir / SPLIT_FILE).is_file() and not (pretrain_dir / PRETRAINED_FILE).exists()


# pretrain

def write_split(pretrain_dir: Path, split: OfflineSplit, dataset: OfflineDataset):
    pretrain_dir.mkdir(parents = True, exist_ok = True)
    content = dict(
        dataset = str(dataset.path),
        dataset_fingerprint = dataset.fingerprint,
        train = split.train.tolist(),
        validation = split.validation.tolist(),
    )
    (pretrain_dir / SPLIT_FILE).write_text(json.dumps(content) + "\n")


@torch.no_grad()
def tokenizer_recon_by_mode(
    tokenizer,
    dataset: OfflineDataset,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Mean tokenizer reconstruction loss per behavior mode."""

    metrics = {}

    with temp_eval(tokenizer):
        for mode, mode_id in OFFLINE_MODE_IDS.items():
            mode_indices = indices[dataset.mode[indices] == mode_id]
            observations = dataset.observations(mode_indices)
            if len(observations) == 0:
                continue

            loss_sum = 0.
            for start in range(0, len(observations), TOKENIZER_EVAL_CHUNK_ROWS):
                chunk = torch.from_numpy(observations[start:start + TOKENIZER_EVAL_CHUNK_ROWS]).to(device)
                loss_sum += tokenizer(chunk).item() * len(chunk)

            metrics[f"offline_pretrain/val/tokenizer_recon_loss/{mode}"] = loss_sum / len(observations)

    return metrics


def prepare_offline_tokenizer(
    tokenizer,
    *,
    dataset: OfflineDataset,
    split: OfflineSplit,
    pretrain_dir: Path,
    signature: Mapping[str, Any],
    train_tokenizer_fn: Callable[[nn.Module, Tensor], float | None],
    writer,
    device: torch.device,
) -> float | None:
    """Train the tokenizer on the offline training episodes, or load it on a pretrain restart."""

    path = pretrain_dir / TOKENIZER_FILE

    if path.is_file():
        package = torch.load(str(path), map_location = "cpu", weights_only = True)
        check_pretrain_signature(package["signature"], signature, source = str(path))

        if package["dataset_fingerprint"] != dataset.fingerprint:
            raise ValueError(f"{path} was trained on a different offline dataset than {dataset.path}")

        tokenizer.load_state_dict(package["tokenizer"])
        print(f"offline pretrain restart: loaded the tokenizer from {path}")
        return None

    write_split(pretrain_dir, split, dataset)
    observations = torch.from_numpy(dataset.observations(split.train))
    print(f"offline tokenizer data: {len(observations)} observations from {len(split.train)} episodes")

    loss = train_tokenizer_fn(tokenizer, observations)
    del observations

    save_torch_atomic(
        dict(
            tokenizer = tokenizer.state_dict(),
            signature = dict(signature),
            dataset_fingerprint = dataset.fingerprint,
        ),
        path,
    )

    metrics = tokenizer_recon_by_mode(tokenizer, dataset, split.validation, device)
    _train_gym().log_scalars(writer, metrics, 0)
    for key, value in metrics.items():
        print(f"{key}: {value:.5f}")

    return loss


def ensure_offline_replays(
    tokenizer,
    *,
    dataset: OfflineDataset,
    split: OfflineSplit,
    pretrain_dir: Path,
    step_size: int,
    action_range: tuple[float, float],
    device: torch.device,
) -> tuple[ReplayBuffer, ReplayBuffer | None]:
    """Open the offline latent replays. Build a replay that is missing or stale."""

    tokenizer_sha256 = sha256_file(pretrain_dir / TOKENIZER_FILE)
    replays = []

    for name, indices in ((REPLAY_DIR, split.train), (VALIDATION_REPLAY_DIR, split.validation)):
        if len(indices) == 0:
            replays.append(None)
            continue

        folder = pretrain_dir / name
        marker = json_normalize(dict(
            dataset_fingerprint = dataset.fingerprint,
            tokenizer_sha256 = tokenizer_sha256,
            step_size = step_size,
            episodes = indices.tolist(),
        ))

        replay = open_offline_replay(folder, marker)
        if exists(replay):
            print(f"offline pretrain restart: reusing {folder}")
        else:
            replay = build_offline_replay(
                tokenizer,
                dataset,
                indices,
                folder,
                step_size = step_size,
                action_range = action_range,
                device = device,
                marker = marker,
            )

        replays.append(replay)

    return replays[0], replays[1]


def pretrain_offline(
    *,
    tokenizer,
    world_model,
    world_optimizer,
    dataset: OfflineDataset,
    split: OfflineSplit,
    pretrain_dir: Path,
    signature: Mapping[str, Any],
    action_range: tuple[float, float],
    device: torch.device,
    train_steps: Callable,
    validate: Callable[[ReplayBuffer], dict[str, float]],
    world_model_steps: int,
    finetune_steps: int,
    checkpoint_every: int,
    writer,
) -> Path:
    """Train the world model on the offline replay. Return the pretrained checkpoint path.

    train_steps(replay, steps, global_step, shortcut_train_mode, desc) runs
    train_world_model() with the batch shape and compiled loss of main().
    """

    final_path = pretrain_dir / PRETRAINED_FILE
    if final_path.exists():
        raise FileExistsError(
            f"{final_path} exists. Use offline_stages=[rl] with offline_pretrained_checkpoint, "
            "or choose a new offline_pretrain_name."
        )

    replay, validation_replay = ensure_offline_replays(
        tokenizer,
        dataset = dataset,
        split = split,
        pretrain_dir = pretrain_dir,
        step_size = interact_step_size(world_model),
        action_range = action_range,
        device = device,
    )

    step = 0
    latest = latest_step_checkpoint(pretrain_dir)

    if exists(latest):
        package = load_offline_pretrained(latest, signature = signature)
        if package["dataset_fingerprint"] != dataset.fingerprint:
            raise ValueError(f"{latest} was trained on a different offline dataset than {dataset.path}")

        # The world model of a step checkpoint trained on the latents of its own tokenizer.
        current_tokenizer = tokenizer.state_dict()
        if any(not torch.equal(value.cpu(), current_tokenizer[key].cpu()) for key, value in package["tokenizer"].items()):
            raise ValueError(f"{latest} belongs to a different tokenizer than {pretrain_dir / TOKENIZER_FILE}")

        load_world_model_without_agent_keys(world_model, package["world_model"])
        load_optimizer_moments(world_optimizer, package["world_optimizer"], "offline pretrain world")
        step = int(package["world_model_steps"])
        print(f"offline pretrain restart: continuing from {latest} at world-model step {step}")

    total_steps = finetune_steps + world_model_steps
    log = _train_gym().log_scalars

    def log_validation(at_step: int):
        if not exists(validation_replay):
            return

        metrics = {f"offline_pretrain/val/{key}": value for key, value in validate(validation_replay).items()}
        log(writer, metrics, at_step)
        print(f"offline pretrain step {at_step}: validation world-model loss {metrics['offline_pretrain/val/world_model/loss']:.4f}")

    def package_at(at_step: int):
        return pretrained_package(
            tokenizer = tokenizer,
            world_model = world_model,
            world_optimizer = world_optimizer,
            signature = signature,
            dataset = dataset,
            world_model_steps = at_step,
        )

    while step < total_steps:
        in_finetune = step < finetune_steps
        phase_end = finetune_steps if in_finetune else total_steps
        chunk_end = phase_end

        if checkpoint_every > 0:
            chunk_end = min(phase_end, (step // checkpoint_every + 1) * checkpoint_every)

        step, _ = train_steps(
            replay,
            steps = chunk_end - step,
            global_step = step,
            shortcut_train_mode = "finetune" if in_finetune else "combined",
            desc = "offline world model finetune" if in_finetune else "offline world model",
        )

        if checkpoint_every > 0 and step % checkpoint_every == 0 and step < total_steps:
            # Save first: a failed validation (for example out of GPU memory) does not lose the steps.
            save_torch_atomic(package_at(step), step_checkpoint_path(pretrain_dir, step))
            log_validation(step)

    save_torch_atomic(package_at(step), final_path)
    log_validation(step)
    print(f"offline pretrained checkpoint: {final_path}")
    return final_path
