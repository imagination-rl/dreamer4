"""Ablation flags and diagnostics from runs/audit_9-24-26/REPORT.md.

Every flag in this module is false by default. When all flags are false, the
trainer does not call this module, and the default path does not change.

Ablations:
- audit_deeper_dynamics (A1): add one temporal and one spatial layer to the
  dynamics transformer (depth 3 S-T-S becomes depth 5 S-T-S-T-S).
- audit_arrival_reward_target (A2): train the reward head on the arrival
  reward r(s_t, a_t) and pair each generated action with that reward.
- audit_quarter_envs_per_loop (A3): collect 4x fewer environment steps per
  loop with the same updates per loop, replay capacity, and env-step budget.

Diagnostics (audit_diagnostics):
- Real-rollout behavior metrics after each rollout.
- World-model quality metrics on a fixed env-step cadence. These metrics
  compare world-model predictions with MuJoCo resimulations from exact states.
  The diagnostics restore the global RNG state, so they do not change the
  training random stream.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any

import numpy as np
import gymnasium as gym

import torch
from torch import Tensor
from torch_einops_utils import temp_eval

AUDIT_ABLATION_FLAGS = (
    "audit_deeper_dynamics",
    "audit_arrival_reward_target",
    "audit_quarter_envs_per_loop",
)

QUARTER_ENVS_FACTOR = 4

# Diagnostic sample sizes. They match the offline audit scripts.
ONE_STEP_WINDOWS = 128
RANKING_START_STATES = 64
RANKING_SEQUENCES = 8
RANKING_ACTION_NOISE = 0.3          # std in units of half the action range
CLOSED_LOOP_DREAMS = 64
STATE_ERROR_HORIZONS = (1, 8, 32)

# HalfCheetah observation layout: obs[0:8] = qpos[1:], obs[8:17] = qvel.
HALF_CHEETAH_ROOT_ANGLE_INDEX = 1
HALF_CHEETAH_X_VELOCITY_INDEX = 8
HALF_CHEETAH_CTRL_COST_WEIGHT = 0.1


def exists(v):
    return v is not None


def resolve_audit_ablations(
    *,
    audit_deeper_dynamics: bool,
    audit_arrival_reward_target: bool,
    audit_quarter_envs_per_loop: bool,
    depth: int,
    time_block_every: int,
    old_reward_shifting: bool,
    align_generated_dream_transitions: bool,
    num_envs: int,
    replay_size: int,
    num_loops: int,
    checkpoint_every: int,
) -> dict[str, Any]:
    """Return only the configuration values that the enabled ablations replace."""

    overrides: dict[str, Any] = {}

    if audit_deeper_dynamics:
        # With time_block_every = 2, layer i (1-based) is temporal when i is even.
        # An odd depth ends with a spatial layer, so agent tokens read the
        # final temporal features. Two extra layers keep that pattern.
        if time_block_every != 2 or depth % 2 == 0:
            raise ValueError(
                "audit_deeper_dynamics requires time_block_every=2 and an odd depth; "
                f"got depth={depth}, time_block_every={time_block_every}"
            )

        overrides["depth"] = depth + 2

    if audit_arrival_reward_target:
        if not old_reward_shifting:
            overrides["old_reward_shifting"] = True

        if not align_generated_dream_transitions:
            overrides["align_generated_dream_transitions"] = True

    if audit_quarter_envs_per_loop:
        if num_envs % QUARTER_ENVS_FACTOR != 0:
            raise ValueError(
                f"audit_quarter_envs_per_loop requires num_envs divisible by {QUARTER_ENVS_FACTOR}; "
                f"got num_envs={num_envs}"
            )

        overrides.update(
            num_envs = num_envs // QUARTER_ENVS_FACTOR,
            replay_size = replay_size * QUARTER_ENVS_FACTOR,
            num_loops = num_loops * QUARTER_ENVS_FACTOR,
            checkpoint_every = checkpoint_every * QUARTER_ENVS_FACTOR,
        )

    return overrides


@contextmanager
def isolated_rng(device: torch.device):
    """Restore the Python, NumPy, and torch RNG states on exit."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [device] if device.type == "cuda" else []

    try:
        with torch.random.fork_rng(devices = devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _mean(x) -> float | None:
    x = np.asarray(x, dtype = np.float64)
    return float(x.mean()) if x.size > 0 else None


def _corr(a, b) -> float | None:
    a = np.asarray(a, dtype = np.float64).ravel()
    b = np.asarray(b, dtype = np.float64).ravel()

    if a.size < 3 or a.std() == 0. or b.std() == 0.:
        return None

    return float(np.corrcoef(a, b)[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x))


def _wrap_angle(x: np.ndarray) -> np.ndarray:
    return np.abs(np.angle(np.exp(1j * x)))


def _numpy(t: Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _frame_latents(latents: Tensor) -> Tensor:
    # generate() and replay can carry a single view axis: (b t 1 n d) -> (b t n d).
    return latents[:, :, 0] if latents.ndim == 5 else latents


# Beta policy actions are stored in [0, 1] and rescaled for the env.

def native_to_env_actions(native_actions: np.ndarray, action_range: tuple[float, float]) -> np.ndarray:
    action_low, action_high = action_range
    env_actions = action_low + native_actions * (action_high - action_low)
    return np.clip(env_actions, action_low, action_high)


def env_to_native_actions(env_actions: np.ndarray, action_range: tuple[float, float]) -> np.ndarray:
    action_low, action_high = action_range
    return (env_actions - action_low) / (action_high - action_low)


class AuditDiagnostics:
    def __init__(
        self,
        *,
        env_name: str,
        obs_dim: int,
        action_range: tuple[float, float],
        prompt_length: int,
        horizon: int,
        generate_steps: int,
        use_time_cache: bool,
        old_reward_shifting: bool,
        every_env_steps: int,
        seed: int,
        device: torch.device,
    ):
        if every_env_steps <= 0:
            raise ValueError("audit_diagnostics_every_env_steps must be positive")

        if prompt_length <= 0:
            raise ValueError("audit_diagnostics requires imagination_prompt_length > 0")

        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_low, self.action_high = action_range
        self.prompt_length = prompt_length
        self.horizon = horizon
        self.generate_steps = generate_steps
        self.use_time_cache = use_time_cache
        self.old_reward_shifting = old_reward_shifting
        self.every_env_steps = every_env_steps
        self.next_env_step = every_env_steps
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.is_half_cheetah = env_name.startswith("HalfCheetah")

        self.sim_env = None
        self.sim_ok = None
        self.current_x_velocity_p10 = None

    def close(self):
        if exists(self.sim_env):
            self.sim_env.close()
            self.sim_env = None

    # action helpers

    def to_env_actions(self, native_actions: np.ndarray) -> np.ndarray:
        return native_to_env_actions(native_actions, (self.action_low, self.action_high))

    def to_native_actions(self, env_actions: np.ndarray) -> np.ndarray:
        return env_to_native_actions(env_actions, (self.action_low, self.action_high))

    # reward alignment

    def aligned_reward_slices(self, horizon: int):
        """Return (dream slice, true slice) for rewards that target the same transition.

        True reward t is r(s_{P-1+t}, a_{P-1+t}) for t in [0, horizon).
        With old_reward_shifting, generated frame P+t reports that reward.
        Without it, generated frame P+t reports r(s_{P+t+1}, a_{P+t+1}).
        """

        P = self.prompt_length

        if self.old_reward_shifting:
            return slice(P, P + horizon), slice(0, horizon)

        return slice(P, P + horizon - 2), slice(2, horizon)

    # schedule

    def is_due(self, env_step: int) -> bool:
        if env_step < self.next_env_step:
            return False

        self.next_env_step = (env_step // self.every_env_steps + 1) * self.every_env_steps
        return True

    # MuJoCo state restore

    def sim(self):
        """Return an unwrapped MuJoCo env whose state can be set from one observation."""

        if self.sim_ok is False:
            return None

        if exists(self.sim_env):
            return self.sim_env

        try:
            env = gym.make(self.env_name).unwrapped
        except Exception:
            self.sim_ok = False
            return None

        model = getattr(env, "model", None)
        can_restore = (
            hasattr(env, "set_state") and
            hasattr(env, "_get_obs") and
            exists(model) and
            self.obs_dim == model.nq - 1 + model.nv
        )

        if not can_restore:
            env.close()
            self.sim_ok = False
            return None

        self.sim_env = env
        return env

    def restore(self, env, obs: np.ndarray) -> bool:
        nq = env.model.nq
        env.reset(seed = 0)
        qpos = np.concatenate([[0.], obs[:nq - 1]])
        qvel = obs[nq - 1:]
        env.set_state(qpos, qvel)

        if self.sim_ok is None:
            # Check the observation layout once. HalfCheetah, Hopper, and
            # Walker2d exclude only the root x position from the observation.
            self.sim_ok = bool(np.allclose(env._get_obs(), obs, atol = 1e-4))

        return self.sim_ok

    def resimulate(self, start_obs: np.ndarray, env_actions: np.ndarray):
        """Return true rewards (b h) and the final observations (b obs_dim), or None."""

        env = self.sim()
        if not exists(env):
            return None

        batch, horizon = env_actions.shape[:2]
        rewards = np.zeros((batch, horizon))
        observations = np.zeros((batch, horizon, self.obs_dim))

        for b in range(batch):
            if not self.restore(env, start_obs[b]):
                return None

            for t in range(horizon):
                obs, reward, *_ = env.step(env_actions[b, t])
                rewards[b, t] = reward
                observations[b, t] = obs

        return rewards, observations

    # real rollout behavior

    def rollout_metrics(self, exp, lengths: list[int]) -> dict[str, float | None]:
        """Behavior statistics of the real rollout. The cost is negligible."""

        if not exists(exp.rewards) or not exists(exp.actions) or not exists(exp.actions.continuous):
            return {}

        rewards = _numpy(exp.rewards)
        native_actions = _numpy(exp.actions.continuous)
        batch, time = rewards.shape
        lengths = np.clip(np.asarray(lengths, dtype = np.int64), 0, time)
        mask = np.arange(time)[None, :] < lengths[:, None]

        env_actions = self.to_env_actions(native_actions[:, :time])
        half_range = (self.action_high - self.action_low) / 2.
        center = (self.action_high + self.action_low) / 2.
        action_sq_sum = (env_actions ** 2).sum(-1)
        saturated = (np.abs(env_actions - center) > 0.9 * half_range).mean(-1)

        half = time // 2
        first_half = mask & (np.arange(time)[None, :] < half)
        second_half = mask & (np.arange(time)[None, :] >= half)

        metrics = {
            "audit/rollout/average_return": _mean(exp.episode_return.float().cpu().numpy()) if exists(exp.episode_return) else None,
            "audit/rollout/reward_per_step": _mean(rewards[mask]),
            "audit/rollout/reward_per_step_first_half": _mean(rewards[first_half]),
            "audit/rollout/reward_per_step_second_half": _mean(rewards[second_half]),
            "audit/rollout/action_sq_sum_per_step": _mean(action_sq_sum[mask]),
            "audit/rollout/action_saturation_frac": _mean(saturated[mask]),
        }

        if self.is_half_cheetah and exists(exp.critic_state):
            obs = _numpy(exp.critic_state)[:, :time]
            x_velocity = obs[..., HALF_CHEETAH_X_VELOCITY_INDEX][mask]
            flipped = _wrap_angle(obs[..., HALF_CHEETAH_ROOT_ANGLE_INDEX]) > np.pi / 2
            control_cost = HALF_CHEETAH_CTRL_COST_WEIGHT * action_sq_sum

            metrics.update({
                "audit/rollout/flipped_frac": _mean(flipped[mask]),
                "audit/rollout/x_velocity": _mean(x_velocity),
                "audit/rollout/control_cost_per_step": _mean(control_cost[mask]),
            })

            if x_velocity.size > 0:
                self.current_x_velocity_p10 = float(np.percentile(x_velocity, 10))

        return metrics

    # world-model quality

    @torch.no_grad()
    def model_metrics(self, world_model, tokenizer, replay, fresh_exp, lengths: list[int]) -> dict[str, float | None]:
        """World-model quality on the fresh rollout and on replay prompts.

        The trainer calls this before the world model trains on fresh_exp, so
        the one-step and ranking metrics use held-out current-policy data.
        """

        metrics: dict[str, float | None] = {}

        with isolated_rng(self.device), temp_eval(world_model), temp_eval(tokenizer):
            metrics.update(self.one_step_metrics(world_model, tokenizer, fresh_exp, lengths))
            metrics.update(self.ranking_metrics(world_model, fresh_exp, lengths))
            metrics.update(self.closed_loop_metrics(world_model, tokenizer, replay))

        metrics["audit/sim/restore_ok"] = None if self.sim_ok is None else float(self.sim_ok)
        return metrics

    def generate(self, world_model, time_steps: int, prompt_latents: Tensor, prompt_actions: Tensor, *, return_agent_actions: bool):
        return world_model.generate(
            time_steps,
            num_steps = self.generate_steps,
            batch_size = prompt_latents.shape[0],
            return_decoded_video = False,
            return_agent_actions = return_agent_actions,
            return_rewards_per_frame = True,
            return_terminals = False,
            use_time_cache = self.use_time_cache,
            prompt_latents = prompt_latents.to(self.device),
            prompt_continuous_actions = prompt_actions.to(self.device),
        )

    def sample_windows(self, lengths: np.ndarray, count: int, window: int):
        valid = np.flatnonzero(lengths >= window)
        if len(valid) == 0:
            return None

        episodes = self.rng.choice(valid, size = count)
        starts = np.array([self.rng.integers(0, lengths[e] - window + 1) for e in episodes])
        return episodes, starts

    def one_step_metrics(self, world_model, tokenizer, exp, lengths) -> dict[str, float | None]:
        P = self.prompt_length

        if not exists(exp.critic_state) or not exists(exp.actions) or not exists(exp.actions.continuous):
            return {}

        # The window needs P prompt frames, one predicted frame, and r_{P+1}.
        windows = self.sample_windows(np.asarray(lengths), ONE_STEP_WINDOWS, P + 2)
        if not exists(windows):
            return {}

        episodes, starts = windows
        index = torch.from_numpy(episodes)
        time_index = torch.from_numpy(starts)[:, None] + torch.arange(P + 2)[None, :]

        latents = _frame_latents(exp.latents)[index[:, None], time_index]
        actions = exp.actions.continuous[index[:, None], time_index]
        obs = _numpy(exp.critic_state[index[:, None], time_index])
        rewards = _numpy(exp.rewards[index[:, None], time_index])

        generated = self.generate(
            world_model,
            P + 1,
            latents[:, :P].float(),
            actions[:, :P].float(),
            return_agent_actions = False,
        )

        predicted = _numpy(tokenizer.decode(_frame_latents(generated.latents)[:, P]))
        target = obs[:, P]
        scale = target.std(0) + 1e-6

        error = np.abs(predicted - target) / scale
        copy_last = np.abs(obs[:, P - 1] - target) / scale

        predicted_reward = _numpy(generated.rewards[:, P])
        target_reward = rewards[:, P - 1] if self.old_reward_shifting else rewards[:, P + 1]

        metrics = {
            "audit/world_model/one_step_norm_mae": _mean(error),
            "audit/world_model/one_step_copy_last_norm_mae": _mean(copy_last),
            "audit/world_model/one_step_error_to_change_ratio": _mean(error) / max(_mean(copy_last), 1e-6),
            "audit/world_model/reward_head_mae": _mean(np.abs(predicted_reward - target_reward)),
            "audit/world_model/reward_head_corr": _corr(predicted_reward, target_reward),
            "audit/world_model/reward_target_std": float(target_reward.std()),
        }

        if self.is_half_cheetah:
            index_x = HALF_CHEETAH_X_VELOCITY_INDEX
            metrics["audit/world_model/one_step_x_velocity_mae"] = _mean(np.abs(predicted[:, index_x] - target[:, index_x]))

        return metrics

    def ranking_metrics(self, world_model, exp, lengths) -> dict[str, float | None]:
        """Open-loop ranking of perturbed action sequences from current-policy states."""

        P, H, M = self.prompt_length, self.horizon, RANKING_SEQUENCES

        if not exists(exp.critic_state) or not exists(self.sim()):
            return {}

        windows = self.sample_windows(np.asarray(lengths), RANKING_START_STATES, P + H)
        if not exists(windows):
            return {}

        episodes, starts = windows
        B = len(episodes)
        index = torch.from_numpy(episodes)
        time_index = torch.from_numpy(starts)[:, None] + torch.arange(P + H)[None, :]

        latents = _frame_latents(exp.latents)[index[:, None], time_index][:, :P].float()
        native_actions = _numpy(exp.actions.continuous[index[:, None], time_index])
        start_obs = _numpy(exp.critic_state[index, torch.from_numpy(starts) + P - 1])

        # Sequence 0 replays the real actions. Sequences 1..M-1 add noise to
        # the H actions that start at the last prompt frame.
        env_actions = np.repeat(self.to_env_actions(native_actions)[:, None], M, axis = 1)
        half_range = (self.action_high - self.action_low) / 2.
        noise = self.rng.normal(0., RANKING_ACTION_NOISE * half_range, size = (B, M - 1, H, env_actions.shape[-1]))
        margin = 1e-3 * half_range
        env_actions[:, 1:, P - 1:P - 1 + H] = np.clip(
            env_actions[:, 1:, P - 1:P - 1 + H] + noise,
            self.action_low + margin,
            self.action_high - margin,
        )

        simulated = self.resimulate(
            np.repeat(start_obs, M, axis = 0),
            env_actions[:, :, P - 1:P - 1 + H].reshape(B * M, H, -1),
        )
        if not exists(simulated):
            return {}

        true_rewards = simulated[0].reshape(B, M, H)

        prompt_latents = latents[:, None].expand(B, M, *latents.shape[1:]).flatten(0, 1)
        prompt_actions = torch.from_numpy(self.to_native_actions(env_actions)).float().flatten(0, 1)

        generated = self.generate(world_model, P + H, prompt_latents, prompt_actions, return_agent_actions = False)
        dream_rewards = _numpy(generated.rewards).reshape(B, M, -1)

        dream_slice, true_slice = self.aligned_reward_slices(H)
        dream_returns = dream_rewards[..., dream_slice].sum(-1)
        true_returns = true_rewards[..., true_slice].sum(-1)

        spearman = [
            _corr(_rank(dream_returns[b]), _rank(true_returns[b]))
            for b in range(B)
        ]
        spearman = [value for value in spearman if exists(value)]

        upper = np.triu_indices(M, k = 1)
        dream_diff = (dream_returns[:, :, None] - dream_returns[:, None, :])[:, upper[0], upper[1]]
        true_diff = (true_returns[:, :, None] - true_returns[:, None, :])[:, upper[0], upper[1]]

        true_spread = true_returns.std(1).mean()

        return {
            "audit/world_model/ranking_pairwise_sign_agreement": _mean(np.sign(dream_diff) == np.sign(true_diff)),
            "audit/world_model/ranking_spearman": _mean(spearman),
            "audit/world_model/ranking_spread_ratio": float(dream_returns.std(1).mean() / max(true_spread, 1e-6)),
            "audit/world_model/ranking_true_spread": float(true_spread),
        }

    def closed_loop_metrics(self, world_model, tokenizer, replay) -> dict[str, float | None]:
        """Dream with the current policy from replay prompts, then replay the actions in MuJoCo.

        Prompts use the training distribution: uniform over stored episodes,
        then a uniform window start.
        """

        P, H = self.prompt_length, self.horizon

        if not exists(self.sim()):
            return {}

        lengths = np.asarray(replay.episode_lens)
        windows = self.sample_windows(lengths, CLOSED_LOOP_DREAMS, P)
        if not exists(windows):
            return {}

        episodes, starts = windows
        data = replay.data

        latents = np.stack([data["latents"][e, s:s + P] for e, s in zip(episodes, starts)])
        native_prompt_actions = np.stack([data["actions_continuous"][e, s:s + P] for e, s in zip(episodes, starts)])
        start_obs = np.stack([data["critic_state"][e, s + P - 1] for e, s in zip(episodes, starts)])

        generated = self.generate(
            world_model,
            P + H,
            _frame_latents(torch.from_numpy(latents)).float(),
            torch.from_numpy(native_prompt_actions).float(),
            return_agent_actions = True,
        )

        # Action P-1 is the last prompt (replay) action. Actions P.. come from the policy.
        dream_native_actions = _numpy(generated.actions.continuous)
        env_actions = self.to_env_actions(dream_native_actions[:, P - 1:P - 1 + H])

        simulated = self.resimulate(start_obs, env_actions)
        if not exists(simulated):
            return {}

        true_rewards, true_obs = simulated
        dream_rewards = _numpy(generated.rewards)

        dream_slice, true_slice = self.aligned_reward_slices(H)
        dream_step = dream_rewards[:, dream_slice]
        true_step = true_rewards[:, true_slice]

        decoded = _numpy(tokenizer.decode(_frame_latents(generated.latents)[:, P:P + H].flatten(0, 1))).reshape(len(episodes), H, -1)
        scale = true_obs.reshape(-1, self.obs_dim).std(0) + 1e-6
        state_error = (np.abs(decoded - true_obs) / scale).mean(-1)

        metrics = {
            "audit/closed_loop/dream_reward_per_step": _mean(dream_step),
            "audit/closed_loop/true_reward_per_step": _mean(true_step),
            "audit/closed_loop/reward_gap_per_step": _mean(dream_step) - _mean(true_step),
            "audit/closed_loop/return_corr": _corr(dream_step.sum(-1), true_step.sum(-1)),
            "audit/closed_loop/dream_above_true_frac": _mean(dream_step.sum(-1) > true_step.sum(-1)),
        }

        for horizon in STATE_ERROR_HORIZONS:
            if horizon <= H:
                metrics[f"audit/closed_loop/state_norm_mae_h{horizon}"] = _mean(state_error[:, horizon - 1])

        if self.is_half_cheetah:
            angle = HALF_CHEETAH_ROOT_ANGLE_INDEX
            x_velocity = start_obs[:, HALF_CHEETAH_X_VELOCITY_INDEX]

            metrics.update({
                "audit/closed_loop/true_flipped_at_end_frac": _mean(_wrap_angle(true_obs[:, -1, angle]) > np.pi / 2),
                "audit/closed_loop/dream_flipped_at_end_frac": _mean(_wrap_angle(decoded[:, -1, angle]) > np.pi / 2),
                "audit/closed_loop/prompt_x_velocity": _mean(x_velocity),
            })

            if exists(self.current_x_velocity_p10):
                metrics["audit/closed_loop/prompt_at_current_speed_frac"] = _mean(x_velocity >= self.current_x_velocity_p10)

        return metrics

    # progress

    @staticmethod
    def progress_metrics(
        *,
        loop: int,
        elapsed_seconds: float,
        world_model_updates: int,
        imagination_updates: int,
        world_model_metrics: dict[str, Any],
        imagination_metrics: dict[str, Any],
    ) -> dict[str, float | None]:
        """Re-log loop-indexed metrics against env steps for runs with different loop sizes."""

        metrics = {
            "audit/progress/loop": loop,
            "audit/progress/elapsed_hours": elapsed_seconds / 3600.,
            "audit/progress/world_model_updates": world_model_updates,
            "audit/progress/imagination_updates": imagination_updates,
        }

        for key, value in {**world_model_metrics, **imagination_metrics}.items():
            if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1):
                metrics[f"audit/env_step/{key}"] = float(value)

        return metrics
