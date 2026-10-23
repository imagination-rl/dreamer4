# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "tqdm",
#     "dreamer4",
#     "adam-atan2-pytorch>=0.3.6",
#     "fire",
#     "gymnasium[mujoco]",
#     "tensorboard",
#     "pyyaml",
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

"""State-only on-policy actor-critic benchmark.

This trainer has no tokenizer, world model, replay buffer, or compilation
runtime.  It keeps actions in the policy distribution range and rescales only
the action that goes to Gym.
"""

from __future__ import annotations

import json
import random
import sys
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Mapping

import fire
import gymnasium as gym
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.distributions import Beta, Normal, kl_divergence
from torch.nn import functional as F
from torch.optim import Adam, Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from x_mlps_pytorch.normed_mlp import create_mlp
from torch_einops_utils import lens_to_mask, masked_mean, safe_cat

from dreamer4.dreamer4 import (
    ActionEmbedder,
    Actions,
    DynamicsWorldModel,
    Embeds,
    Experience,
    HLGaussRewardEncoder,
    SymExpTwoHot,
    SymExpValueTwoHot,
    TDMPC2ValueTwoHot,
    calc_gae,
    combine_experiences,
    exists,
    frac_gradient,
    get_activation,
    symexp,
    z_score,
)

try:
    from . import train_gym
except ImportError:  # Support `python train/train_model_free.py`.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from train import train_gym


SmallObjective = Literal["ppo", "pmpo", "reinforce"]
NormalizationMode = Literal["none", "fixed_random", "running"]

SMALL_MODEL_DEFAULTS = {
    "dim": 128,
    "encoder_depth": 2,
    "update_epochs": 1,
    "minibatch_num_envs": None,
    "observation_normalization": "fixed_random",
    "normalization_observations": 16384,
    "cleanrl_updated_ppo": False,
    "cleanrl_pmpo": False,
    "cleanrl_pmpo_constant_learning_rate": False,
    "cleanrl_pmpo_joint_log_prob": False,
    "cleanrl_pmpo_q_ranked_preferences": False,
}


def resolve_small_model_config(config: Mapping[str, Any] | None):
    """Validate and fill the state-only model configuration."""

    if config is None:
        config = {}

    if not isinstance(config, Mapping):
        raise TypeError("small_model must be a mapping")

    unknown = sorted(set(config) - set(SMALL_MODEL_DEFAULTS))
    if unknown:
        raise KeyError(f"small_model contains unknown keys: {unknown}")

    resolved = dict(SMALL_MODEL_DEFAULTS)
    resolved.update(config)

    for name in ("dim", "encoder_depth", "update_epochs", "normalization_observations"):
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"small_model.{name} must be a positive integer")

    minibatch_num_envs = resolved["minibatch_num_envs"]
    if minibatch_num_envs is not None:
        if isinstance(minibatch_num_envs, bool) or not isinstance(minibatch_num_envs, int) or minibatch_num_envs <= 0:
            raise ValueError("small_model.minibatch_num_envs must be null or a positive integer")

    mode = resolved["observation_normalization"]
    if mode not in ("none", "fixed_random", "running"):
        raise ValueError(
            "small_model.observation_normalization must be 'none', "
            "'fixed_random', or 'running'"
        )

    for name in (
        "cleanrl_updated_ppo",
        "cleanrl_pmpo",
        "cleanrl_pmpo_constant_learning_rate",
        "cleanrl_pmpo_joint_log_prob",
        "cleanrl_pmpo_q_ranked_preferences",
    ):
        if not isinstance(resolved[name], bool):
            raise TypeError(f"small_model.{name} must be a boolean")

    return resolved


class ObservationNormalizer(nn.Module):
    """Normalize state observations with fixed or running statistics."""

    def __init__(self, dim: int, mode: NormalizationMode):
        super().__init__()
        self.mode = mode
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("variance", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype = torch.long))

    @property
    def std(self):
        return self.variance.clamp_min(1e-6).sqrt()

    @torch.no_grad()
    def set_fixed_statistics(self, observations: Tensor):
        if observations.ndim != 2:
            raise ValueError("normalization observations must have shape (batch, state)")

        self.mean.copy_(observations.mean(dim = 0).to(self.mean))
        self.variance.copy_(observations.var(dim = 0, correction = 0).to(self.variance).clamp_min(1e-6))
        self.count.fill_(observations.shape[0])

    @torch.no_grad()
    def update(self, observations: Tensor):
        """Update only after an optimizer step to retain behavior ratios."""

        if self.mode != "running" or observations.numel() == 0:
            return

        observations = observations.detach().reshape(-1, observations.shape[-1]).to(self.mean)
        batch_count = observations.shape[0]
        batch_mean = observations.mean(dim = 0)
        batch_variance = observations.var(dim = 0, correction = 0)

        if self.count.item() == 0:
            self.mean.copy_(batch_mean)
            self.variance.copy_(batch_variance.clamp_min(1e-6))
            self.count.fill_(batch_count)
            return

        previous_count = self.count.to(dtype = observations.dtype)
        current_count = torch.tensor(float(batch_count), device = observations.device, dtype = observations.dtype)
        total_count = previous_count + current_count
        delta = batch_mean - self.mean
        next_mean = self.mean + delta * (current_count / total_count)
        merged_second_moment = (
            self.variance * previous_count
            + batch_variance * current_count
            + delta.square() * previous_count * current_count / total_count
        )

        self.mean.copy_(next_mean)
        self.variance.copy_((merged_second_moment / total_count).clamp_min(1e-6))
        self.count.add_(batch_count)

    def forward(self, observations: Tensor):
        if self.mode == "none":
            return observations
        return (observations - self.mean) / self.std


def make_value_encoder(
    *,
    dim: int,
    reward_encoder_type: str,
    use_symexp_value_head: bool,
    use_tdmpc2_value_head: bool,
    use_symlog_hl_gauss_value_head: bool,
    value_range: tuple[float, float],
    value_encoder_num_bins: int,
    value_encoder_sigma_to_bin_ratio: float,
    value_min_max_value_on_bin_center: bool,
    value_use_symlog: bool,
    value_decode_symlog_expected_raw: bool,
):
    special_value_head_count = sum((
        use_symexp_value_head,
        use_tdmpc2_value_head,
        use_symlog_hl_gauss_value_head,
    ))
    if special_value_head_count > 1:
        raise ValueError("special value-head encoder flags are mutually exclusive")

    if use_tdmpc2_value_head:
        value_encoder_class = TDMPC2ValueTwoHot
    elif use_symexp_value_head:
        value_encoder_class = SymExpValueTwoHot
    elif use_symlog_hl_gauss_value_head:
        value_encoder_class = HLGaussRewardEncoder
    elif reward_encoder_type == "hl_gauss":
        value_encoder_class = HLGaussRewardEncoder
    elif reward_encoder_type == "symexp_two_hot":
        value_encoder_class = SymExpTwoHot
    else:
        raise ValueError(f"unknown reward_encoder_type {reward_encoder_type!r}")

    encoder_kwargs = dict(reward_range = value_range, num_bins = value_encoder_num_bins)
    if value_encoder_class is HLGaussRewardEncoder:
        encoder_kwargs.update(
            sigma_to_bin_ratio = value_encoder_sigma_to_bin_ratio,
            min_max_value_on_bin_center = value_min_max_value_on_bin_center,
            use_symlog = value_use_symlog,
            decode_symlog_expected_raw = value_decode_symlog_expected_raw,
        )

    if use_symlog_hl_gauss_value_head:
        encoder_kwargs.update(use_symlog = True, decode_symlog_expected_raw = True)

    return value_encoder_class(**encoder_kwargs, dim_embed = dim, learned_embedding = False)


def initialize_value_head(
    value_head: nn.Module,
    value_encoder: nn.Module,
    *,
    use_symexp_value_head: bool,
    use_tdmpc2_value_head: bool,
    use_symlog_hl_gauss_value_head: bool,
    use_baseline_symexp_value_head_init: bool,
    old_value_head: bool,
    value_head_init_value: float | None,
):
    zero_initialize = (
        use_tdmpc2_value_head
        or use_symlog_hl_gauss_value_head
        or (use_symexp_value_head and not use_baseline_symexp_value_head_init)
    )
    final_linear = next((module for module in reversed(list(value_head.modules())) if isinstance(module, nn.Linear)), None)

    if zero_initialize and exists(final_linear):
        with torch.no_grad():
            final_linear.weight.zero_()
            if exists(final_linear.bias):
                final_linear.bias.zero_()

    if old_value_head and value_head_init_value is not None and exists(final_linear):
        with torch.no_grad():
            initial_value = torch.tensor(value_head_init_value, dtype = final_linear.weight.dtype, device = final_linear.weight.device)
            initial_probs = value_encoder(initial_value.reshape(1)).reshape(-1).clamp_min(1e-8)
            initial_logits = initial_probs.log()
            initial_logits = initial_logits - initial_logits.mean()
            final_linear.weight.zero_()
            if exists(final_linear.bias):
                final_linear.bias.copy_(initial_logits)


class StateActorCritic(nn.Module):
    """A state MLP with Dreamer's policy distribution and categorical value head."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        action_range: tuple[float, float],
        continuous_dist_type: Literal["gaussian", "squashed_gaussian", "beta"] = "beta",
        dim: int,
        encoder_depth: int,
        observation_normalization: NormalizationMode,
        reward_encoder_type: str,
        use_symexp_value_head: bool,
        use_tdmpc2_value_head: bool,
        use_symlog_hl_gauss_value_head: bool,
        use_hl_gauss_actor_critic: bool,
        use_baseline_symexp_value_head_init: bool,
        value_encoder_num_bins: int,
        value_range: tuple[float, float],
        value_encoder_sigma_to_bin_ratio: float,
        value_min_max_value_on_bin_center: bool,
        value_use_symlog: bool,
        value_decode_symlog_expected_raw: bool,
        hl_gauss_actor_critic_range: tuple[float, float],
        hl_gauss_actor_critic_num_bins: int,
        hl_gauss_actor_critic_sigma_to_bin_ratio: float,
        hl_gauss_actor_critic_min_max_value_on_bin_center: bool,
        normalize_advantages: bool,
        pmpo_pos_to_neg_weight: float,
        pmpo_kl_div_loss_weight: float,
        use_delight_gating: bool,
        old_continuous_action_encoder: bool,
        old_pmpo: bool,
        old_policy_head: bool,
        old_value_head: bool,
        value_head_init_value: float | None,
        reinforce_discount_factor: float = 0.99,
        reinforce_gae_lambda: float = 1.,
        reinforce_return_normalization: Literal["none", "percentile"] = "none",
        reinforce_use_discounted_trajectory_weights: bool = False,
        policy_entropy_weight: float = 0.01,
    ):
        super().__init__()

        if use_symexp_value_head and old_value_head:
            raise ValueError("use_symexp_value_head cannot use old_value_head")
        if use_tdmpc2_value_head and old_value_head:
            raise ValueError("use_tdmpc2_value_head cannot use old_value_head")
        if use_symlog_hl_gauss_value_head and old_value_head:
            raise ValueError("use_symlog_hl_gauss_value_head cannot use old_value_head")
        if use_baseline_symexp_value_head_init and not use_symexp_value_head:
            raise ValueError("use_baseline_symexp_value_head_init requires use_symexp_value_head")
        if use_hl_gauss_actor_critic and not use_symexp_value_head:
            raise ValueError("use_hl_gauss_actor_critic requires use_symexp_value_head")

        self.observation_normalizer = ObservationNormalizer(obs_dim, observation_normalization)

        encoder_layers: list[nn.Module] = []
        encoder_input_dim = obs_dim
        for _ in range(encoder_depth):
            encoder_layers.extend((nn.Linear(encoder_input_dim, dim * 4), nn.SiLU()))
            encoder_input_dim = dim * 4
        encoder_layers.append(nn.Linear(encoder_input_dim, dim))
        self.state_mlp = nn.Sequential(*encoder_layers)

        self.policy_head = create_mlp(
            dim_in = dim,
            dim = dim * 4,
            dim_out = dim * 4,
            depth = 2,
            activation = get_activation("relu_squared" if old_policy_head else "silu"),
        )
        self.action_embedder = ActionEmbedder(
            dim = dim,
            num_discrete_actions = 0,
            num_continuous_actions = action_dim,
            continuous_dist_type = continuous_dist_type,
            continuous_target_action_range = action_range,
            old_continuous_action_encoder = old_continuous_action_encoder,
            can_unembed = True,
            unembed_dim = dim * 4,
            num_unembed_preds = 1,
            squeeze_unembed_preds = False,
        )

        self.value_encoder = make_value_encoder(
            dim = dim,
            reward_encoder_type = reward_encoder_type,
            use_symexp_value_head = use_symexp_value_head,
            use_tdmpc2_value_head = use_tdmpc2_value_head,
            use_symlog_hl_gauss_value_head = use_symlog_hl_gauss_value_head,
            value_range = value_range,
            value_encoder_num_bins = value_encoder_num_bins,
            value_encoder_sigma_to_bin_ratio = value_encoder_sigma_to_bin_ratio,
            value_min_max_value_on_bin_center = value_min_max_value_on_bin_center,
            value_use_symlog = value_use_symlog,
            value_decode_symlog_expected_raw = value_decode_symlog_expected_raw,
        )
        self.value_head = create_mlp(
            dim_in = dim,
            dim = dim * 4,
            dim_out = self.value_encoder.num_bins,
            depth = 2,
            activation = get_activation("relu_squared" if old_value_head else "silu"),
        )
        initialize_value_head(
            self.value_head,
            self.value_encoder,
            use_symexp_value_head = use_symexp_value_head,
            use_tdmpc2_value_head = use_tdmpc2_value_head,
            use_symlog_hl_gauss_value_head = use_symlog_hl_gauss_value_head,
            use_baseline_symexp_value_head_init = use_baseline_symexp_value_head_init,
            old_value_head = old_value_head,
            value_head_init_value = value_head_init_value,
        )

        self.use_hl_gauss_actor_critic = use_hl_gauss_actor_critic
        self.actor_critic_value_encoder = None
        self.actor_critic_value_head = None
        if use_hl_gauss_actor_critic:
            self.actor_critic_value_encoder = HLGaussRewardEncoder(
                reward_range = hl_gauss_actor_critic_range,
                num_bins = hl_gauss_actor_critic_num_bins,
                sigma_to_bin_ratio = hl_gauss_actor_critic_sigma_to_bin_ratio,
                min_max_value_on_bin_center = hl_gauss_actor_critic_min_max_value_on_bin_center,
                use_symlog = False,
                decode_symlog_expected_raw = False,
                dim_embed = dim,
                learned_embedding = False,
            )
            self.actor_critic_value_head = create_mlp(
                dim_in = dim,
                dim = dim * 4,
                dim_out = self.actor_critic_value_encoder.num_bins,
                depth = 2,
                activation = get_activation("silu"),
            )

        self.normalize_advantages = normalize_advantages
        self.policy_all_tokens = False
        self.critic_state_embedder = None
        self.keep_reward_ema_stats = False
        if reinforce_return_normalization not in ("none", "percentile"):
            raise ValueError("reinforce_return_normalization must be 'none' or 'percentile'")

        self.reinforce_discount_factor = reinforce_discount_factor
        self.reinforce_gae_lambda = reinforce_gae_lambda
        self.reinforce_return_normalization = reinforce_return_normalization
        self.reinforce_use_discounted_trajectory_weights = reinforce_use_discounted_trajectory_weights
        if reinforce_return_normalization == "percentile":
            self.register_buffer("reinforce_return_low", torch.tensor(0.))
            self.register_buffer("reinforce_return_high", torch.tensor(0.))
        else:
            self.reinforce_return_low = None
            self.reinforce_return_high = None
        self.actor_spr = False
        self.agent_policy_gradient_frac = 1.
        self.agent_value_gradient_frac = 1.
        self.max_steps = 1
        self.old_pmpo = old_pmpo
        self.pmpo_pos_to_neg_weight = pmpo_pos_to_neg_weight
        self.pmpo_kl_div_loss_weight = pmpo_kl_div_loss_weight
        self.pmpo_reverse_kl = True
        self.ppo_eps_clip = 0.2
        self.value_clip = 0.4
        self.clip_values = False
        self.policy_entropy_weight = policy_entropy_weight
        self.gae_discount_factor = 0.99
        self.gae_lambda = 0.95
        self.gae_use_accelerated = False
        self.use_delight_gating = use_delight_gating
        self.delight_temperature = 1.
        self.register_buffer("zero", torch.zeros(()), persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_states(self, states: Tensor):
        return self.state_mlp(self.observation_normalizer(states.float()))

    def forward(self, *, latents: Tensor, **_kwargs):
        """Provide the feature contract consumed by Dreamer's loss routine."""

        state_features = self.encode_states(latents)
        agent_features = state_features.unsqueeze(2)
        return None, (Embeds(agent_features, None, None, None), None)

    def policy_parameters(self):
        return [*self.policy_head.parameters(), *self.action_embedder.unembed_parameters()]

    def value_parameters(self):
        params = list(self.value_head.parameters())
        if exists(self.actor_critic_value_head):
            params.extend(self.actor_critic_value_head.parameters())
        return params

    def optimizer_parameters(self):
        return [*self.state_mlp.parameters(), *self.policy_parameters(), *self.value_parameters()]


def _align_time(tensor: Tensor | None, time: int):
    if not exists(tensor):
        return None
    return tensor[:, :time]


def _state_actor_critic_losses_reference(
    agent: StateActorCritic,
    experience: Experience,
    *,
    objective: SmallObjective,
    use_delight_gating: bool | None = None,
    normalize_advantages: bool | None = None,
    return_diagnostics: bool = False,
    eps: float = 1e-6,
):
    """Use the Dreamer policy, value, and entropy formulas on states."""

    if objective not in ("ppo", "pmpo", "reinforce"):
        raise ValueError("small_model_only supports objective='ppo', objective='pmpo', and objective='reinforce'; objective='spo' is not supported")

    exp = experience.to(agent.device)
    if not all((exists(exp.actions), exists(exp.log_probs), exists(exp.values), exists(exp.rewards), exists(exp.step_size))):
        raise ValueError("state experience needs actions, log probabilities, values, rewards, and a step size")

    states = exp.latents
    batch, time = states.shape[:2]
    old_values = exp.values
    rewards = exp.rewards
    lens = exp.lens
    if not exists(lens):
        lens = torch.full((batch,), time, device = agent.device, dtype = torch.long)
    is_truncated = exp.is_truncated
    if not exists(is_truncated):
        is_truncated = torch.ones(batch, device = agent.device, dtype = torch.bool)

    mask_for_gae = lens_to_mask(lens, time)
    rewards = rewards.masked_fill(~mask_for_gae, 0.)
    old_values = old_values.masked_fill(~mask_for_gae, 0.)
    learnable_lens = lens - is_truncated.long()
    mask = lens_to_mask(learnable_lens.clamp(min = 0), time)
    gae_masks = lens_to_mask((lens - 1).clamp(min = 0), time)

    if exists(exp.terminals):
        terminals = exp.terminals
        if terminals.ndim == 1:
            sequence = torch.arange(time, device = agent.device)[None, :]
            terminals = (sequence == (lens - 1).clamp(min = 0)[:, None]) & terminals[:, None]
        gae_masks.masked_fill_(terminals.bool(), 0.)

    discount_factor = agent.reinforce_discount_factor if objective == "reinforce" else agent.gae_discount_factor
    gae_lambda = agent.reinforce_gae_lambda if objective == "reinforce" else agent.gae_lambda
    returns = calc_gae(
        rewards,
        old_values,
        masks = gae_masks,
        learn_masks = mask,
        gamma = discount_factor,
        lam = gae_lambda,
        use_accelerated = agent.gae_use_accelerated,
    )
    if objective == "reinforce" and agent.reinforce_return_normalization == "percentile":
        return_quantiles = torch.quantile(
            returns[mask].detach().float(),
            torch.tensor((0.05, 0.95), device = returns.device),
        )
        with torch.no_grad():
            agent.reinforce_return_low.lerp_(return_quantiles[0], 0.01)
            agent.reinforce_return_high.lerp_(return_quantiles[1], 0.01)
        return_scale = (agent.reinforce_return_high - agent.reinforce_return_low).clamp_min(1.)
        advantage = (returns - old_values) / return_scale
    else:
        advantage = returns - old_values
    normalize_advantages = agent.normalize_advantages if normalize_advantages is None else normalize_advantages
    if objective != "pmpo" and normalize_advantages:
        advantage = z_score(advantage, mask = mask, eps = eps)

    features = agent.encode_states(states)
    policy_embed = agent.policy_head(frac_gradient(features, 1.))
    policy_time = policy_embed.shape[1]
    actions = exp.actions
    continuous_actions = _align_time(actions.continuous, policy_time)
    old_log_probs = _align_time(exp.log_probs.continuous, policy_time)
    log_probs, entropies = agent.action_embedder.log_probs(
        policy_embed,
        pred_head_index = 0,
        continuous_targets = continuous_actions,
        return_entropies = True,
    )
    log_probs = safe_cat(log_probs, dim = -1).sum(dim = -1)
    old_log_probs = safe_cat(Actions(None, old_log_probs), dim = -1).sum(dim = -1)
    entropies = safe_cat(entropies, dim = -1)

    advantage = advantage[:, :policy_time]
    mask = mask[:, :policy_time]
    importance_ratio = (log_probs - old_log_probs).exp()
    use_delight_gating = agent.use_delight_gating if use_delight_gating is None else use_delight_gating
    delight_gate = None
    if use_delight_gating:
        delight_gate = ((-log_probs * advantage) / agent.delight_temperature).sigmoid().detach()

    if objective == "pmpo":
        if agent.old_pmpo:
            pos_advantage_mask = advantage > 0.
            neg_advantage_mask = advantage < 0.
        else:
            pos_advantage_mask = advantage >= 0.
            neg_advantage_mask = ~pos_advantage_mask

        maybe_gated_log_probs = log_probs if not exists(delight_gate) else log_probs * delight_gate
        pos = pos_advantage_mask & mask
        neg = neg_advantage_mask & mask
        if agent.old_pmpo:
            pos_loss = masked_mean(maybe_gated_log_probs, pos)
            neg_loss = masked_mean(maybe_gated_log_probs, neg)
            policy_loss = -agent.pmpo_pos_to_neg_weight * pos_loss + (1. - agent.pmpo_pos_to_neg_weight) * neg_loss
        else:
            scaled_log_probs = maybe_gated_log_probs * advantage.tanh().abs()
            pos_loss = scaled_log_probs.masked_fill(~pos, 0.).sum()
            neg_loss = scaled_log_probs.masked_fill(~neg, 0.).sum()
            policy_loss = -agent.pmpo_pos_to_neg_weight * (pos_loss - neg_loss) / mask.sum().clamp(min = 1).to(scaled_log_probs.dtype)

        if agent.pmpo_kl_div_loss_weight > 0.:
            if not exists(exp.old_action_unembeds):
                raise ValueError("PMPO state experience needs behavior action parameters")
            new_action_unembeds = agent.action_embedder.unembed(policy_embed, pred_head_index = 0)
            old_action_unembeds = Actions(*(_align_time(value, policy_time) for value in exp.old_action_unembeds))
            kl_inputs, kl_targets = new_action_unembeds, old_action_unembeds
            if agent.pmpo_reverse_kl:
                kl_inputs, kl_targets = kl_targets, kl_inputs
            _, continuous_kl = agent.action_embedder.kl_div(kl_inputs, kl_targets)
            if exists(continuous_kl):
                policy_loss = policy_loss + masked_mean(continuous_kl, mask) * agent.pmpo_kl_div_loss_weight
    elif objective == "ppo":
        clipped_ratio = importance_ratio.clamp(1. - agent.ppo_eps_clip, 1. + agent.ppo_eps_clip)
        policy_loss = -torch.min(importance_ratio * advantage, clipped_ratio * advantage)
        if exists(delight_gate):
            policy_loss = policy_loss * delight_gate
        policy_loss = masked_mean(policy_loss, mask)
    else:
        policy_loss = -(log_probs * advantage.detach())
        reinforce_trajectory_weights = None
        if agent.reinforce_use_discounted_trajectory_weights:
            reinforce_trajectory_weights = agent.reinforce_discount_factor ** torch.arange(
                log_probs.shape[1],
                device = log_probs.device,
                dtype = log_probs.dtype,
            )
            policy_loss = policy_loss * reinforce_trajectory_weights
        if exists(delight_gate):
            policy_loss = policy_loss * delight_gate
        policy_loss = masked_mean(policy_loss, mask)

    entropy_loss = -entropies.sum(dim = -1)
    if objective == "reinforce" and exists(reinforce_trajectory_weights):
        entropy_loss = entropy_loss * reinforce_trajectory_weights
    entropy_loss = masked_mean(entropy_loss, mask)
    total_policy_loss = policy_loss + entropy_loss * agent.policy_entropy_weight

    value_bins = agent.value_head(frac_gradient(features, 1.))
    values = agent.value_encoder.bins_to_scalar_value(value_bins)
    value_time = value_bins.shape[1]
    value_returns = returns[:, :value_time]
    value_mask = mask[:, :value_time]
    value_old_values = old_values[:, :value_time]
    value_targets = agent.value_encoder(value_returns)
    value_bins_t, value_targets_t = (tensor.transpose(1, 2) for tensor in (value_bins, value_targets))
    value_loss = -(value_targets_t * value_bins_t.log_softmax(dim = 1)).sum(dim = 1)

    if agent.clip_values:
        clipped_values = value_old_values + (values - value_old_values).clamp(-agent.value_clip, agent.value_clip)
        clipped_bins = agent.value_encoder(clipped_values).transpose(1, 2)
        clipped_loss = -(value_targets_t * clipped_bins.log_softmax(dim = 1)).sum(dim = 1)
        value_loss = torch.maximum(value_loss, clipped_loss)

    value_loss = value_loss[value_mask].mean()
    if agent.use_hl_gauss_actor_critic:
        actor_critic_bins = agent.actor_critic_value_head(features)
        actor_critic_targets = agent.actor_critic_value_encoder(value_returns)
        actor_critic_loss = -(actor_critic_targets.transpose(1, 2) * actor_critic_bins.transpose(1, 2).log_softmax(dim = 1)).sum(dim = 1)
        value_loss = value_loss + actor_critic_loss[value_mask].mean()

    if not return_diagnostics:
        return total_policy_loss, value_loss

    valid_values = values[value_mask]
    valid_returns = value_returns[value_mask]
    diagnostics = {
        "critic/gae_return_mean": valid_returns.mean().detach(),
        "critic/value_mean": valid_values.mean().detach(),
        "critic/value_abs_error": (valid_values - valid_returns).abs().mean().detach(),
        "extra/policy_entropy": (-entropy_loss).detach(),
        "extra/importance_ratio_clip_frac": (((importance_ratio - 1.).abs() > agent.ppo_eps_clip) & mask).float().sum().div(mask.sum().clamp(min = 1)).detach(),
    }
    return total_policy_loss, value_loss, diagnostics


def actor_critic_losses(
    agent: StateActorCritic,
    experience: Experience,
    *,
    objective: SmallObjective,
    use_delight_gating: bool | None = None,
    normalize_advantages: bool | None = None,
    return_diagnostics: bool = False,
    eps: float = 1e-6,
):
    """Call Dreamer's unchanged actor-critic routine on state-MLP features."""

    if objective not in ("ppo", "pmpo", "reinforce"):
        raise ValueError("small_model_only supports objective='ppo', objective='pmpo', and objective='reinforce'; objective='spo' is not supported")

    # The state model supplies the embeddings through ``forward``.  This uses
    # DynamicsWorldModel.learn_from_experience without constructing a world model.
    return DynamicsWorldModel.learn_from_experience(
        agent,
        experience,
        only_learn_policy_value_heads = False,
        objective = objective,
        use_delight_gating = use_delight_gating,
        normalize_advantages = normalize_advantages,
        return_diagnostics = return_diagnostics,
        eps = eps,
    )


@torch.no_grad()
def interact_with_env(
    agent: StateActorCritic,
    env,
    *,
    seed: int | None,
    max_timesteps: int,
):
    """Collect one vector batch and append a value bootstrap after truncation."""

    observations = train_gym.reset_env(env, seed = seed)
    states = [torch.as_tensor(train_gym.obs_array(observations), device = agent.device)]
    batch = states[0].shape[0]

    rewards: list[Tensor] = []
    actions: list[Tensor] = []
    log_probs: list[Tensor] = []
    action_unembeds: list[Tensor] = []
    values: list[Tensor] = []
    terminals = torch.zeros(batch, device = agent.device, dtype = torch.bool)
    truncated = torch.zeros(batch, device = agent.device, dtype = torch.bool)
    done = torch.zeros(batch, device = agent.device, dtype = torch.bool)
    lens = torch.zeros(batch, device = agent.device, dtype = torch.long)

    for step in range(max_timesteps):
        state = states[-1]
        features = agent.encode_states(state).unsqueeze(1)
        policy_embed = agent.policy_head(features)
        action, = (agent.action_embedder.sample(policy_embed, pred_head_index = 0, squeeze = True)[1],)
        value = agent.value_encoder.bins_to_scalar_value(agent.value_head(features))
        log_prob = agent.action_embedder.log_probs(
            policy_embed,
            pred_head_index = 0,
            continuous_targets = action,
        )[1]
        action_params = agent.action_embedder.unembed(policy_embed, pred_head_index = 0)[1]

        environment_action = agent.action_embedder.rescale_for_env(action).clamp(*agent.action_embedder.continuous_target_action_range)
        next_observations, reward, was_terminated, was_truncated, *_ = env.step(environment_action[:, 0].cpu().numpy())
        was_terminated = torch.as_tensor(np.asarray(was_terminated), device = agent.device, dtype = torch.bool).reshape(batch)
        was_truncated = torch.as_tensor(np.asarray(was_truncated), device = agent.device, dtype = torch.bool).reshape(batch)
        reward = torch.as_tensor(np.asarray(reward), device = agent.device, dtype = torch.float32).reshape(batch, 1)

        lens = torch.where(done, lens, lens + 1)
        terminals |= was_terminated
        truncated |= was_truncated
        if step + 1 >= max_timesteps:
            truncated |= ~terminals
        done |= terminals | truncated

        rewards.append(reward)
        actions.append(action)
        log_probs.append(log_prob)
        action_unembeds.append(action_params)
        values.append(value)
        states.append(torch.as_tensor(train_gym.obs_array(next_observations), device = agent.device))

        if done.all():
            break

    need_bootstrap = truncated & ~terminals
    if need_bootstrap.any():
        bootstrap_features = agent.encode_states(states[-1]).unsqueeze(1)
        bootstrap_value = agent.value_encoder.bins_to_scalar_value(agent.value_head(bootstrap_features))
        bootstrap_action_params = agent.action_embedder.unembed(agent.policy_head(bootstrap_features), pred_head_index = 0)[1]
        values.append(bootstrap_value)
        action_unembeds.append(bootstrap_action_params)
        rewards.append(torch.zeros_like(rewards[-1]))
        actions.append(torch.zeros_like(actions[-1]))
        log_probs.append(torch.zeros_like(log_probs[-1]))
        lens = lens + need_bootstrap.long()

    stacked_states = torch.stack(states, dim = 1)[:, :len(rewards)]
    return Experience(
        latents = stacked_states,
        critic_state = stacked_states,
        rewards = torch.cat(rewards, dim = 1),
        terminals = terminals,
        actions = Actions(None, torch.cat(actions, dim = 1)),
        log_probs = Actions(None, torch.cat(log_probs, dim = 1)),
        old_action_unembeds = Actions(None, torch.cat(action_unembeds, dim = 1)),
        values = torch.cat(values, dim = 1),
        step_size = 1,
        lens = lens,
        is_truncated = truncated,
        is_from_world_model = False,
        episode_return = (torch.cat(rewards, dim = 1) * lens_to_mask(lens, len(rewards)).float()).sum(dim = 1),
    )


def select_environment_batch(experience: Experience, indices: Tensor):
    """Select environments without changing the time axis or GAE sequence."""

    batch_size = experience.latents.shape[0]
    values = {}
    for field in fields(Experience):
        value = getattr(experience, field.name)
        if isinstance(value, Actions):
            values[field.name] = Actions(*(
                item.index_select(0, indices) if exists(item) else None
                for item in value
            ))
        elif torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            values[field.name] = value.index_select(0, indices)
        else:
            values[field.name] = value
    return Experience(**values)


def valid_states(experience: Experience):
    mask = lens_to_mask(experience.lens, experience.latents.shape[1])
    return experience.latents[mask]


def train_on_experience(
    agent: StateActorCritic,
    optimizer: Optimizer,
    experience: Experience,
    *,
    objective: SmallObjective,
    update_epochs: int,
    minibatch_num_envs: int | None,
    max_grad_norm: float,
    update_step: int,
    writer: SummaryWriter | None,
    env_step: int,
):
    last_metrics = {}
    batch = experience.latents.shape[0]
    minibatch_size = batch if minibatch_num_envs is None else min(minibatch_num_envs, batch)
    grad_clip_groups = train_gym.agent_clip_groups(agent, optimizer)

    for _ in range(update_epochs):
        indices = torch.randperm(batch, device = experience.latents.device)
        for start in range(0, batch, minibatch_size):
            mini_experience = select_environment_batch(experience, indices[start:start + minibatch_size])
            policy_loss, value_loss, diagnostics = actor_critic_losses(
                agent,
                mini_experience,
                objective = objective,
                return_diagnostics = True,
            )
            loss = policy_loss + value_loss
            loss.backward()
            grad_norm, grad_metrics = train_gym.clip_grad_norm_by_group(grad_clip_groups, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            # Do this after the pass.  The current ratio uses the prior state statistics.
            agent.observation_normalizer.update(valid_states(mini_experience))
            last_metrics = {
                "state_actor_critic/loss": loss.item(),
                "state_actor_critic/policy_loss": policy_loss.item(),
                "state_actor_critic/value_loss": value_loss.item(),
                "state_actor_critic/grad_norm": grad_norm,
                **{
                    f"state_actor_critic/{key}": value
                    for key, value in grad_metrics.items()
                },
                **{key: value.item() for key, value in diagnostics.items()},
            }
            train_gym.log_scalars(writer, last_metrics, env_step)
            update_step += 1

    return update_step, last_metrics


def save_checkpoint(
    path: Path,
    *,
    agent: StateActorCritic,
    optimizer: Optimizer,
    loop: int,
    update_step: int,
    env_step: int,
):
    path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(
        {
            "checkpoint_type": "state_actor_critic",
            "agent": agent.state_dict(),
            "agent_optimizer": train_gym.optimizer_state_dict_for_checkpoint(optimizer),
            "observation_normalizer": agent.observation_normalizer.state_dict(),
            "loop": loop,
            "update_step": update_step,
            "env_step": env_step,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    agent: StateActorCritic,
    optimizer: Optimizer,
):
    checkpoint = torch.load(path, map_location = agent.device, weights_only = False)
    if checkpoint.get("checkpoint_type") != "state_actor_critic":
        raise ValueError(
            "checkpoint type error: expected a state_actor_critic checkpoint. "
            "Dreamer tokenizer and world-model checkpoints are incompatible."
        )
    agent.load_state_dict(checkpoint["agent"])
    if "observation_normalizer" in checkpoint:
        agent.observation_normalizer.load_state_dict(checkpoint["observation_normalizer"])
    train_gym.load_optimizer_state_if_compatible(optimizer, checkpoint["agent_optimizer"], "state actor-critic")
    return checkpoint


def cleanrl_layer_init(layer: nn.Linear, std = np.sqrt(2), bias_const = 0.):
    """Initialize one CleanRL linear layer."""

    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# Match the algorithm defaults in CarsonBurke/cleanrl's ppo_continuous_action.py.
CLEANRL_UPDATED_PPO_GAMMA = .99
CLEANRL_UPDATED_PPO_GAE_LAMBDA = .95
CLEANRL_UPDATED_PPO_NUM_MINIBATCHES = 32
CLEANRL_UPDATED_PPO_UPDATE_EPOCHS = 10
CLEANRL_UPDATED_PPO_CLIP_COEF = .2
CLEANRL_UPDATED_PPO_VALUE_LOSS_WEIGHT = .5
CLEANRL_UPDATED_PPO_ENTROPY_WEIGHT = 0.
CLEANRL_UPDATED_PPO_MAX_GRAD_NORM = .5
CLEANRL_UPDATED_PPO_SAMPLE_EPS = 1e-6


class CleanRLUpdatedPPOAgent(nn.Module):
    """The bounded-action Beta actor and critic from the updated CleanRL PPO."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
    ):
        super().__init__()

        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("CleanRL updated PPO requires a flat Box observation space")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("CleanRL updated PPO requires a Box action space")

        action_low = np.asarray(action_space.low, dtype = np.float32).reshape(-1)
        action_high = np.asarray(action_space.high, dtype = np.float32).reshape(-1)
        if not (
            np.isfinite(action_low).all()
            and np.isfinite(action_high).all()
            and np.all(action_high > action_low)
        ):
            raise ValueError("CleanRL updated PPO requires finite, ordered action bounds")

        self.action_shape = tuple(action_space.shape)
        self.action_dim = int(np.prod(action_space.shape))
        observation_dim = int(np.prod(observation_space.shape))
        self.register_buffer("action_low", torch.as_tensor(action_low))
        self.register_buffer("action_high", torch.as_tensor(action_high))
        self.register_buffer("action_scale", self.action_high - self.action_low)
        self.register_buffer("log_action_scale", self.action_scale.log())
        self.critic = nn.Sequential(
            cleanrl_layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            cleanrl_layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            cleanrl_layer_init(nn.Linear(64, 1), std = 1.),
        )
        self.actor = nn.Sequential(
            cleanrl_layer_init(nn.Linear(observation_dim, 64)),
            nn.Tanh(),
            cleanrl_layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            cleanrl_layer_init(nn.Linear(64, 2 * self.action_dim), std = .01),
        )

    def get_value(self, observations: Tensor):
        return self.critic(observations.float())

    def get_policy_and_value(self, observations: Tensor):
        concentrations = F.softplus(self.actor(observations.float())) + 1.
        alpha, beta = concentrations.chunk(2, dim = -1)
        return alpha, beta, self.critic(observations.float())

    def action_log_prob(self, alpha: Tensor, beta: Tensor, native_action: Tensor):
        distribution = Beta(alpha, beta, validate_args = False)
        return (distribution.log_prob(native_action) - self.log_action_scale).sum(dim = -1)

    def sample_action_and_value(self, observations: Tensor):
        """Sample native actions and return their physical action equivalents."""

        alpha, beta, value = self.get_policy_and_value(observations)
        distribution = Beta(alpha, beta, validate_args = False)
        native_action = distribution.sample().clamp(
            CLEANRL_UPDATED_PPO_SAMPLE_EPS,
            1. - CLEANRL_UPDATED_PPO_SAMPLE_EPS,
        )
        action = self.action_low + self.action_scale * native_action
        log_prob = (distribution.log_prob(native_action) - self.log_action_scale).sum(dim = -1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(dim = -1)
        action = action.reshape((observations.shape[0],) + self.action_shape)
        return native_action, action, log_prob, entropy, value

    def get_action_and_value(self, observations: Tensor, action: Tensor | None = None):
        """Use physical actions at this public interface."""

        alpha, beta, value = self.get_policy_and_value(observations)
        distribution = Beta(alpha, beta, validate_args = False)
        if action is None:
            native_action = distribution.sample().clamp(
                CLEANRL_UPDATED_PPO_SAMPLE_EPS,
                1. - CLEANRL_UPDATED_PPO_SAMPLE_EPS,
            )
            action = self.action_low + self.action_scale * native_action
            action = action.reshape((observations.shape[0],) + self.action_shape)
        else:
            native_action = (
                (action.reshape(observations.shape[0], -1) - self.action_low)
                / self.action_scale
            ).clamp(
                CLEANRL_UPDATED_PPO_SAMPLE_EPS,
                1. - CLEANRL_UPDATED_PPO_SAMPLE_EPS,
            )

        log_prob = (distribution.log_prob(native_action) - self.log_action_scale).sum(dim = -1)
        entropy = (distribution.entropy() + self.log_action_scale).sum(dim = -1)
        return action, log_prob, entropy, value


def cleanrl_updated_ppo_loss(
    agent: CleanRLUpdatedPPOAgent,
    *,
    observations: Tensor,
    native_actions: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    old_values: Tensor,
    normalize_advantages: bool = True,
    clip_coef: float = CLEANRL_UPDATED_PPO_CLIP_COEF,
    clip_value_loss: bool = True,
    entropy_weight: float = CLEANRL_UPDATED_PPO_ENTROPY_WEIGHT,
    value_loss_weight: float = CLEANRL_UPDATED_PPO_VALUE_LOSS_WEIGHT,
):
    """Calculate the updated CleanRL clipped PPO loss."""

    alpha, beta, new_values = agent.get_policy_and_value(observations)
    distribution = Beta(alpha, beta, validate_args = False)
    new_log_probs = (
        distribution.log_prob(native_actions) - agent.log_action_scale
    ).sum(dim = -1)
    entropy = (distribution.entropy() + agent.log_action_scale).sum(dim = -1)
    log_ratio = new_log_probs - old_log_probs
    ratio = log_ratio.exp()

    with torch.no_grad():
        old_approx_kl = (-log_ratio).mean()
        approx_kl = ((ratio - 1.) - log_ratio).mean()
        clip_fraction = ((ratio - 1.).abs() > clip_coef).float().mean()

    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss_unclipped = -advantages * ratio
    policy_loss_clipped = -advantages * ratio.clamp(1. - clip_coef, 1. + clip_coef)
    policy_loss = torch.maximum(policy_loss_unclipped, policy_loss_clipped).mean()

    new_values = new_values.view(-1)
    if clip_value_loss:
        value_loss_unclipped = (new_values - returns).square()
        clipped_values = old_values + (new_values - old_values).clamp(-clip_coef, clip_coef)
        value_loss_clipped = (clipped_values - returns).square()
        value_loss = .5 * torch.maximum(value_loss_unclipped, value_loss_clipped).mean()
    else:
        value_loss = .5 * (new_values - returns).square().mean()

    entropy_value = entropy.mean()
    loss = policy_loss - entropy_weight * entropy_value + value_loss_weight * value_loss
    metrics = torch.stack((
        policy_loss.detach(),
        value_loss.detach(),
        entropy_value.detach(),
        old_approx_kl,
        approx_kl,
        clip_fraction,
    ))
    return loss, metrics


def cleanrl_updated_ppo_gae(
    rewards: Tensor,
    values: Tensor,
    terminations: Tensor,
    truncations: Tensor,
    truncation_bootstrap_values: Tensor,
    rollout_tail_value: Tensor,
    *,
    gamma: float = CLEANRL_UPDATED_PPO_GAMMA,
    gae_lambda: float = CLEANRL_UPDATED_PPO_GAE_LAMBDA,
):
    """Calculate GAE with separate bootstrap and trace boundary rules."""

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rollout_tail_value)
    for step in reversed(range(rewards.shape[0])):
        ordinary_next_value = (
            rollout_tail_value
            if step == rewards.shape[0] - 1
            else values[step + 1]
        )
        next_value = torch.where(
            truncations[step].bool(),
            truncation_bootstrap_values[step],
            ordinary_next_value,
        )
        bootstrap_nonterminal = 1. - terminations[step]
        trace_nonterminal = 1. - torch.maximum(
            terminations[step],
            truncations[step],
        )
        delta = rewards[step] + gamma * bootstrap_nonterminal * next_value - values[step]
        last_advantage = delta + gamma * gae_lambda * trace_nonterminal * last_advantage
        advantages[step] = last_advantage

    return advantages, advantages + values


def cleanrl_updated_ppo_bootstrap_observations(
    next_observations: np.ndarray,
    truncations: np.ndarray,
    infos: Mapping[str, Any],
):
    """Replace autoreset observations with final observations at time limits."""

    bootstrap_observations = np.array(next_observations, copy = True)
    truncations = np.asarray(truncations, dtype = bool)
    if not np.any(truncations):
        return bootstrap_observations

    final_observations = infos.get("final_observation", infos.get("final_obs"))
    final_mask = infos.get("_final_observation", infos.get("_final_obs"))
    if final_observations is None:
        raise RuntimeError("truncated transition lacks infos['final_observation']")

    for env_index in np.flatnonzero(truncations):
        if final_mask is not None and not final_mask[env_index]:
            raise RuntimeError(f"truncated environment {env_index} lacks a final observation")
        final_observation = final_observations[env_index]
        if final_observation is None:
            raise RuntimeError(f"truncated environment {env_index} lacks a final observation")
        bootstrap_observations[env_index] = final_observation

    return bootstrap_observations


class CleanRLPmpoAgent(nn.Module):
    """The actor and critic architecture from CleanRL PMPO v3."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        q_ranked_preferences: bool = False,
    ):
        super().__init__()
        self.critic = nn.Sequential(
            cleanrl_layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            cleanrl_layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            cleanrl_layer_init(nn.Linear(64, 1), std = 1.),
        )
        self.actor_mean = nn.Sequential(
            cleanrl_layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            cleanrl_layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            cleanrl_layer_init(nn.Linear(64, action_dim), std = .01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        self.q_critic = None
        if q_ranked_preferences:
            self.q_critic = nn.Sequential(
                cleanrl_layer_init(nn.Linear(obs_dim + action_dim, 64)),
                nn.Tanh(),
                cleanrl_layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                cleanrl_layer_init(nn.Linear(64, 1), std = 1.),
            )

    def get_value(self, observations: Tensor):
        return self.critic(observations)

    def get_q_value(self, observations: Tensor, actions: Tensor):
        if self.q_critic is None:
            raise RuntimeError("the Q critic requires q_ranked_preferences=True")
        return self.q_critic(torch.cat((observations, actions), dim = -1))

    def get_action_and_value(self, observations: Tensor, actions: Tensor | None = None):
        action_mean = self.actor_mean(observations)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        distribution = Normal(action_mean, action_logstd.exp())

        if actions is None:
            actions = distribution.sample()

        return (
            actions,
            distribution.log_prob(actions).sum(dim = 1),
            distribution.entropy().sum(dim = 1),
            self.critic(observations),
        )

    def get_action_value_and_dist(self, observations: Tensor, actions: Tensor):
        action_mean = self.actor_mean(observations)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        distribution = Normal(action_mean, action_logstd.exp())
        return (
            distribution.log_prob(actions),
            distribution.entropy().sum(dim = 1),
            self.critic(observations),
            distribution,
        )


def make_cleanrl_updated_ppo_env(env_name: str, gamma: float):
    """Build the standard CleanRL continuous-control wrapper sequence."""

    env = gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # The Beta policy enforces bounds. Keep the physical space visible to the actor.
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env,
        lambda observations: np.clip(observations, -10., 10.),
        observation_space = env.observation_space,
    )
    env = gym.wrappers.NormalizeReward(env, gamma = gamma)
    return gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10., 10.))


def make_cleanrl_pmpo_env(env_name: str, gamma: float):
    """Build the exact wrapper sequence used by CleanRL PMPO v3."""

    env = gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env,
        lambda observations: np.clip(observations, -10., 10.),
        observation_space = env.observation_space,
    )
    env = gym.wrappers.NormalizeReward(env, gamma = gamma)
    return gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10., 10.))


def cleanrl_completed_episodes(infos: Mapping[str, Any]):
    """Return raw completed episode returns and lengths from Gymnasium vector infos."""

    completed = []
    final_infos = infos.get("final_info")
    if isinstance(final_infos, Mapping):
        episode = final_infos.get("episode")
        if isinstance(episode, Mapping):
            returns = np.asarray(episode["r"]).reshape(-1)
            lengths = np.asarray(episode["l"]).reshape(-1)
            mask = final_infos.get("_episode", infos.get("_final_info"))
            if mask is None:
                mask = episode.get("_r")
            if mask is None:
                mask = np.ones_like(returns, dtype = bool)
            mask = np.asarray(mask, dtype = bool).reshape(-1)
            if not (returns.shape == lengths.shape == mask.shape):
                raise ValueError("Gymnasium final episode statistics have inconsistent shapes")
            completed.extend(
                (float(return_), int(length))
                for return_, length in zip(returns[mask], lengths[mask])
            )
    elif final_infos is not None:
        for final_info in np.asarray(final_infos, dtype = object).reshape(-1):
            if not isinstance(final_info, Mapping):
                continue
            episode = final_info.get("episode")
            if not isinstance(episode, Mapping):
                continue
            completed.append((float(episode["r"]), int(episode["l"])))

    if len(completed) > 0:
        return completed

    episode = infos.get("episode")
    if not isinstance(episode, Mapping) or "r" not in episode or "l" not in episode:
        return completed

    returns = np.asarray(episode["r"]).reshape(-1)
    lengths = np.asarray(episode["l"]).reshape(-1)
    mask = infos.get("_episode", episode.get("_r"))
    if mask is None:
        mask = np.ones_like(returns, dtype = bool)
    mask = np.asarray(mask, dtype = bool).reshape(-1)

    if not (returns.shape == lengths.shape == mask.shape):
        raise ValueError("Gymnasium vector episode statistics have inconsistent shapes")

    return [(float(return_), int(length)) for return_, length in zip(returns[mask], lengths[mask])]


def cleanrl_pmpo_losses(
    agent: CleanRLPmpoAgent,
    *,
    observations: Tensor,
    actions: Tensor,
    advantages: Tensor,
    returns: Tensor,
    old_values: Tensor,
    prior_actor_mean: nn.Module,
    prior_logstd: Tensor,
    pmpo_alpha: float,
    pmpo_kl_div_loss_weight: float,
    pmpo_reverse_kl: bool = True,
    clip_coef: float = .2,
    clip_values: bool = True,
    value_loss_weight: float = .5,
    joint_action_log_prob: bool = False,
    q_ranked_preferences: bool = False,
    q_value_loss_weight: float = .5,
    num_action_candidates: int = 4,
):
    """Calculate the CleanRL PMPO v3 minibatch loss."""

    log_probs, entropy, values, distribution = agent.get_action_value_and_dist(observations, actions)
    prior_distribution = None
    if q_ranked_preferences or pmpo_kl_div_loss_weight > 0.:
        with torch.no_grad():
            prior_mean = prior_actor_mean(observations)
            prior_distribution = Normal(prior_mean, prior_logstd.expand_as(prior_mean).exp())

    q_value_loss = values.new_zeros(())
    q_preference_gap = values.new_zeros(())
    if q_ranked_preferences:
        if num_action_candidates != 4:
            raise ValueError("q_ranked_preferences requires four action candidates")
        if prior_distribution is None:
            raise RuntimeError("q_ranked_preferences requires a prior distribution")

        with torch.no_grad():
            candidate_actions = prior_distribution.sample((num_action_candidates,)).movedim(0, 1)
            candidate_observations = observations.unsqueeze(1).expand(
                -1,
                num_action_candidates,
                -1,
            )
            candidate_q_values = agent.get_q_value(
                candidate_observations.reshape(-1, observations.shape[-1]),
                candidate_actions.reshape(-1, actions.shape[-1]),
            ).reshape(observations.shape[0], num_action_candidates)
            ranked_indices = candidate_q_values.argsort(dim = 1)
            negative_indices = ranked_indices[:, :2]
            positive_indices = ranked_indices[:, -2:]
            negative_action_indices = negative_indices.unsqueeze(-1).expand(
                -1,
                -1,
                actions.shape[-1],
            )
            positive_action_indices = positive_indices.unsqueeze(-1).expand(
                -1,
                -1,
                actions.shape[-1],
            )
            negative_actions = candidate_actions.gather(
                1,
                negative_action_indices,
            )
            positive_actions = candidate_actions.gather(
                1,
                positive_action_indices,
            )
            negative_q_values = candidate_q_values.gather(1, negative_indices)
            positive_q_values = candidate_q_values.gather(1, positive_indices)
            q_preference_gap = positive_q_values.mean() - negative_q_values.mean()

        candidate_distribution = Normal(
            distribution.loc.unsqueeze(1),
            distribution.scale.unsqueeze(1),
        )
        positive_log_probs = candidate_distribution.log_prob(positive_actions)
        negative_log_probs = candidate_distribution.log_prob(negative_actions)
        if joint_action_log_prob:
            positive_log_probs = positive_log_probs.sum(dim = -1)
            negative_log_probs = negative_log_probs.sum(dim = -1)
        positive_loss = positive_log_probs.mean()
        negative_loss = -negative_log_probs.mean()
        predicted_q_values = agent.get_q_value(observations, actions).view(-1)
        q_value_loss = .5 * (predicted_q_values - returns.detach()).square().mean()
    elif joint_action_log_prob:
        joint_log_probs = log_probs.sum(dim = -1)
        positive_mask = advantages >= 0.
        negative_mask = ~positive_mask
        positive_loss = joint_log_probs[positive_mask].sum().div(
            positive_mask.sum().clamp(min = 1)
        )
        negative_loss = -joint_log_probs[negative_mask].sum().div(
            negative_mask.sum().clamp(min = 1)
        )
    else:
        positive_mask = (advantages >= 0.).unsqueeze(-1).expand_as(log_probs)
        negative_mask = ~positive_mask
        positive_loss = log_probs[positive_mask].sum().div(
            positive_mask.sum().clamp(min = 1)
        )
        negative_loss = -log_probs[negative_mask].sum().div(
            negative_mask.sum().clamp(min = 1)
        )
    policy_loss = -(pmpo_alpha * positive_loss + (1. - pmpo_alpha) * negative_loss)

    kl_value = values.new_zeros(())
    if pmpo_kl_div_loss_weight > 0.:
        if prior_distribution is None:
            raise RuntimeError("PMPO KL requires a prior distribution")

        kl_inputs, kl_targets = prior_distribution, distribution
        if not pmpo_reverse_kl:
            kl_inputs, kl_targets = kl_targets, kl_inputs

        kl_value = kl_divergence(kl_inputs, kl_targets).sum(dim = -1).mean()
        policy_loss = policy_loss + pmpo_kl_div_loss_weight * kl_value

    values = values.view(-1)
    if clip_values:
        value_loss_unclipped = (values - returns).square()
        clipped_values = old_values + (values - old_values).clamp(-clip_coef, clip_coef)
        value_loss_clipped = (clipped_values - returns).square()
        value_loss = .5 * torch.maximum(value_loss_unclipped, value_loss_clipped).mean()
    else:
        value_loss = .5 * (values - returns).square().mean()

    entropy_loss = entropy.mean()
    loss = policy_loss + value_loss_weight * value_loss
    if q_ranked_preferences:
        loss = loss + q_value_loss_weight * q_value_loss
    return (
        loss,
        policy_loss,
        value_loss,
        entropy_loss,
        kl_value,
        q_value_loss,
        q_preference_gap,
    )


def cleanrl_pmpo_learning_rate(
    initial_learning_rate: float,
    loop: int,
    num_loops: int,
    *,
    constant: bool = False,
):
    """Return the flagged constant rate or the default linear decay."""

    if constant:
        return initial_learning_rate
    return (1. - loop / num_loops) * initial_learning_rate


def run_cleanrl_updated_ppo(*, config: Mapping[str, Any]):
    """Run updated CleanRL PPO behind an explicit small-model flag."""

    if config["objective"] != "ppo":
        raise ValueError("small_model.cleanrl_updated_ppo requires objective='ppo'")
    if config["checkpoint_path"]:
        raise ValueError("small_model.cleanrl_updated_ppo does not support checkpoint resume")
    if config["checkpoint_every"] > 0:
        raise ValueError("small_model.cleanrl_updated_ppo requires checkpoint_every=0")
    if config["clear_log_dir"]:
        raise ValueError("small_model.cleanrl_updated_ppo does not support clear_log_dir")
    if config["rollouts_per_loop"] != 1:
        raise ValueError("small_model.cleanrl_updated_ppo requires rollouts_per_loop=1")

    env_name = config["env_name"]
    seed = config["seed"]
    num_loops = config["num_loops"]
    num_envs = config["total_envs"]
    num_steps = config["max_timesteps"]
    device = torch.device("cpu" if config["cpu"] or not torch.cuda.is_available() else "cuda")

    if config["require_cuda"] and device.type != "cuda":
        raise RuntimeError("CUDA was required for this run, but no CUDA device is available")
    if min(num_loops, num_envs, num_steps) <= 0:
        raise ValueError("loop, environment, and rollout step counts must be positive")

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // CLEANRL_UPDATED_PPO_NUM_MINIBATCHES
    if minibatch_size == 0:
        raise ValueError(
            "small_model.cleanrl_updated_ppo requires num_envs * max_timesteps to be at least 32"
        )
    if minibatch_size < 2 or batch_size % minibatch_size == 1:
        raise ValueError(
            "small_model.cleanrl_updated_ppo requires at least two samples in each minibatch"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    run_log_dir = train_gym.resolve_log_dir(
        config["log_dir"],
        run_name = config["run_name"],
        checkpoint_path = None,
        unique_log_dir = config["unique_log_dir"],
        checkpoint_folder = None,
    )
    if run_log_dir.exists():
        raise FileExistsError(
            f"run directory already exists for run_name={config['run_name']!r}: {run_log_dir}"
        )

    envs = gym.vector.SyncVectorEnv(
        [
            lambda: make_cleanrl_updated_ppo_env(
                env_name,
                gamma = CLEANRL_UPDATED_PPO_GAMMA,
            )
            for _ in range(num_envs)
        ],
        autoreset_mode = gym.vector.AutoresetMode.SAME_STEP,
    )
    writer = SummaryWriter(str(run_log_dir)) if config["use_tensorboard"] else None

    try:
        if exists(writer):
            writer.add_text("run/details", config["run_details"], 0)
            writer.add_text(
                "run/hyperparameters",
                train_gym.serialize_hyperparameters(dict(config)),
                0,
            )
            writer.add_text(
                "run/policy",
                "Beta: alpha,beta=1+softplus(head); FP32; native-action storage",
                0,
            )

        observation_space = envs.single_observation_space
        action_space = envs.single_action_space
        agent = CleanRLUpdatedPPOAgent(observation_space, action_space).to(device)
        optimizer = Adam(
            agent.parameters(),
            lr = config["agent_learning_rate"],
            eps = 1e-5,
            fused = device.type == "cuda",
        )

        observation_shape = observation_space.shape
        native_action_shape = (agent.action_dim,)
        observations = torch.zeros(
            (num_steps, num_envs, *observation_shape),
            device = device,
        )
        native_actions = torch.zeros(
            (num_steps, num_envs, *native_action_shape),
            device = device,
        )
        old_log_probs = torch.zeros((num_steps, num_envs), device = device)
        rewards = torch.zeros((num_steps, num_envs), device = device)
        terminations = torch.zeros((num_steps, num_envs), device = device)
        truncations = torch.zeros((num_steps, num_envs), device = device)
        truncation_bootstrap_values = torch.zeros(
            (num_steps, num_envs),
            device = device,
        )
        values = torch.zeros((num_steps, num_envs), device = device)
        next_observations, _ = envs.reset(seed = seed)
        next_observations = torch.as_tensor(
            next_observations,
            device = device,
            dtype = torch.float32,
        )
        shuffle_generator = torch.Generator(device = device).manual_seed(seed)
        env_step = 0
        start_time = perf_counter()

        pbar = tqdm(range(num_loops), desc = "CleanRL updated PPO", dynamic_ncols = True)
        for loop in pbar:
            learning_rate = cleanrl_pmpo_learning_rate(
                config["agent_learning_rate"],
                loop,
                num_loops,
            )
            optimizer.param_groups[0]["lr"] = learning_rate
            rollout_start = perf_counter()
            rollout_environment_seconds = 0.
            rollout_returns = []
            rollout_lengths = []

            for step in range(num_steps):
                env_step += num_envs
                observations[step] = next_observations

                with torch.no_grad():
                    native_action, action, log_prob, _, value = agent.sample_action_and_value(
                        next_observations
                    )
                    native_actions[step] = native_action
                    old_log_probs[step] = log_prob
                    values[step] = value.flatten()

                if not torch.isfinite(action).all():
                    raise FloatingPointError("policy produced nonfinite actions")

                environment_step_start = perf_counter()
                (
                    raw_next_observations,
                    reward,
                    terminated,
                    truncated,
                    infos,
                ) = envs.step(action.cpu().numpy())
                rollout_environment_seconds += perf_counter() - environment_step_start

                terminated_tensor = torch.as_tensor(
                    terminated,
                    device = device,
                    dtype = torch.float32,
                )
                truncated_tensor = torch.as_tensor(
                    truncated,
                    device = device,
                    dtype = torch.float32,
                )
                terminations[step] = terminated_tensor
                truncations[step] = truncated_tensor
                rewards[step] = torch.as_tensor(
                    reward,
                    device = device,
                    dtype = torch.float32,
                ).view(-1)

                if np.any(truncated):
                    bootstrap_observations = cleanrl_updated_ppo_bootstrap_observations(
                        raw_next_observations,
                        truncated,
                        infos,
                    )
                    bootstrap_observations = torch.as_tensor(
                        bootstrap_observations,
                        device = device,
                        dtype = torch.float32,
                    )
                    with torch.no_grad():
                        bootstrap_values = agent.get_value(bootstrap_observations).flatten()
                    truncation_bootstrap_values[step] = torch.where(
                        truncated_tensor.bool(),
                        bootstrap_values,
                        torch.zeros_like(bootstrap_values),
                    )
                else:
                    truncation_bootstrap_values[step].zero_()

                next_observations = torch.as_tensor(
                    raw_next_observations,
                    device = device,
                    dtype = torch.float32,
                )
                for episode_return, episode_length in cleanrl_completed_episodes(infos):
                    rollout_returns.append(episode_return)
                    rollout_lengths.append(episode_length)
                    train_gym.log_scalars(writer, {
                        "charts/episodic_return": episode_return,
                        "charts/episodic_length": episode_length,
                    }, env_step)

            rollout_total_seconds = perf_counter() - rollout_start
            with torch.no_grad():
                rollout_tail_value = agent.get_value(next_observations).flatten()
                advantages, returns = cleanrl_updated_ppo_gae(
                    rewards,
                    values,
                    terminations,
                    truncations,
                    truncation_bootstrap_values,
                    rollout_tail_value,
                )

            batch_observations = observations.flatten(0, 1)
            batch_native_actions = native_actions.flatten(0, 1)
            batch_old_log_probs = old_log_probs.flatten()
            batch_advantages = advantages.flatten()
            batch_returns = returns.flatten()
            batch_values = values.flatten()
            last_update_metrics = None
            clip_fractions = []
            grad_norm = float("nan")

            for _ in range(CLEANRL_UPDATED_PPO_UPDATE_EPOCHS):
                indices = torch.randperm(
                    batch_size,
                    device = device,
                    generator = shuffle_generator,
                )
                for start in range(0, batch_size, minibatch_size):
                    minibatch_indices = indices[start:start + minibatch_size]
                    loss, update_metrics = cleanrl_updated_ppo_loss(
                        agent,
                        observations = batch_observations[minibatch_indices],
                        native_actions = batch_native_actions[minibatch_indices],
                        old_log_probs = batch_old_log_probs[minibatch_indices],
                        advantages = batch_advantages[minibatch_indices],
                        returns = batch_returns[minibatch_indices],
                        old_values = batch_values[minibatch_indices],
                    )
                    optimizer.zero_grad(set_to_none = True)
                    loss.backward()
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        agent.parameters(),
                        CLEANRL_UPDATED_PPO_MAX_GRAD_NORM,
                    ))
                    optimizer.step()
                    last_update_metrics = update_metrics
                    clip_fractions.append(update_metrics[5])

            if last_update_metrics is None:
                raise RuntimeError("CleanRL updated PPO completed no optimizer updates")

            target_variance = torch.var(batch_returns, correction = 0)
            explained_variance = (
                float("nan")
                if target_variance == 0.
                else 1. - torch.var(
                    batch_returns - batch_values,
                    correction = 0,
                ).div(target_variance).item()
            )
            metrics = {
                "charts/learning_rate": learning_rate,
                "charts/SPS": env_step / max(perf_counter() - start_time, 1e-6),
                "rollout/average_return": float(np.mean(rollout_returns)) if rollout_returns else None,
                "rollout/average_length": float(np.mean(rollout_lengths)) if rollout_lengths else None,
                "timings/rollout_environment_seconds": rollout_environment_seconds,
                "timings/rollout_total_seconds": rollout_total_seconds,
                "cleanrl_updated_ppo/loss": (
                    last_update_metrics[0]
                    + CLEANRL_UPDATED_PPO_VALUE_LOSS_WEIGHT * last_update_metrics[1]
                ).item(),
                "cleanrl_updated_ppo/policy_loss": last_update_metrics[0].item(),
                "cleanrl_updated_ppo/value_loss": last_update_metrics[1].item(),
                "cleanrl_updated_ppo/entropy": last_update_metrics[2].item(),
                "cleanrl_updated_ppo/old_approx_kl": last_update_metrics[3].item(),
                "cleanrl_updated_ppo/approx_kl": last_update_metrics[4].item(),
                "cleanrl_updated_ppo/clip_fraction": torch.stack(clip_fractions).mean().item(),
                "cleanrl_updated_ppo/explained_variance": explained_variance,
                "cleanrl_updated_ppo/grad_norm": grad_norm,
            }
            finite_metrics = (
                value
                for name, value in metrics.items()
                if value is not None and name != "cleanrl_updated_ppo/explained_variance"
            )
            if any(not np.isfinite(value) for value in finite_metrics):
                raise FloatingPointError("nonfinite CleanRL updated PPO learner metrics")

            train_gym.log_scalars(writer, metrics, env_step)
            pbar.set_postfix(
                return_ = f"{np.mean(rollout_returns):.1f}" if rollout_returns else "n/a",
                loss = f"{metrics['cleanrl_updated_ppo/loss']:.3f}",
            )
    finally:
        envs.close()
        if exists(writer):
            writer.flush()
            writer.close()


def run_cleanrl_pmpo(*, config: Mapping[str, Any]):
    """Run the CleanRL PMPO v3 comparison behind an explicit small-model flag."""

    if config["objective"] != "pmpo":
        raise ValueError("small_model.cleanrl_pmpo requires objective='pmpo'")
    if config["checkpoint_path"]:
        raise ValueError("small_model.cleanrl_pmpo does not support checkpoint resume")
    if config["checkpoint_every"] > 0:
        raise ValueError("small_model.cleanrl_pmpo requires checkpoint_every=0")
    if config["clear_log_dir"]:
        raise ValueError("small_model.cleanrl_pmpo does not support clear_log_dir")

    small_config = resolve_small_model_config(config.get("small_model"))
    env_name = config["env_name"]
    seed = config["seed"]
    num_loops = config["num_loops"]
    num_envs = config["total_envs"]
    num_steps = config["max_timesteps"]
    device = torch.device("cpu" if config["cpu"] or not torch.cuda.is_available() else "cuda")

    if config["require_cuda"] and device.type != "cuda":
        raise RuntimeError("CUDA was required for this run, but no CUDA device is available")

    batch_size = num_envs * num_steps
    num_minibatches = 32
    if batch_size < num_minibatches or batch_size % num_minibatches != 0:
        raise ValueError(
            "small_model.cleanrl_pmpo requires num_envs * max_timesteps to be divisible by 32"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    envs = gym.vector.SyncVectorEnv([
        lambda: make_cleanrl_pmpo_env(env_name, gamma = .99)
        for _ in range(num_envs)
    ])
    run_log_dir = train_gym.resolve_log_dir(
        config["log_dir"],
        run_name = config["run_name"],
        checkpoint_path = None,
        unique_log_dir = config["unique_log_dir"],
        checkpoint_folder = None,
    )
    if run_log_dir.exists():
        envs.close()
        raise FileExistsError(f"run directory already exists for run_name={config['run_name']!r}: {run_log_dir}")

    writer = SummaryWriter(str(run_log_dir)) if config["use_tensorboard"] else None
    if exists(writer):
        writer.add_text("run/details", config["run_details"], 0)
        writer.add_text("run/hyperparameters", train_gym.serialize_hyperparameters(dict(config)), 0)

    observation_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    if (
        not isinstance(envs.single_action_space, gym.spaces.Box)
        or len(observation_shape) != 1
        or len(action_shape) != 1
    ):
        envs.close()
        if exists(writer):
            writer.close()
        raise ValueError("small_model.cleanrl_pmpo requires flat Box observations and actions")

    agent = CleanRLPmpoAgent(
        observation_shape[0],
        action_shape[0],
        q_ranked_preferences = small_config["cleanrl_pmpo_q_ranked_preferences"],
    ).to(device)
    optimizer = Adam(agent.parameters(), lr = config["agent_learning_rate"], eps = 1e-5)
    minibatch_size = batch_size // num_minibatches

    observations = torch.zeros((num_steps, num_envs, *observation_shape), device = device)
    actions = torch.zeros((num_steps, num_envs, *action_shape), device = device)
    rewards = torch.zeros((num_steps, num_envs), device = device)
    dones = torch.zeros((num_steps, num_envs), device = device)
    values = torch.zeros((num_steps, num_envs), device = device)
    next_observations, _ = envs.reset(seed = seed)
    next_observations = torch.as_tensor(next_observations, device = device, dtype = torch.float32)
    next_done = torch.zeros(num_envs, device = device)
    env_step = 0
    start_time = perf_counter()

    try:
        pbar = tqdm(range(num_loops), desc = "CleanRL PMPO", dynamic_ncols = True)
        for loop in pbar:
            learning_rate = cleanrl_pmpo_learning_rate(
                config["agent_learning_rate"],
                loop,
                num_loops,
                constant = small_config["cleanrl_pmpo_constant_learning_rate"],
            )
            optimizer.param_groups[0]["lr"] = learning_rate
            rollout_start = perf_counter()
            rollout_environment_seconds = 0.
            rollout_returns = []
            rollout_lengths = []

            for step in range(num_steps):
                env_step += num_envs
                observations[step] = next_observations
                dones[step] = next_done

                with torch.no_grad():
                    action, _, _, value = agent.get_action_and_value(next_observations)
                    values[step] = value.flatten()

                actions[step] = action
                environment_step_start = perf_counter()
                next_observations, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                rollout_environment_seconds += perf_counter() - environment_step_start
                next_done = torch.as_tensor(np.logical_or(terminated, truncated), device = device, dtype = torch.float32)
                rewards[step] = torch.as_tensor(reward, device = device, dtype = torch.float32).view(-1)
                next_observations = torch.as_tensor(next_observations, device = device, dtype = torch.float32)

                for episode_return, episode_length in cleanrl_completed_episodes(infos):
                    rollout_returns.append(episode_return)
                    rollout_lengths.append(episode_length)
                    train_gym.log_scalars(writer, {
                        "charts/episodic_return": episode_return,
                        "charts/episodic_length": episode_length,
                    }, env_step)

            rollout_total_seconds = perf_counter() - rollout_start

            with torch.no_grad():
                next_value = agent.get_value(next_observations).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                last_gae_lambda = torch.zeros(num_envs, device = device)

                for step in reversed(range(num_steps)):
                    if step == num_steps - 1:
                        next_nonterminal = 1. - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1. - dones[step + 1]
                        next_values = values[step + 1]

                    delta = rewards[step] + .99 * next_values * next_nonterminal - values[step]
                    last_gae_lambda = delta + .99 * .95 * next_nonterminal * last_gae_lambda
                    advantages[step] = last_gae_lambda

                returns = advantages + values

            batch_observations = observations.reshape(batch_size, *observation_shape)
            batch_actions = actions.reshape(batch_size, *action_shape)
            batch_advantages = advantages.reshape(batch_size)
            batch_returns = returns.reshape(batch_size)
            batch_values = values.reshape(batch_size)
            prior_actor_mean = deepcopy(agent.actor_mean).eval()
            for parameter in prior_actor_mean.parameters():
                parameter.requires_grad = False
            prior_logstd = agent.actor_logstd.detach().clone()

            minibatch_indices = np.arange(batch_size)
            np.random.shuffle(minibatch_indices)
            last_metrics = {}
            for start in range(0, batch_size, minibatch_size):
                indices = torch.as_tensor(minibatch_indices[start:start + minibatch_size], device = device)
                (
                    loss,
                    policy_loss,
                    value_loss,
                    entropy_loss,
                    kl_value,
                    q_value_loss,
                    q_preference_gap,
                ) = cleanrl_pmpo_losses(
                    agent,
                    observations = batch_observations[indices],
                    actions = batch_actions[indices],
                    advantages = batch_advantages[indices],
                    returns = batch_returns[indices],
                    old_values = batch_values[indices],
                    prior_actor_mean = prior_actor_mean,
                    prior_logstd = prior_logstd,
                    pmpo_alpha = config["pmpo_pos_to_neg_weight"],
                    pmpo_kl_div_loss_weight = config["pmpo_kl_div_loss_weight"],
                    joint_action_log_prob = small_config["cleanrl_pmpo_joint_log_prob"],
                    q_ranked_preferences = small_config["cleanrl_pmpo_q_ranked_preferences"],
                )
                optimizer.zero_grad()
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"]))
                optimizer.step()
                last_metrics = {
                    "cleanrl_pmpo/loss": loss.item(),
                    "cleanrl_pmpo/policy_loss": policy_loss.item(),
                    "cleanrl_pmpo/value_loss": value_loss.item(),
                    "cleanrl_pmpo/entropy": entropy_loss.item(),
                    "cleanrl_pmpo/kl_div": kl_value.item(),
                    "cleanrl_pmpo/grad_norm": grad_norm,
                }
                if small_config["cleanrl_pmpo_q_ranked_preferences"]:
                    last_metrics.update({
                        "cleanrl_pmpo/q_value_loss": q_value_loss.item(),
                        "cleanrl_pmpo/q_preference_gap": q_preference_gap.item(),
                    })

            target_variance = torch.var(batch_returns, correction = 0)
            explained_variance = float("nan") if target_variance == 0. else 1. - torch.var(
                batch_returns - batch_values,
                correction = 0,
            ).div(target_variance).item()
            train_gym.log_scalars(writer, {
                "charts/learning_rate": learning_rate,
                "charts/SPS": env_step / max(perf_counter() - start_time, 1e-6),
                "rollout/average_return": float(np.mean(rollout_returns)) if len(rollout_returns) > 0 else None,
                "rollout/average_length": float(np.mean(rollout_lengths)) if len(rollout_lengths) > 0 else None,
                "timings/rollout_environment_seconds": rollout_environment_seconds,
                "timings/rollout_total_seconds": rollout_total_seconds,
                "cleanrl_pmpo/explained_variance": explained_variance,
                **last_metrics,
            }, env_step)
            pbar.set_postfix(
                return_ = f"{np.mean(rollout_returns):.1f}" if len(rollout_returns) > 0 else "n/a",
                loss = f"{last_metrics['cleanrl_pmpo/loss']:.3f}",
            )
    finally:
        envs.close()
        if exists(writer):
            writer.flush()
            writer.close()


def run(**config):
    """Run the small benchmark from a full Gym configuration mapping."""

    train_gym.configure_matmul_precision(
        config.get("allow_tf32", True),
        torch.device("cpu" if config["cpu"] or not torch.cuda.is_available() else "cuda"),
    )
    small_config = resolve_small_model_config(config.get("small_model"))
    if small_config["cleanrl_updated_ppo"] and small_config["cleanrl_pmpo"]:
        raise ValueError(
            "small_model.cleanrl_updated_ppo and small_model.cleanrl_pmpo are mutually exclusive"
        )
    if small_config["cleanrl_updated_ppo"]:
        return run_cleanrl_updated_ppo(config = config)
    if small_config["cleanrl_pmpo"]:
        return run_cleanrl_pmpo(config = config)

    objective = config.get("objective", "pmpo")
    if objective == "spo":
        raise ValueError("small_model_only supports objective='ppo', objective='pmpo', and objective='reinforce'; objective='spo' is not supported")
    if objective not in ("ppo", "pmpo", "reinforce"):
        raise ValueError(f"small_model_only does not support objective={objective!r}")

    if objective == "reinforce" and (
        small_config["update_epochs"] != 1 or
        small_config["minibatch_num_envs"] is not None
    ):
        raise ValueError(
            "objective='reinforce' requires small_model.update_epochs=1 and "
            "small_model.minibatch_num_envs=null to keep each update on-policy"
        )

    env_name = config["env_name"]
    num_loops = config["num_loops"]
    rollouts_per_loop = config["rollouts_per_loop"]
    num_envs = config["total_envs"]
    max_timesteps = config["max_timesteps"]
    seed = config["seed"]
    cpu = config["cpu"]
    require_cuda = config["require_cuda"]
    env_vectorization_mode = config["env_vectorization_mode"]
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required for this run, but no CUDA device is available")

    if env_vectorization_mode not in ("async", "sync"):
        raise ValueError(f"env_vectorization_mode must be 'async' or 'sync', got {env_vectorization_mode!r}")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    obs_dim, action_dim, action_range = train_gym.inspect_env_spaces(env_name, config.get("obs_dim"))
    agent = StateActorCritic(
        obs_dim = obs_dim,
        action_dim = action_dim,
        action_range = action_range,
        continuous_dist_type = config["continuous_dist_type"],
        dim = small_config["dim"],
        encoder_depth = small_config["encoder_depth"],
        observation_normalization = small_config["observation_normalization"],
        reward_encoder_type = config["reward_encoder_type"],
        use_symexp_value_head = config["use_symexp_value_head"],
        use_tdmpc2_value_head = config["use_tdmpc2_value_head"],
        use_symlog_hl_gauss_value_head = config["use_symlog_hl_gauss_value_head"],
        use_hl_gauss_actor_critic = config["use_hl_gauss_actor_critic"],
        use_baseline_symexp_value_head_init = config["use_baseline_symexp_value_head_init"],
        value_encoder_num_bins = config["value_encoder_num_bins"],
        value_range = tuple(config["value_range"]),
        value_encoder_sigma_to_bin_ratio = config["value_encoder_sigma_to_bin_ratio"],
        value_min_max_value_on_bin_center = config["value_min_max_value_on_bin_center"],
        value_use_symlog = config["value_use_symlog"],
        value_decode_symlog_expected_raw = config["value_decode_symlog_expected_raw"],
        hl_gauss_actor_critic_range = tuple(config["hl_gauss_actor_critic_range"]),
        hl_gauss_actor_critic_num_bins = config["hl_gauss_actor_critic_num_bins"],
        hl_gauss_actor_critic_sigma_to_bin_ratio = config["hl_gauss_actor_critic_sigma_to_bin_ratio"],
        hl_gauss_actor_critic_min_max_value_on_bin_center = config["hl_gauss_actor_critic_min_max_value_on_bin_center"],
        normalize_advantages = config["normalize_advantages"],
        pmpo_pos_to_neg_weight = config["pmpo_pos_to_neg_weight"],
        pmpo_kl_div_loss_weight = config["pmpo_kl_div_loss_weight"],
        use_delight_gating = config["use_delight_gating"],
        old_continuous_action_encoder = config["old_continuous_action_encoder"],
        old_pmpo = config["old_pmpo"],
        old_policy_head = config["old_policy_head"],
        old_value_head = config["old_value_head"],
        value_head_init_value = config["value_head_init_value"],
        reinforce_discount_factor = config["reinforce_discount_factor"],
        reinforce_gae_lambda = config["reinforce_gae_lambda"],
        reinforce_return_normalization = config["reinforce_return_normalization"],
        reinforce_use_discounted_trajectory_weights = config["reinforce_use_discounted_trajectory_weights"],
        policy_entropy_weight = config["policy_entropy_weight"],
    ).to(device)
    optimizer = train_gym.make_optimizer(
        agent.optimizer_parameters(),
        lr = config["agent_learning_rate"],
        weight_decay = config["optimizer_weight_decay"],
        use_muon = config["use_muon_optimizer"],
        muon_params = (),
    )

    checkpoint_path = config.get("checkpoint_path")
    run_log_dir = train_gym.resolve_log_dir(
        config["log_dir"],
        run_name = config["run_name"],
        checkpoint_path = checkpoint_path,
        unique_log_dir = config["unique_log_dir"],
        checkpoint_folder = config.get("checkpoint_folder"),
    )
    checkpoint_root = run_log_dir / "checkpoints" if config.get("checkpoint_folder") is None else Path(config["checkpoint_folder"])
    if not checkpoint_path and run_log_dir.exists():
        raise FileExistsError(f"run directory already exists for run_name={config['run_name']!r}: {run_log_dir}")
    if config.get("clear_log_dir"):
        raise ValueError("clear_log_dir is not supported by the state-only trainer")

    writer = SummaryWriter(str(run_log_dir)) if config["use_tensorboard"] else None
    if exists(writer):
        writer.add_text("run/details", config["run_details"], 0)
        writer.add_text("run/hyperparameters", train_gym.serialize_hyperparameters(config), 0)

    start_loop = update_step = env_step = 0
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path, agent = agent, optimizer = optimizer)
        start_loop = int(checkpoint["loop"])
        update_step = int(checkpoint["update_step"])
        env_step = int(checkpoint["env_step"])
    elif small_config["observation_normalization"] == "fixed_random":
        observations = train_gym.collect_random_observations(
            env_name,
            num_steps = max(1, small_config["normalization_observations"] // num_envs),
            num_envs = num_envs,
            seed = seed,
            env_vectorization_mode = env_vectorization_mode,
        )[:small_config["normalization_observations"]]
        agent.observation_normalizer.set_fixed_statistics(observations.to(device))

    env = train_gym.make_env(
        env_name,
        seed,
        vectorized = True,
        num_envs = num_envs,
        episode_stats_buffer_length = max(100, num_envs * max_timesteps),
        env_vectorization_mode = env_vectorization_mode,
    )
    try:
        pbar = tqdm(range(start_loop, num_loops), desc = "state actor-critic", dynamic_ncols = True)
        for loop in pbar:
            agent.eval()
            rollouts = []
            rollout_returns = []
            rollout_lengths = []
            rollout_start = perf_counter()
            for rollout_index in range(rollouts_per_loop):
                exp = interact_with_env(
                    agent,
                    env,
                    seed = seed + loop * rollouts_per_loop + rollout_index,
                    max_timesteps = max_timesteps,
                )
                rollouts.append(exp)
                returns, lengths = train_gym.get_rollout_episode_stats(exp)
                rollout_returns.extend(returns)
                rollout_lengths.extend(lengths)
                env_step += int(sum(lengths))
                for episode_return, episode_length in zip(returns, lengths):
                    train_gym.log_scalars(writer, {"charts/episodic_return": episode_return, "charts/episodic_length": episode_length}, env_step)

            reset_seconds, step_seconds = train_gym.pop_rollout_env_timings(env)
            train_gym.log_scalars(writer, {
                "rollout/average_return": float(np.mean(rollout_returns)),
                "rollout/average_length": float(np.mean(rollout_lengths)),
                "timings/rollout_environment_seconds": reset_seconds + step_seconds,
                "timings/rollout_total_seconds": perf_counter() - rollout_start,
            }, env_step)

            # One default update consumes every rollout collected in this loop.
            experience = combine_experiences(rollouts)
            agent.train()
            update_step, metrics = train_on_experience(
                agent,
                optimizer,
                experience,
                objective = objective,
                update_epochs = small_config["update_epochs"],
                minibatch_num_envs = small_config["minibatch_num_envs"],
                max_grad_norm = config["max_grad_norm"],
                update_step = update_step,
                writer = writer,
                env_step = env_step,
            )
            train_gym.log_scalars(writer, {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
            }, env_step)
            pbar.set_postfix(return_ = f"{np.mean(rollout_returns):.1f}", loss = f"{metrics['state_actor_critic/loss']:.3f}")

            if config["checkpoint_every"] > 0 and (loop + 1) % config["checkpoint_every"] == 0:
                save_checkpoint(
                    checkpoint_root / f"loop_{loop + 1}.pt",
                    agent = agent,
                    optimizer = optimizer,
                    loop = loop + 1,
                    update_step = update_step,
                    env_step = env_step,
                )
    finally:
        env.close()
        if exists(writer):
            writer.flush()
            writer.close()


def main(config_path: str = "train/config.yaml", **overrides):
    """Load a normal Gym config and run only its state actor-critic section."""

    with Path(config_path).open("r", encoding = "utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a top-level mapping")
    config.update(overrides)
    return run(**config)


if __name__ == "__main__":
    fire.Fire(main)
