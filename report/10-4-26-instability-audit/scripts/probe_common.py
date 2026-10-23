"""Shared helpers of the checkpoint probes: build a run's tokenizer and world model on the CPU, load a checkpoint,
read real episodes from the replay, and evaluate the acting policy on them.

The model construction is the one of train/train_gym.py main(). The settings come from the run_settings.json of
the queue record that started the run. A strict state-dict load checks that the construction is the same.
"""

import glob
import inspect
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from dreamer4.dreamer4 import DynamicsWorldModel  # noqa: E402
from train import train_gym  # noqa: E402
from train.train_gym import ObservationTokenizer, inspect_env_spaces  # noqa: E402

RUNS = REPO / "runs" / "gym_imagination"
EPISODES_PER_LOOP = 64
STEPS = 1000
_settings = None


def settings(run):
    global _settings
    if _settings is None:
        _settings = {}
        for path in sorted(glob.glob(str(REPO / "runs" / "queues" / "*" / "run_settings.json"))):
            try:
                records = json.load(open(path))
            except (OSError, ValueError):
                continue
            for record in records.values():
                if isinstance(record, dict) and "train_kwargs" in record:
                    _settings.setdefault(record["train_kwargs"]["run_name"], record["train_kwargs"])
    defaults = {name: p.default for name, p in inspect.signature(train_gym.main).parameters.items() if p.default is not inspect.Parameter.empty}
    return {**defaults, **_settings[run]}


def run_dir(run):
    return Path(os.path.realpath(RUNS / run))


def build(run, checkpoint):
    k = settings(run)
    obs_dim, action_dim, action_range = inspect_env_spaces(k["env_name"], None)
    tokenizer = ObservationTokenizer(obs_dim = obs_dim, num_latent_tokens = k["num_latent_tokens"], dim_latent = k["dim_latent"],
                                     old_state_tokenizer = k["old_state_tokenizer"])

    value_encoder_kwargs = dict(reward_range = k["value_range"], num_bins = k["value_encoder_num_bins"])
    if not (k["use_symexp_value_head"] or k["use_tdmpc2_value_head"]):
        value_encoder_kwargs.update(sigma_to_bin_ratio = k["value_encoder_sigma_to_bin_ratio"], min_max_value_on_bin_center = k["value_min_max_value_on_bin_center"],
                                    use_symlog = k["value_use_symlog"], decode_symlog_expected_raw = k["value_decode_symlog_expected_raw"])
    if k["use_symlog_hl_gauss_value_head"]:
        value_encoder_kwargs.update(use_symlog = True, decode_symlog_expected_raw = True)
    hl_gauss_actor_critic_kwargs = None
    if k["use_hl_gauss_actor_critic"]:
        hl_gauss_actor_critic_kwargs = dict(reward_range = k["hl_gauss_actor_critic_range"], num_bins = k["hl_gauss_actor_critic_num_bins"],
                                            sigma_to_bin_ratio = k["hl_gauss_actor_critic_sigma_to_bin_ratio"],
                                            min_max_value_on_bin_center = k["hl_gauss_actor_critic_min_max_value_on_bin_center"],
                                            use_symlog = False, decode_symlog_expected_raw = False)
    world_model = DynamicsWorldModel(
        dim = k["model_dim"], dim_latent = k["dim_latent"], max_steps = 64, num_latent_tokens = k["num_latent_tokens"],
        num_spatial_tokens = k["num_latent_tokens"], num_register_tokens = 1, dim_critic_state = obs_dim, depth = k["depth"],
        time_block_every = k["time_block_every"],
        transformer_kwargs = dict(spatial_flex_attention_backend = k["spatial_flex_attention_backend"], use_attn_pool = k["use_attn_pool"],
                                  attn_pool_einsum_attend = k["attn_pool_einsum_attend"], attn_res = k["attn_res"], attn_res_block_size = k["attn_res_block_size"]),
        final_special_cross_attn = k["final_special_cross_attn"], num_discrete_actions = 0, num_continuous_actions = action_dim,
        continuous_dist_type = k["continuous_dist_type"], continuous_target_action_range = action_range, reward_encoder_type = k["reward_encoder_type"],
        use_symexp_value_head = k["use_symexp_value_head"], use_tdmpc2_value_head = k["use_tdmpc2_value_head"],
        use_symlog_hl_gauss_value_head = k["use_symlog_hl_gauss_value_head"], use_hl_gauss_actor_critic = k["use_hl_gauss_actor_critic"],
        use_baseline_symexp_value_head_init = k["use_baseline_symexp_value_head_init"],
        reward_encoder_kwargs = dict(reward_range = k["reward_range"], num_bins = k["reward_encoder_num_bins"], sigma_to_bin_ratio = k["reward_encoder_sigma_to_bin_ratio"],
                                     min_max_value_on_bin_center = k["reward_min_max_value_on_bin_center"], use_symlog = k["reward_use_symlog"],
                                     decode_symlog_expected_raw = k["reward_decode_symlog_expected_raw"]),
        value_encoder_kwargs = value_encoder_kwargs, hl_gauss_actor_critic_kwargs = hl_gauss_actor_critic_kwargs, predict_terminals = True,
        continuous_action_loss_weight = k["world_model_behavior_cloning_loss_weight"] if k["behavior_cloning_in_world_model_updates"] else 0.,
        discrete_action_loss_weight = 0., agent_predicts_state = k["agent_predicts_state"], agent_state_pred_loss_weight = k["agent_state_pred_loss_weight"],
        prob_shortcut_train = k["prob_shortcut_train"], policy_all_tokens = k["policy_all_tokens"], gae_discount_factor = 0.99,
        reinforce_discount_factor = k["reinforce_discount_factor"], reinforce_gae_lambda = k["reinforce_gae_lambda"],
        reinforce_return_normalization = k["reinforce_return_normalization"],
        reinforce_use_discounted_trajectory_weights = k["reinforce_use_discounted_trajectory_weights"],
        pmpo_pos_to_neg_weight = k["pmpo_pos_to_neg_weight"], pmpo_kl_div_loss_weight = k["pmpo_kl_div_loss_weight"], ppo_eps_clip = 0.2,
        value_clip = 0.2 if k["cleanrl_updated_ppo"] else 0.4, clip_values = k["cleanrl_updated_ppo"], cleanrl_ppo_closer = k["cleanrl_ppo_closer"],
        cleanrl_ppo_closer_hl_gauss_value_loss = k["cleanrl_ppo_closer_hl_gauss_value_loss"], normalize_advantages = k["normalize_advantages"],
        add_action_embed_to_agent_token = k["add_action_embed_to_agent_token"], old_continuous_action_encoder = k["old_continuous_action_encoder"],
        old_reward_shifting = k["old_reward_shifting"], old_pmpo = k["old_pmpo"], old_reward_head = k["old_reward_head"], old_terminal_head = k["old_terminal_head"],
        old_dynamics_ff = k["old_dynamics_ff"], old_policy_head = k["old_policy_head"], old_value_head = k["old_value_head"],
        old_embedding_init = k["old_embedding_init"], value_head_init_value = k["value_head_init_value"],
        policy_entropy_weight = k["cleanrl_ppo_entropy_weight"] if k["cleanrl_updated_ppo"] else k["policy_entropy_weight"],
        scale_reward_gradient_into_transformer = k["scale_reward_gradient_into_transformer"],
        reward_gradient_into_transformer_scale = k["reward_gradient_into_transformer_scale"],
        use_loss_normalization = k["use_dynamics_loss_normalization"], use_paper_aligned_shortcut_objective = k["use_paper_aligned_shortcut_objective"],
        **(dict(group_dynamics_loss_normalization = True) if k["group_dynamics_loss_normalization"] else dict()),
        **(dict(reward_loss_weight = k["world_model_reward_loss_weight"]) if k["world_model_reward_loss_weight"] != 1. else dict()),
        **(dict(ff_kwargs = dict(expansion_factor = k["dynamics_ff_expansion_factor"])) if k["dynamics_ff_expansion_factor"] is not None else dict()),
        attn_heads = k["attn_heads"], attn_dim_head = k["attn_dim_head"], policy_head_mlp_depth = 2, value_head_mlp_depth = 2,
    )
    package = torch.load(checkpoint, map_location = "cpu", weights_only = False)
    tokenizer.load_state_dict(package["tokenizer"], strict = True)
    world_model.load_state_dict(package["world_model"], strict = True)
    tokenizer.eval()
    world_model.eval()
    return k, tokenizer, world_model, package


def replay_arrays(run, episodes, fields = ("critic_state", "actions_continuous", "rewards", "log_probs_continuous")):
    replay = run_dir(run) / "replay_buffer"
    out = {}
    for field in fields:
        out[field] = np.asarray(np.load(replay / f"{field}.data.npy", mmap_mode = "r")[episodes, :STEPS])
    return out


@torch.no_grad()
def encode(tokenizer, states):
    """states (b, t, 17) -> latents (b, t, 1, n, d); the tokenizer encodes each frame alone."""
    b, t, _ = states.shape
    latents = tokenizer(torch.from_numpy(states).float().reshape(b * t, -1), return_latents = True)
    return latents.reshape(b, t, 1, *latents.shape[-2:])


@torch.no_grad()
def policy_embed_of(world_model, embeds):
    """The policy embed of the last frame, as interact_with_env computes it."""
    actor = embeds.actor if embeds.actor is not None else embeds.agent
    policy_input = world_model.select_policy_input(embeds, 0)[:, -1:] if world_model.policy_all_tokens else actor[..., -1:, 0, :]
    return world_model.policy_head(policy_input)


@torch.no_grad()
def acting_policy(world_model, latents, actions, rewards, start = 0, stop = None, keep = None):
    """Run the cached step loop of interact_with_env on a recorded real episode (teacher forcing).

    latents (b, T, 1, n, d), actions (b, T, 6) native, rewards (b, T). The step at frame t reads latents t, action t-1
    and reward t-1 with the time cache of frames start .. t-1. Returns {t: policy_embed (b, 1, d)} for t in keep.
    """
    stop = latents.shape[1] if stop is None else stop
    step_size = world_model.max_steps // 4
    time_cache = None
    out = {}
    for t in range(start, stop):
        _, (embeds, time_cache) = world_model(
            latents = latents[:, t:t + 1], signal_levels = world_model.max_steps - 1, step_sizes = step_size,
            rewards = rewards[:, t - 1:t] if t > start else None,
            continuous_actions = actions[:, t - 1:t] if t > start else None,
            time_cache = time_cache, latent_is_noised = True, latent_has_view_dim = True,
            return_pred_only = True, return_intermediates = True)
        if keep is None or t in keep:
            out[t] = policy_embed_of(world_model, embeds)
    return out


def beta(world_model, policy_embed):
    """The Beta policy (6 dims) of the policy embeds."""
    _, params = world_model.action_embedder.unembed(policy_embed, pred_head_index = 0, return_split_discrete = True)
    return world_model.action_embedder.continuous_readout.continuous_dist.dist(params)


LOG_RANGE = 6 * np.log(2.)


@torch.no_grad()
def entropy(dist):
    """Entropy summed over the action dims, with the 6 log 2 range term of the training log."""
    return dist.entropy().sum(-1) + LOG_RANGE


@torch.no_grad()
def log_prob(dist, native_actions, eps = 1e-6):
    return dist.log_prob(native_actions.clamp(eps, 1. - eps)).sum(-1)


@torch.no_grad()
def kl(dist_p, dist_q):
    """KL(p || q), summed over the action dims."""
    return torch.distributions.kl_divergence(dist_p, dist_q).sum(-1)
