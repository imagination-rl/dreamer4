from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

from dreamer4.dreamer4 import Actions, Experience
from train import train_gym, train_model_free


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "train" / "config.yaml"


def make_agent(*, normalization = "none"):
    return train_model_free.StateActorCritic(
        obs_dim = 3,
        action_dim = 1,
        action_range = (-2., 2.),
        dim = 8,
        encoder_depth = 2,
        observation_normalization = normalization,
        reward_encoder_type = "hl_gauss",
        use_symexp_value_head = False,
        use_tdmpc2_value_head = False,
        use_symlog_hl_gauss_value_head = False,
        use_hl_gauss_actor_critic = False,
        use_baseline_symexp_value_head_init = False,
        value_encoder_num_bins = 7,
        value_range = (-20., 20.),
        value_encoder_sigma_to_bin_ratio = .75,
        value_min_max_value_on_bin_center = False,
        value_use_symlog = False,
        value_decode_symlog_expected_raw = False,
        hl_gauss_actor_critic_range = (-20., 20.),
        hl_gauss_actor_critic_num_bins = 7,
        hl_gauss_actor_critic_sigma_to_bin_ratio = .75,
        hl_gauss_actor_critic_min_max_value_on_bin_center = False,
        normalize_advantages = False,
        pmpo_pos_to_neg_weight = .5,
        pmpo_kl_div_loss_weight = .3,
        use_delight_gating = False,
        old_continuous_action_encoder = False,
        old_pmpo = True,
        old_policy_head = False,
        old_value_head = False,
        value_head_init_value = None,
    )


@torch.no_grad()
def make_experience(agent, *, batch = 4, time = 4):
    states = torch.randn(batch, time, 3)
    features = agent.encode_states(states)
    policy_embed = agent.policy_head(features)
    actions = agent.action_embedder.sample(policy_embed, pred_head_index = 0)[1]
    log_probs = agent.action_embedder.log_probs(
        policy_embed,
        pred_head_index = 0,
        continuous_targets = actions,
    )[1]
    values = agent.value_encoder.bins_to_scalar_value(agent.value_head(features))
    action_params = agent.action_embedder.unembed(policy_embed, pred_head_index = 0)[1]
    return Experience(
        latents = states,
        critic_state = states,
        rewards = torch.randn(batch, time),
        terminals = torch.zeros(batch, dtype = torch.bool),
        actions = Actions(None, actions),
        log_probs = Actions(None, log_probs),
        old_action_unembeds = Actions(None, action_params),
        values = values,
        step_size = 1,
        lens = torch.full((batch,), time, dtype = torch.long),
        is_truncated = torch.zeros(batch, dtype = torch.bool),
        is_from_world_model = False,
    )


def test_small_model_flag_dispatches_before_dreamer_construction(monkeypatch):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["small_model_only"] = True
    captured = {}

    def run_small_model(**kwargs):
        captured.update(kwargs)

    class UnexpectedModel:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the state-only dispatch constructed a Dreamer model")

    monkeypatch.setattr(train_model_free, "run", run_small_model)
    monkeypatch.setattr(train_gym, "ObservationTokenizer", UnexpectedModel)
    monkeypatch.setattr(train_gym, "DynamicsWorldModel", UnexpectedModel)
    train_gym.main(**config)

    assert captured["small_model_only"] is True


def test_cleanrl_pmpo_flag_dispatches_to_isolated_trainer(monkeypatch):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["small_model"]["cleanrl_pmpo"] = True
    captured = {}

    def run_cleanrl_pmpo(*, config):
        captured.update(config)

    monkeypatch.setattr(train_model_free, "run_cleanrl_pmpo", run_cleanrl_pmpo)
    train_model_free.run(**config)

    assert captured["small_model"]["cleanrl_pmpo"] is True
    assert captured["objective"] == "pmpo"


def test_state_actor_critic_logs_optimizer_learning_rate(monkeypatch, tmp_path):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config.update(
        small_model_only = True,
        cpu = True,
        require_cuda = False,
        num_loops = 1,
        rollouts_per_loop = 1,
        num_envs = 1,
        max_timesteps = 2,
        env_vectorization_mode = "sync",
        agent_learning_rate = .0123,
        use_muon_optimizer = False,
        checkpoint_every = 0,
        checkpoint_path = None,
        checkpoint_folder = None,
        clear_log_dir = False,
        unique_log_dir = False,
        use_tensorboard = False,
        log_dir = str(tmp_path),
        run_name = "small_model_learning_rate",
    )
    config["small_model"].update(
        dim = 8,
        encoder_depth = 2,
        observation_normalization = "none",
        cleanrl_pmpo = False,
    )
    logged_scalars = []

    class StubEnv:
        def close(self):
            return None

    monkeypatch.setattr(
        train_gym,
        "inspect_env_spaces",
        lambda env_name, obs_dim: (3, 1, (-1., 1.)),
    )
    monkeypatch.setattr(train_gym, "make_env", lambda *args, **kwargs: StubEnv())
    monkeypatch.setattr(
        train_model_free,
        "interact_with_env",
        lambda agent, env, **kwargs: make_experience(agent, batch = 1, time = 2),
    )
    monkeypatch.setattr(
        train_gym,
        "get_rollout_episode_stats",
        lambda experience: ([1.], [2]),
    )
    monkeypatch.setattr(train_gym, "pop_rollout_env_timings", lambda env: (0., 0.))
    monkeypatch.setattr(train_model_free, "combine_experiences", lambda experiences: experiences[0])
    monkeypatch.setattr(
        train_model_free,
        "train_on_experience",
        lambda *args, **kwargs: (1, {"state_actor_critic/loss": 0.}),
    )
    monkeypatch.setattr(
        train_gym,
        "log_scalars",
        lambda writer, scalars, step: logged_scalars.append((scalars, step)),
    )

    train_model_free.run(**config)

    learning_rate_logs = [
        (scalars, step)
        for scalars, step in logged_scalars
        if "charts/learning_rate" in scalars
    ]
    assert learning_rate_logs == [({"charts/learning_rate": .0123}, 2)]


def test_cleanrl_pmpo_ablation_flags_require_booleans():
    with pytest.raises(TypeError, match = "cleanrl_pmpo_joint_log_prob"):
        train_model_free.resolve_small_model_config({"cleanrl_pmpo_joint_log_prob": 1})


def test_cleanrl_updated_ppo_uses_bounded_beta_actions_and_physical_densities():
    torch.manual_seed(17)
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape = (4,), dtype = np.float32)
    action_space = gym.spaces.Box(
        low = np.array((-2., 1.), dtype = np.float32),
        high = np.array((2., 3.), dtype = np.float32),
    )
    agent = train_model_free.CleanRLUpdatedPPOAgent(observation_space, action_space)
    observations = torch.randn(8, 4)

    native_actions, actions, log_probs, entropies, values = agent.sample_action_and_value(
        observations
    )
    _, evaluated_log_probs, evaluated_entropies, evaluated_values = agent.get_action_and_value(
        observations,
        actions,
    )

    assert native_actions.shape == (8, 2)
    assert torch.all((native_actions > 0.) & (native_actions < 1.))
    assert torch.all(actions >= agent.action_low)
    assert torch.all(actions <= agent.action_high)
    assert torch.allclose(evaluated_log_probs, log_probs)
    assert torch.allclose(evaluated_entropies, entropies)
    assert torch.allclose(evaluated_values, values)


def test_cleanrl_updated_ppo_gae_bootstraps_truncation_without_trace_leakage():
    advantages, returns = train_model_free.cleanrl_updated_ppo_gae(
        rewards = torch.tensor(((1.,), (2.,))),
        values = torch.tensor(((10.,), (20.,))),
        terminations = torch.zeros(2, 1),
        truncations = torch.tensor(((1.,), (0.,))),
        truncation_bootstrap_values = torch.tensor(((30.,), (0.,))),
        rollout_tail_value = torch.tensor((40.,)),
    )

    assert advantages[:, 0].tolist() == pytest.approx((20.7, 21.6))
    assert returns[:, 0].tolist() == pytest.approx((30.7, 41.6))


def test_cleanrl_updated_ppo_accepts_current_gymnasium_final_observation_keys():
    next_observations = np.zeros((2, 3), dtype = np.float32)
    final_observation = np.array((1., 2., 3.), dtype = np.float32)

    bootstrap = train_model_free.cleanrl_updated_ppo_bootstrap_observations(
        next_observations,
        np.array((True, False)),
        {
            "final_obs": np.array((final_observation, None), dtype = object),
            "_final_obs": np.array((True, False)),
        },
    )

    assert np.array_equal(bootstrap[0], final_observation)
    assert np.array_equal(bootstrap[1], next_observations[1])


def test_cleanrl_pmpo_constant_learning_rate_flag_disables_decay():
    initial_learning_rate = 3e-4

    assert train_model_free.cleanrl_pmpo_learning_rate(
        initial_learning_rate,
        50,
        100,
    ) == pytest.approx(1.5e-4)
    assert train_model_free.cleanrl_pmpo_learning_rate(
        initial_learning_rate,
        50,
        100,
        constant = True,
    ) == initial_learning_rate


def test_cleanrl_pmpo_joint_log_prob_restores_action_dimension_scale():
    torch.manual_seed(7)
    action_dim = 3
    agent = train_model_free.CleanRLPmpoAgent(obs_dim = 4, action_dim = action_dim)
    observations = torch.randn(8, 4)
    actions = torch.randn(8, action_dim)
    advantages = torch.tensor((1., -1., 2., -2., 3., -3., 4., -4.))
    returns = torch.randn(8)
    old_values = torch.randn(8)
    prior_actor_mean = deepcopy(agent.actor_mean).eval()
    prior_logstd = agent.actor_logstd.detach().clone()
    loss_kwargs = dict(
        observations = observations,
        actions = actions,
        advantages = advantages,
        returns = returns,
        old_values = old_values,
        prior_actor_mean = prior_actor_mean,
        prior_logstd = prior_logstd,
        pmpo_alpha = .5,
        pmpo_kl_div_loss_weight = 0.,
    )

    legacy_policy_loss = train_model_free.cleanrl_pmpo_losses(
        agent,
        **loss_kwargs,
    )[1]
    joint_policy_loss = train_model_free.cleanrl_pmpo_losses(
        agent,
        **loss_kwargs,
        joint_action_log_prob = True,
    )[1]

    assert joint_policy_loss.item() == pytest.approx(
        action_dim * legacy_policy_loss.item()
    )


def test_cleanrl_pmpo_q_ranked_preferences_train_q_and_actor():
    torch.manual_seed(11)
    agent = train_model_free.CleanRLPmpoAgent(
        obs_dim = 4,
        action_dim = 2,
        q_ranked_preferences = True,
    )
    observations = torch.randn(8, 4)
    actions = torch.randn(8, 2)
    prior_actor_mean = deepcopy(agent.actor_mean).eval()
    results = train_model_free.cleanrl_pmpo_losses(
        agent,
        observations = observations,
        actions = actions,
        advantages = torch.randn(8),
        returns = torch.randn(8),
        old_values = torch.randn(8),
        prior_actor_mean = prior_actor_mean,
        prior_logstd = agent.actor_logstd.detach().clone(),
        pmpo_alpha = .5,
        pmpo_kl_div_loss_weight = .3,
        q_ranked_preferences = True,
    )
    loss, _, _, _, _, q_value_loss, q_preference_gap = results
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(q_value_loss)
    assert q_preference_gap >= 0.
    assert any(parameter.grad is not None for parameter in agent.actor_mean.parameters())
    assert any(parameter.grad is not None for parameter in agent.q_critic.parameters())


def test_cleanrl_completed_episodes_reads_current_and_legacy_vector_infos():
    current_infos = {
        "episode": {
            "r": np.array((12.5, 0., -3.)),
            "l": np.array((1000, 0, 400)),
        },
        "_episode": np.array((True, False, True)),
    }
    legacy_infos = {
        "final_info": np.array((
            {"episode": {"r": 8.5, "l": 700}},
            None,
        ), dtype = object),
    }
    same_step_infos = {
        "final_info": {
            "episode": {
                "r": np.array((4.5, 0., 6.5)),
                "l": np.array((200, 0, 300)),
            },
            "_episode": np.array((True, False, True)),
        },
        "_final_info": np.array((True, False, True)),
    }

    assert train_model_free.cleanrl_completed_episodes(current_infos) == [(12.5, 1000), (-3., 400)]
    assert train_model_free.cleanrl_completed_episodes(legacy_infos) == [(8.5, 700)]
    assert train_model_free.cleanrl_completed_episodes(same_step_infos) == [(4.5, 200), (6.5, 300)]


class OneStepVectorEnv:
    def reset(self, seed = None):
        self.actions = []
        return np.zeros((2, 3), dtype = np.float32)

    def step(self, actions):
        self.actions.append(actions.copy())
        assert actions.shape == (2, 1)
        return (
            np.ones((2, 3), dtype = np.float32),
            np.array((1., 2.), dtype = np.float32),
            np.array((False, True)),
            np.array((True, False)),
            {},
        )


def test_state_rollout_keeps_native_actions_and_bootstraps_truncation():
    agent = make_agent()
    env = OneStepVectorEnv()
    experience = train_model_free.interact_with_env(agent, env, seed = 42, max_timesteps = 3)

    assert experience.actions.continuous.min() >= 0.
    assert experience.actions.continuous.max() <= 1.
    assert np.all(env.actions[0] >= -2.)
    assert np.all(env.actions[0] <= 2.)
    assert experience.lens.tolist() == [2, 1]
    assert experience.terminals.tolist() == [False, True]
    assert experience.is_truncated.tolist() == [True, False]
    assert torch.isfinite(experience.log_probs.continuous).all()
    assert torch.isfinite(experience.values).all()


@pytest.mark.parametrize("objective", ("ppo", "pmpo", "reinforce"))
def test_state_actor_critic_losses_are_finite_and_backpropagate(objective):
    agent = make_agent()
    experience = make_experience(agent)
    policy_loss, value_loss = train_model_free.actor_critic_losses(agent, experience, objective = objective)
    (policy_loss + value_loss).backward()

    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)
    assert any(parameter.grad is not None for parameter in agent.state_mlp.parameters())
    assert any(parameter.grad is not None for parameter in agent.policy_head.parameters())
    assert any(parameter.grad is not None for parameter in agent.value_head.parameters())
    assert agent.action_embedder.continuous_action_unembed.grad is not None


def test_state_actor_critic_rejects_spo():
    agent = make_agent()
    with pytest.raises(ValueError, match = "objective='spo'"):
        train_model_free.actor_critic_losses(agent, make_experience(agent), objective = "spo")


def test_reinforce_small_model_requires_one_full_batch_update():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["objective"] = "reinforce"
    config["small_model"]["update_epochs"] = 2

    with pytest.raises(ValueError, match = "keep each update on-policy"):
        train_model_free.run(**config)

    config["small_model"]["update_epochs"] = 1
    config["small_model"]["minibatch_num_envs"] = 2

    with pytest.raises(ValueError, match = "keep each update on-policy"):
        train_model_free.run(**config)


def test_minibatches_preserve_sequences_and_running_statistics():
    agent = make_agent(normalization = "running")
    experience = make_experience(agent, batch = 4, time = 5)
    optimizer = torch.optim.AdamW(agent.optimizer_parameters(), lr = 1e-3)
    update_step, metrics = train_model_free.train_on_experience(
        agent,
        optimizer,
        experience,
        objective = "ppo",
        update_epochs = 2,
        minibatch_num_envs = 2,
        max_grad_norm = 1.,
        update_step = 0,
        writer = None,
        env_step = 20,
    )

    assert update_step == 4
    assert agent.observation_normalizer.count.item() == 40
    assert torch.isfinite(torch.tensor(metrics["state_actor_critic/loss"]))


def test_state_actor_critic_uses_standard_gradient_clip_groups(monkeypatch):
    agent = make_agent()
    experience = make_experience(agent)
    optimizer = torch.optim.AdamW(agent.optimizer_parameters(), lr = 1e-3)
    original_clip = train_gym.clip_grad_norm_by_group
    captured = {}

    def capture_clip(groups, max_grad_norm):
        captured["names"] = [name for name, _ in groups]
        return original_clip(groups, max_grad_norm)

    monkeypatch.setattr(train_gym, "clip_grad_norm_by_group", capture_clip)
    _, metrics = train_model_free.train_on_experience(
        agent,
        optimizer,
        experience,
        objective = "pmpo",
        update_epochs = 1,
        minibatch_num_envs = None,
        max_grad_norm = 1.,
        update_step = 0,
        writer = None,
        env_step = 16,
    )

    assert captured["names"] == ["policy_head", "action_unembed", "value_head", "agent_rest"]
    assert "state_actor_critic/policy_head_grad_norm" in metrics
    assert "state_actor_critic/action_unembed_grad_norm" in metrics
    assert "state_actor_critic/value_head_grad_norm" in metrics
    assert "state_actor_critic/agent_rest_grad_norm" in metrics


@pytest.mark.parametrize("mode", ("none", "fixed_random", "running"))
def test_observation_normalizer_modes_and_checkpoint_restore(tmp_path, mode):
    agent = make_agent(normalization = mode)
    observations = torch.tensor(((1., 2., 3.), (3., 4., 5.)))
    if mode == "fixed_random":
        agent.observation_normalizer.set_fixed_statistics(observations)
    if mode == "running":
        agent.observation_normalizer.update(observations)

    optimizer = torch.optim.AdamW(agent.optimizer_parameters(), lr = 1e-3)
    checkpoint_path = tmp_path / "state.pt"
    train_model_free.save_checkpoint(
        checkpoint_path,
        agent = agent,
        optimizer = optimizer,
        loop = 3,
        update_step = 4,
        env_step = 5,
    )
    restored = make_agent(normalization = mode)
    restored_optimizer = torch.optim.AdamW(restored.optimizer_parameters(), lr = 1e-3)
    checkpoint = train_model_free.load_checkpoint(checkpoint_path, agent = restored, optimizer = restored_optimizer)

    assert checkpoint["loop"] == 3
    assert torch.equal(restored.observation_normalizer.mean, agent.observation_normalizer.mean)
    assert torch.equal(restored.observation_normalizer.variance, agent.observation_normalizer.variance)

    wrong_checkpoint = tmp_path / "dreamer.pt"
    torch.save({"world_model": {}}, wrong_checkpoint)
    with pytest.raises(ValueError, match = "checkpoint type error"):
        train_model_free.load_checkpoint(wrong_checkpoint, agent = restored, optimizer = restored_optimizer)
