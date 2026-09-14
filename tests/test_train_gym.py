import warnings

import gymnasium as gym
import numpy as np
import pytest
import torch

import train.train_gym as train_gym
from dreamer4.dreamer4 import DynamicsWorldModel
from train.compile_runtime import ImaginationGenerateRollout
from train.train_gym import RolloutTimingWrapper, pop_rollout_env_timings, repeat_batch_to_size


def test_repeat_batch_to_size_repeats_all_tensor_fields():
    batch = {
        "latents": torch.tensor([[1], [2]]),
        "rewards": torch.tensor([[3], [4]]),
        "metadata": "unchanged",
    }

    repeated = repeat_batch_to_size(batch, 5)

    assert repeated["latents"].squeeze(-1).tolist() == [1, 2, 1, 2, 1]
    assert repeated["rewards"].squeeze(-1).tolist() == [3, 4, 3, 4, 3]
    assert repeated["metadata"] == "unchanged"


def test_old_observation_tokenizer_uses_plain_silu_mlp():
    tokenizer = train_gym.ObservationTokenizer(
        obs_dim = 3,
        num_latent_tokens = 2,
        dim_latent = 4,
        hidden_dim = 8,
        old_state_tokenizer = True,
    )
    observations = torch.randn(5, 3)

    latents = tokenizer(observations, return_latents = True)

    assert tokenizer.old_state_tokenizer
    assert isinstance(tokenizer.tokenizer, train_gym.OldStateTokenizer)
    assert latents.shape == (5, 2, 4)
    assert any(isinstance(module, torch.nn.SiLU) for module in tokenizer.tokenizer.encoder)


def test_resampling_dataloader_iterator_recreates_each_epoch():
    class ChangingLoader:
        def __init__(self):
            self.epoch = 0

        def __iter__(self):
            self.epoch += 1
            yield self.epoch

    iterator = train_gym.resampling_dataloader_iterator(ChangingLoader())

    assert [next(iterator), next(iterator), next(iterator)] == [1, 2, 3]


def test_old_world_model_heads_get_dedicated_clip_groups():
    class LegacyHeads(torch.nn.Module):
        old_reward_head = True
        old_terminal_head = True

        def __init__(self):
            super().__init__()
            self.to_agent_embed_reward_pred = torch.nn.Linear(2, 2)
            self.to_latent_state_terminal_pred = torch.nn.Linear(2, 1)

    world_model = LegacyHeads()
    optimizer = torch.optim.AdamW(world_model.parameters(), lr = 1e-3)

    groups = dict(train_gym.world_model_clip_groups(world_model, optimizer))

    assert set(groups["reward_head"]) == set(world_model.to_agent_embed_reward_pred.parameters())
    assert set(groups["terminal_head"]) == set(world_model.to_latent_state_terminal_pred.parameters())
    assert not set(groups["reward_head"]) & set(groups["terminal_head"])


def test_agent_optimizer_parameters_include_world_model_only_when_enabled():
    world_model = DynamicsWorldModel(
        dim = 8,
        dim_latent = 4,
        max_steps = 8,
        num_latent_tokens = 1,
        num_spatial_tokens = 1,
        depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        num_continuous_actions = 2,
        reward_encoder_kwargs = dict(num_bins = 5),
        value_encoder_kwargs = dict(num_bins = 5),
    )

    disabled = train_gym.agent_optimizer_parameters(world_model, False)
    enabled = train_gym.agent_optimizer_parameters(world_model, True)

    assert disabled == world_model.agent_parameters()
    assert set(enabled) == set(world_model.parameters())
    assert not set(disabled) & set(world_model.transformer.parameters())
    assert set(world_model.transformer.parameters()) <= set(enabled)


def test_dynamics_loss_normalization_freeze_requires_normalization():
    with pytest.raises(ValueError, match = "requires use_dynamics_loss_normalization=true"):
        train_gym.validate_dynamics_loss_normalization(False, True)


def test_active_dynamics_loss_normalization_warns_without_freeze():
    with pytest.warns(UserWarning, match = "RMS statistics remain active"):
        train_gym.validate_dynamics_loss_normalization(True, False)


def test_disabled_dynamics_loss_normalization_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train_gym.validate_dynamics_loss_normalization(False, False)


def test_rollout_episode_stats_use_returns_and_exclude_bootstrap_padding():
    experience = type("Experience", (), {
        "episode_return": torch.tensor([10., 4.]),
        "lens": torch.tensor([6, 3]),
        "is_truncated": torch.tensor([True, False]),
        "terminals": torch.tensor([False, True]),
    })()

    returns, lengths = train_gym.get_rollout_episode_stats(experience)

    assert returns == [10., 4.]
    assert lengths == [5, 3]


def test_checkpoint_saves_muon_optimizer_with_lambda_group_metadata(tmp_path):
    tokenizer = torch.nn.Linear(2, 2)
    world_model = torch.nn.Linear(2, 2)
    world_params = list(world_model.parameters())
    world_optimizer = train_gym.make_optimizer(
        world_params,
        lr = 1e-4,
        weight_decay = 0.,
        use_muon = True,
        muon_params = [world_model.weight],
    )
    agent_optimizer = torch.optim.AdamW(tokenizer.parameters(), lr = 1e-4)

    world_model(torch.ones(1, 2)).sum().backward()
    world_optimizer.step()

    # adam-atan2-pytorch versions affected by the checkpoint failure copied this
    # default lambda into every parameter group returned by state_dict().
    for group in world_optimizer.param_groups:
        group["muon_bypass_update_fn"] = world_optimizer.muon_bypass_update_fn

    checkpoint_path = tmp_path / "checkpoint.pt"
    train_gym.save_checkpoint(
        checkpoint_path,
        loop = 3,
        tokenizer = tokenizer,
        world_model = world_model,
        world_optimizer = world_optimizer,
        agent_optimizer = agent_optimizer,
    )

    checkpoint = torch.load(checkpoint_path, map_location = "cpu", weights_only = True)
    resumed_optimizer = train_gym.make_optimizer(
        world_params,
        lr = 1e-4,
        weight_decay = 0.,
        use_muon = True,
        muon_params = [world_model.weight],
    )
    for group in resumed_optimizer.param_groups:
        group["muon_bypass_update_fn"] = resumed_optimizer.muon_bypass_update_fn

    assert train_gym.load_optimizer_state_if_compatible(
        resumed_optimizer,
        checkpoint["world_optimizer"],
        "world",
    )
    assert checkpoint["loop"] == 3
    assert len(resumed_optimizer.state) == len(world_optimizer.state)
    assert all(
        callable(group["muon_bypass_update_fn"])
        for group in resumed_optimizer.param_groups
    )


def test_generate_rollout_forwards_preallocation_and_time_cache():
    class FakeWorldModel:
        def generate(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return "dream"

    world_model = FakeWorldModel()
    rollout = ImaginationGenerateRollout(
        world_model,
        generate_steps = 4,
        store_old_action_unembeds = True,
        use_time_cache = True,
        preallocate_outputs = True,
        optimize_generate_compilation = True,
    )

    assert rollout(12, 8) == "dream"
    assert world_model.args == (12,)
    assert world_model.kwargs["batch_size"] == 8
    assert world_model.kwargs["use_time_cache"] is True
    assert world_model.kwargs["preallocate_outputs"] is True
    assert world_model.kwargs["optimize_generate_compilation"] is True


@pytest.mark.parametrize(
    ("mode", "expected_vector_kwargs"),
    (("async", {"context": "spawn"}), ("sync", None)),
)
def test_make_env_forwards_vectorization_mode(monkeypatch, mode, expected_vector_kwargs):
    calls = []
    original_make_vec = gym.make_vec

    def recording_make_vec(env_name, **kwargs):
        calls.append((env_name, kwargs))
        return original_make_vec("Pendulum-v1", num_envs = 1, vectorization_mode = "sync")

    monkeypatch.setattr(train_gym.gym, "make_vec", recording_make_vec)
    env = train_gym.make_env("Pendulum-v1", 42, num_envs = 2, env_vectorization_mode = mode)
    env.close()

    assert calls == [(
        "Pendulum-v1",
        {
            "num_envs": 2,
            "vectorization_mode": mode,
            "vector_kwargs": expected_vector_kwargs,
        },
    )]


def test_rollout_timing_wrapper_aggregates_and_resets():
    base_env = gym.make_vec("Pendulum-v1", num_envs = 2, vectorization_mode = "sync")
    env = RolloutTimingWrapper(base_env)

    env.reset(seed = 42)
    env.step(np.zeros(env.action_space.shape, dtype = np.float32))
    reset_seconds, step_seconds = pop_rollout_env_timings(env)

    assert reset_seconds > 0.
    assert step_seconds > 0.
    assert pop_rollout_env_timings(env) == (0., 0.)
    env.close()


@pytest.mark.parametrize("old_continuous_action_encoder", (False, True))
def test_interact_uses_host_done_flags_and_injected_forward(old_continuous_action_encoder):
    class StaggeredDoneEnv:
        def reset(self, seed = None):
            self.step_index = 0
            return np.zeros((2, 4), dtype = np.float32)

        def step(self, actions):
            assert actions.shape == (2, 1)
            self.step_index += 1
            observations = np.full((2, 4), self.step_index, dtype = np.float32)
            rewards = np.ones((2,), dtype = np.float32)
            terminated = np.array((self.step_index >= 1, self.step_index >= 3))
            truncated = np.zeros((2,), dtype = np.bool_)
            return observations, rewards, terminated, truncated, {}

    model = DynamicsWorldModel(
        dim = 8,
        dim_latent = 4,
        max_steps = 8,
        num_latent_tokens = 1,
        num_spatial_tokens = 1,
        depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        num_discrete_actions = 0,
        num_continuous_actions = 1,
        continuous_dist_type = "beta",
        continuous_target_action_range = (-1., 1.),
        old_continuous_action_encoder = old_continuous_action_encoder,
        reward_encoder_kwargs = dict(num_bins = 5),
        value_encoder_kwargs = dict(num_bins = 5),
        policy_head_mlp_depth = 1,
        value_head_mlp_depth = 1,
    ).eval()
    forward_calls = []

    def forward_fn(**kwargs):
        forward_calls.append(kwargs)
        return model.forward(**kwargs)

    def observations_to_latents(_model, obs, time_cache):
        latents = torch.as_tensor(obs["state"], dtype = torch.float32)
        return latents[:, None, None, :], time_cache

    experience = model.interact_with_env(
        StaggeredDoneEnv(),
        max_timesteps = 5,
        env_is_vectorized = True,
        obs_to_latents_fn = observations_to_latents,
        forward_fn = forward_fn,
    )

    assert len(forward_calls) == 3
    assert experience.lens.tolist() == [1, 3]
    assert experience.episode_return.tolist() == [1., 3.]
    assert experience.terminals.tolist() == [True, True]


def test_async_vector_env_reset_and_step_smoke():
    env = train_gym.make_env("Pendulum-v1", 42, num_envs = 2, env_vectorization_mode = "async")

    try:
        observations, _ = env.reset(seed = 42)
        step_out = env.step(np.zeros(env.action_space.shape, dtype = np.float32))
    finally:
        env.close()

    assert observations.shape == (2, 3)
    assert step_out[0].shape == (2, 3)
