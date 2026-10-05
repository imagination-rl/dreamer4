import subprocess
import sys
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
from memmap_replay_buffer.replay_buffer import ReplayBuffer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import train.train_gym as train_gym
from dreamer4.dreamer4 import Actions, DynamicsWorldModel, Experience
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


def test_cleanrl_updated_ppo_flattens_only_learnable_transitions():
    target_settings = type(
        "TargetSettings",
        (),
        {
            "gae_discount_factor": 0.99,
            "gae_lambda": 0.95,
            "gae_use_accelerated": False,
        },
    )()
    latents = torch.arange(8).reshape(2, 4, 1, 1).float()
    dream = Experience(
        latents = latents,
        rewards = torch.tensor(((1., 2., 3., 4.), (4., 3., 2., 0.))),
        actions = Actions(None, torch.zeros(2, 4, 1)),
        log_probs = Actions(None, torch.zeros(2, 4, 1)),
        values = torch.zeros(2, 4),
        step_size = 1,
        lens = torch.tensor((4, 3)),
        is_truncated = torch.tensor((True, False)),
    )

    transitions = train_gym.flatten_cleanrl_updated_ppo_transitions(
        target_settings,
        dream,
    )

    assert transitions.latents[:, 0, 0, 0].tolist() == [0., 1., 2., 4., 5., 6.]
    assert transitions.rewards.shape == (6, 1)
    assert transitions.frozen_returns.shape == (6, 1)
    assert transitions.frozen_advantages.shape == (6, 1)
    assert transitions.lens.tolist() == [1] * 6
    assert not transitions.is_truncated.any()
    assert transitions.frozen_advantages.std() != pytest.approx(1.)


def test_agent_approx_kl_adds_cleanrl_metrics_only_when_requested():
    class FakeActionEmbedder:
        def log_probs(self, policy_embed, **_kwargs):
            return Actions(None, policy_embed)

    world_model = type("WorldModel", (), {
        "policy_all_tokens": False,
        "policy_head": torch.nn.Identity(),
        "action_embedder": FakeActionEmbedder(),
        "ppo_eps_clip": .2,
    })()
    dream = Experience(
        latents = torch.zeros(2, 1, 1, 1),
        agent_embed = torch.tensor((((np.log(.5),),), ((0.,),))),
        actions = Actions(None, torch.zeros(2, 1, 1)),
        log_probs = Actions(None, torch.zeros(2, 1, 1)),
        lens = torch.ones(2, dtype = torch.long),
        is_truncated = torch.zeros(2, dtype = torch.bool),
    )

    default_metrics = train_gym.agent_approx_kl_metrics(world_model, dream)
    closer_metrics = train_gym.agent_approx_kl_metrics(
        world_model,
        dream,
        include_cleanrl_metrics = True,
    )

    assert default_metrics.keys() == {
        "approx_kl_mean",
        "approx_kl_min",
        "approx_kl_max",
    }
    assert closer_metrics.keys() == default_metrics.keys() | {
        "old_approx_kl",
        "clip_fraction",
    }
    assert closer_metrics["clip_fraction"] == pytest.approx(.5)


def test_dynamics_loss_normalization_freeze_requires_normalization():
    with pytest.raises(ValueError, match = "requires use_dynamics_loss_normalization=true"):
        train_gym.validate_dynamics_loss_normalization(False, True)


def test_grouped_dynamics_loss_normalization_requires_normalization_and_the_default_objective():
    with pytest.raises(ValueError, match = "requires use_dynamics_loss_normalization=true"):
        train_gym.validate_dynamics_loss_normalization(False, False, group_dynamics_loss_normalization = True)

    with pytest.raises(ValueError, match = "already has one dynamics loss normalizer"):
        train_gym.validate_dynamics_loss_normalization(
            True, True, group_dynamics_loss_normalization = True, use_paper_aligned_shortcut_objective = True
        )


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


def test_muon_optimizer_keeps_the_module_parameter_order():
    # A saved optimizer state maps to parameters by position. A set-based order
    # differs in each process and loads the state into the wrong parameters.
    model = torch.nn.Sequential(*(torch.nn.Linear(4, 4) for _ in range(8)))
    params = list(model.parameters())
    muon_params = [layer.weight for layer in model]

    optimizer = train_gym.make_optimizer(
        params,
        lr = 1e-4,
        weight_decay = 0.,
        use_muon = True,
        muon_params = muon_params,
    )

    def ids(group_params):
        return [id(param) for param in group_params]

    assert ids(optimizer.param_groups[0]["params"]) == ids(layer.bias for layer in model)
    assert ids(optimizer.param_groups[1]["params"]) == ids(muon_params)


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


def test_link_run_dir_to_storage_writes_through_symlink(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = Path("runs/gym_imagination/example")

    storage = train_gym.link_run_dir_to_storage(run_dir, tmp_path / "storage")

    assert storage == (tmp_path / "storage/runs/gym_imagination/example").resolve()
    assert run_dir.is_symlink() and run_dir.resolve() == storage
    (run_dir / "events.txt").write_text("x")
    assert (storage / "events.txt").read_text() == "x"

    with pytest.raises(FileExistsError):
        train_gym.link_run_dir_to_storage(run_dir, tmp_path / "storage")

    # A directory outside the working directory stays in place.
    assert train_gym.link_run_dir_to_storage(tmp_path.parent / "elsewhere", tmp_path / "storage") is None


def test_grouped_async_env_matches_sync_env():
    # 5 envs in 2 workers (3 + 2). Pendulum truncates at 200 steps, so the steps cross one autoreset.
    sync_env = train_gym.make_env("Pendulum-v1", 7, num_envs = 5, env_vectorization_mode = "sync")
    grouped_env = train_gym.make_env("Pendulum-v1", 7, num_envs = 5, parallel_envs = 2, env_vectorization_mode = "async")

    try:
        assert isinstance(grouped_env.unwrapped, train_gym.GroupedAsyncVectorEnv)

        sync_obs, _ = sync_env.reset(seed = 11)
        grouped_obs, _ = grouped_env.reset(seed = 11)
        np.testing.assert_array_equal(grouped_obs, sync_obs)

        episodes = 0
        for _ in range(230):
            actions = sync_env.action_space.sample()
            sync_out = sync_env.step(actions)
            grouped_out = grouped_env.step(actions)

            for sync_value, grouped_value in zip(sync_out[:4], grouped_out[:4]):
                np.testing.assert_array_equal(grouped_value, sync_value)

            if "episode" in sync_out[4]:
                episodes += 1
                np.testing.assert_array_equal(grouped_out[4]["episode"]["r"], sync_out[4]["episode"]["r"])
                np.testing.assert_array_equal(grouped_out[4]["_episode"], sync_out[4]["_episode"])

        assert episodes == 1
    finally:
        sync_env.close()
        grouped_env.close()


def test_grouped_env_worker_module_does_not_import_torch():
    # A spawn worker imports the module of its target. With torch, each worker held about 700 MB of host memory.
    code = "import sys, train.env_worker; assert 'torch' not in sys.modules, 'train.env_worker imports torch'"
    subprocess.run([sys.executable, "-c", code], check = True, cwd = Path(__file__).resolve().parents[1])
    assert train_gym._grouped_env_worker.__module__ == "train.env_worker"


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


def small_cpu_config(tmp_path, **overrides):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "train" / "config.yaml").read_text())
    config.update(
        log_dir = str(tmp_path / "logs"),
        run_name = "resume",
        run_details = "resume test",
        cpu = True,
        require_cuda = False,
        compile = False,
        compile_interact = False,
        compile_world_model = False,
        compile_generate = False,
        compile_learn = False,
        total_envs = 2,
        env_vectorization_mode = "sync",
        max_timesteps = 48,
        replay_size = 4,
        model_dim = 32,
        attn_heads = 2,
        attn_dim_head = 16,
        pretrain_tokenizer_steps = 2,
        tokenizer_batch_size = 32,
        pretrain_world_model_finetune_steps = 1,
        pretrain_world_model_combined_steps = 1,
        world_model_train_steps = 1,
        world_model_batch_size = 4,
        world_model_train_sequence_length = 16,
        imagination_batch_size = 4,
        imagination_horizon = 8,
        imagination_prompt_length = 4,
        imagination_train_steps = 1,
        preservation_sequence_length = 16,
        tokenizer_eval_every = 0,
        collect_timings = False,
    )
    config.update(overrides)
    return config


def test_closer_ppo_logs_the_graphs_of_the_other_objectives(tmp_path):
    from train.ablations import CLEANRL_PPO_CLOSER, NEXT_STATE_REWARD_TARGET

    # The A2 objective of the queued runs. The horizon of 9 gives PPO minibatches of two or more transitions.
    config = small_cpu_config(
        tmp_path, **{**CLEANRL_PPO_CLOSER, **NEXT_STATE_REWARD_TARGET},
        run_name = "closer", num_loops = 2, total_envs = 4, imagination_horizon = 9, checkpoint_every = 0,
    )
    train_gym.main(**config)

    accumulator = EventAccumulator(str(tmp_path / "logs" / "closer")).Reload()
    for tag in ("imagination/dream_return", "critic/value_mean", "extra/policy_entropy", "cleanrl_updated_ppo/entropy"):
        assert len(accumulator.Scalars(tag)) == 2, tag


def test_behavior_cloning_in_world_model_updates_steps_the_policy_head_with_the_world_model_optimizer(tmp_path, capsys):
    from train.ablations import CLEANRL_PPO_CLOSER, NEXT_STATE_REWARD_TARGET

    config = small_cpu_config(
        tmp_path, **{**CLEANRL_PPO_CLOSER, **NEXT_STATE_REWARD_TARGET},
        run_name = "bc", num_loops = 2, total_envs = 4, imagination_horizon = 9, checkpoint_every = 1,
        behavior_cloning_in_world_model_updates = True,
    )
    train_gym.main(**config)

    scalars = EventAccumulator(str(tmp_path / "logs" / "bc")).Reload().Tags()["scalars"]
    assert {"world_model/behavior_cloning_loss", "world_model/policy_head_grad_norm"} <= set(scalars)

    # The policy head is a third group of the world-model optimizer, and that optimizer stepped it.
    checkpoint = tmp_path / "logs" / "bc" / "checkpoints" / "loop_2.pt"
    world_optimizer = torch.load(checkpoint, weights_only = True)["world_optimizer"]
    assert len(world_optimizer["param_groups"]) == 3
    # The empty discrete-action unembed (no discrete actions) gets no gradient and no state.
    policy_ids = world_optimizer["param_groups"][2]["params"]
    stepped = [world_optimizer["state"][index]["steps"] for index in policy_ids if index in world_optimizer["state"]]
    assert len(stepped) == len(policy_ids) - 1 and min(stepped) > 0

    # A resume loads the three groups.
    capsys.readouterr()
    train_gym.main(**dict(config, num_loops = 3, checkpoint_path = str(checkpoint)))
    assert "skipping world optimizer state" not in capsys.readouterr().out


def test_replay_snapshot_drops_episodes_stored_after_the_checkpoint(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    replay = ReplayBuffer(tmp_path / "replay", max_episodes = 4, max_timesteps = 3, fields = dict(x = "float"), circular = True)

    def save(loop):
        train_gym.save_resume_checkpoint(
            tmp_path / "checkpoints" / f"loop_{loop}.pt",
            replay = replay,
            loop = loop - 1,
            tokenizer = model,
            world_model = model,
            world_optimizer = optimizer,
            agent_optimizer = optimizer,
        )

    replay.store_episode(x = np.ones(3))
    save(1)
    replay.store_episode(x = np.full(3, 2.))
    save(2)
    # These episodes overwrite slots 0 and 1 of the circular buffer after the last checkpoint.
    for value in (3., 4., 5., 6.):
        replay.store_episode(x = np.full(3, value))

    restored = train_gym.restore_replay_snapshot(tmp_path / "checkpoints" / "loop_2.pt", tmp_path / "replay")

    assert restored.num_episodes == 2
    assert restored.data["x"][:2, 0].tolist() == [1., 2.]
    # Only the newest checkpoint keeps a snapshot.
    with pytest.raises(FileNotFoundError):
        train_gym.restore_replay_snapshot(tmp_path / "checkpoints" / "loop_1.pt", tmp_path / "replay")

    # A snapshot without one of its field files (for example deleted latents) cannot resume.
    (tmp_path / "checkpoints" / "replay_loop_2" / "x.data.npy").unlink()
    with pytest.raises(FileNotFoundError, match = "x.data.npy"):
        train_gym.restore_replay_snapshot(tmp_path / "checkpoints" / "loop_2.pt", tmp_path / "replay")


def test_sparse_copy_keeps_the_content_and_leaves_zero_blocks_as_holes(tmp_path):
    source = tmp_path / "dense.npy"
    content = np.zeros(1 << 21, dtype = np.float32)
    content[:1024] = 1.
    # A dense file: every zero block is on the disk.
    source.write_bytes(content.tobytes())
    if source.stat().st_blocks * 512 < source.stat().st_size:
        pytest.skip("the filesystem compresses the zero blocks of a dense file")

    copy = train_gym.sparse_copy(source, tmp_path / "copy.npy")

    assert copy.read_bytes() == source.read_bytes()
    assert copy.stat().st_blocks < source.stat().st_blocks // 4


def test_resume_continues_the_step_counters_and_rejects_changed_optimizer_settings(tmp_path):
    config = small_cpu_config(tmp_path, num_loops = 2, checkpoint_every = 1)
    train_gym.main(**config)

    checkpoint = tmp_path / "logs" / "resume" / "checkpoints" / "loop_2.pt"
    progress = torch.load(checkpoint, weights_only = True)["progress"]
    assert progress["env_step"] == 2 * 2 * 48

    with pytest.raises(AssertionError, match = "optimizer settings"):
        train_gym.main(**dict(config, num_loops = 3, checkpoint_path = str(checkpoint), world_model_learning_rate = 1e-3))

    train_gym.main(**dict(config, num_loops = 3, checkpoint_path = str(checkpoint)))

    accumulator = EventAccumulator(str(tmp_path / "logs" / "resume")).Reload()
    steps = [event.step for event in accumulator.Scalars("charts/episodic_return")]
    assert sorted(steps) == [96, 96, 192, 192, 288, 288]


def test_world_model_micro_batches_give_the_one_pass_update(tmp_path):
    from types import SimpleNamespace

    from train import offline_pretraining as offline

    torch.manual_seed(0)
    episodes = [
        offline.offline_episode_experience(
            latents = torch.randn(6, 2, 8),
            observations = np.zeros((6, 17), dtype = np.float32),
            env_actions = np.zeros((5, 6), dtype = np.float32),
            rewards = np.full(5, float(index), dtype = np.float32),
            terminated = False,
            truncated = True,
            action_range = (-1., 1.),
            step_size = 16,
        )
        for index in range(4)
    ]
    replay = Experience.create_memmap_replay_buffer(episodes[0], tmp_path, max_episodes = 4, max_timesteps = 16, circular = False)
    for episode in episodes:
        offline.store_experience(replay, episode)

    class FakeWorldModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        @property
        def device(self):
            return self.weight.device

    def update(micro_batch_size):
        world_model = FakeWorldModel()
        ema_updates = []

        def loss_fn(latents, rewards, terminals, actions, lens, shortcut_train_mode, update_ema):
            ema_updates.append(update_ema)
            zero = world_model.weight * 0.
            losses = SimpleNamespace(
                flow = zero, shortcut = zero, reward_mae = zero, agent_state_pred = zero,
                agent_embed_reward = zero, latent_state_reward = zero,
                agent_embed_terminal = zero, latent_state_terminal = zero,
                diagnostics = SimpleNamespace(
                    raw_flow = zero, raw_shortcut = zero, raw_dynamics = zero,
                    normalized_dynamics = zero, raw_reward = zero, raw_terminal = zero,
                ),
            )
            # A batch mean, as in the world-model loss.
            return (world_model.weight * rewards.sum(dim = -1)).mean(), losses

        torch.manual_seed(1)
        train_gym.train_world_model(
            world_model,
            torch.optim.SGD(world_model.parameters(), lr = 1.),
            replay,
            world_model_loss_fn = loss_fn,
            world_model_loss_timing = None,
            world_model_step_timing = None,
            steps = 2,
            batch_size = 8,
            sequence_length = 4,
            static_batch_shape = True,
            max_grad_norm = 1e6,
            global_step = 0,
            writer = None,
            training_profiler = None,
            shortcut_train_mode = "combined",
            desc = "test",
            micro_batch_size = micro_batch_size,
        )
        return world_model.weight.item(), ema_updates

    one_pass_weight, one_pass_ema_updates = update(None)
    micro_weight, micro_ema_updates = update(4)

    assert one_pass_weight < 0.
    assert micro_weight == pytest.approx(one_pass_weight, rel = 1e-6)
    # The loss EMA updates once for each step in both paths.
    assert one_pass_ema_updates == [True, True]
    assert micro_ema_updates == [True, False, True, False]

    with pytest.raises(ValueError, match = "multiple"):
        update(3)


def test_head_reanchor_restores_the_heads_after_a_transformer_change():
    torch.manual_seed(0)
    world_model = DynamicsWorldModel(
        dim = 16,
        dim_latent = 4,
        max_steps = 8,
        num_latent_tokens = 1,
        num_spatial_tokens = 1,
        depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        num_continuous_actions = 2,
        continuous_dist_type = "beta",
        reward_encoder_kwargs = dict(num_bins = 5),
        value_encoder_kwargs = dict(num_bins = 5),
        policy_head_mlp_depth = 1,
        value_head_mlp_depth = 1,
    )
    rollout = Experience(
        latents = torch.randn(3, 12, 1, 4),
        rewards = torch.randn(3, 12),
        actions = Actions(None, torch.rand(3, 12, 2).clamp(0.05, 0.95)),
        lens = torch.tensor((12, 12, 5)),
        step_size = 2,
        is_from_world_model = False,
    )

    context = train_gym.rollout_context_windows(rollout, num_windows = 6, window_length = 8, rng = np.random.default_rng(0))

    # Windows come only from episodes that are long enough, with all fields cut at the same times.
    assert context.latents.shape == (6, 8, 1, 4) and context.lens.tolist() == [8] * 6
    for window in range(6):
        match = [(e, s) for e in (0, 1) for s in range(5) if torch.equal(rollout.latents[e, s:s + 8], context.latents[window])]
        assert len(match) == 1
        episode, start = match[0]
        assert torch.equal(rollout.actions.continuous[episode, start:start + 8], context.actions.continuous[window])
        assert torch.equal(rollout.rewards[episode, start:start + 8], context.rewards[window])

    # New heads are near-constant (zero-initialized outputs). Random heads make the outputs depend on the features.
    with torch.no_grad():
        for param in [*world_model.policy_head_parameters(), *world_model.value_head_parameters()]:
            param.add_(torch.randn_like(param) * 0.5)

    reanchor = train_gym.HeadReanchor(world_model, steps = 300, learning_rate = 1e-2, step_size = 2, chunk_size = 4)
    reanchor.capture(context)

    # A world-model phase: only the shared transformer changes.
    with torch.no_grad():
        for param in world_model.transformer.parameters():
            param.add_(torch.randn_like(param) * 0.05)
    transformer = [param.detach().clone() for param in world_model.transformer.parameters()]

    metrics = reanchor.fit()

    assert metrics["reanchor/policy_kl_before"] > 1e-3
    assert metrics["reanchor/policy_kl_after"] < 0.25 * metrics["reanchor/policy_kl_before"]
    assert metrics["reanchor/value_ce_after"] < metrics["reanchor/value_ce_before"]
    assert all(torch.equal(old, new) for old, new in zip(transformer, world_model.transformer.parameters()))
    # One fit for each capture.
    assert reanchor.fit() == {}


def small_agent_world_model():
    torch.manual_seed(0)
    world_model = DynamicsWorldModel(
        dim = 16,
        dim_latent = 4,
        max_steps = 8,
        num_latent_tokens = 1,
        num_spatial_tokens = 1,
        depth = 2,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = 8,
        num_continuous_actions = 2,
        continuous_dist_type = "beta",
        reward_encoder_kwargs = dict(num_bins = 5),
        value_encoder_kwargs = dict(num_bins = 5),
        policy_head_mlp_depth = 1,
        value_head_mlp_depth = 1,
        policy_all_tokens = True,
        old_reward_head = True,
        old_pmpo = True,
        pmpo_kl_div_loss_weight = 0.3,
    )
    # New heads are near-constant (zero-initialized outputs). Random weights make all outputs depend on the features.
    with torch.no_grad():
        for param in world_model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    return world_model


def test_fixed_full_transformer_finetuning_learner_reads_the_rollout_inputs():
    world_model = small_agent_world_model()
    prompt_length, horizon, batch = 3, 5, 4
    prompts = (
        torch.rand(batch, prompt_length, 1, 4) * 2. - 1.,
        None,
        None,
        torch.rand(batch, prompt_length, 2).clamp(0.05, 0.95),
        torch.randn(batch, prompt_length - 1),
    )

    def rollout(clamp_generated_latents):
        generate = ImaginationGenerateRollout(
            world_model, generate_steps = 2, store_old_action_unembeds = False, use_time_cache = True,
            preallocate_outputs = False, clamp_generated_latents = clamp_generated_latents,
        )
        torch.manual_seed(1)
        with torch.no_grad():
            full_dream = generate(prompt_length + horizon, batch, *prompts)
            return full_dream, train_gym.dream_policy_inputs(world_model, full_dream, prompt_length)

    # One clean forward over the prompt and the dream gives the inputs that the heads read in the rollout.
    full_dream, (agent_embeds, policy_inputs) = rollout(clamp_generated_latents = False)
    assert full_dream.latents.abs().max() > 1.
    assert torch.allclose(agent_embeds, full_dream.agent_embed, atol = 1e-5)
    assert torch.allclose(policy_inputs, full_dream.policy_input, atol = 1e-5)

    # The default rollout clamps its latents after the heads read them, so the same forward gives other inputs.
    full_dream, (_, policy_inputs) = rollout(clamp_generated_latents = True)
    assert (policy_inputs - full_dream.policy_input).abs().max() > 1e-3


@pytest.mark.parametrize(("objective", "value_loss_weight"), (("pmpo", 1.), ("ppo", .5)))
def test_fixed_full_transformer_finetuning_chunks_give_the_one_pass_gradient(objective, value_loss_weight):
    world_model = small_agent_world_model()
    # Closer PPO has no policy prior.
    policy_prior = train_gym.FrozenModelPolicyPrior(world_model) if objective == "pmpo" else None
    generate = ImaginationGenerateRollout(
        world_model, generate_steps = 2, store_old_action_unembeds = False, use_time_cache = True,
        preallocate_outputs = False, clamp_generated_latents = False,
    )

    def joint_loss(micro_batch_size):
        return train_gym.JointImaginationLoss(
            world_model, policy_prior, None, objective = objective, value_loss_weight = value_loss_weight,
            generate_rollout_fn = generate, batch_size = 5, horizon = 6,
            prompt_length = 0, static_generate_shape = False, loss_weight = 1., micro_batch_size = micro_batch_size,
            use_delight_gating = False, grad_diagnostics = False, autocast_bf16 = False, global_step = 0,
            writer = None, training_profiler = None,
        )

    def gradients(micro_batch_size):
        world_model.zero_grad(set_to_none = True)
        joint = joint_loss(micro_batch_size)
        torch.manual_seed(2)
        loss = joint(0, None)
        # One pass keeps the graph for the single backward of train_world_model. Chunks leave the gradient in place.
        assert loss.requires_grad == (micro_batch_size is None)
        if loss.requires_grad:
            loss.backward()
        return {name: param.grad.clone() for name, param in world_model.named_parameters() if param.grad is not None}, joint.last_metrics

    one_pass, metrics = gradients(None)
    chunked, _ = gradients(2)

    # The RL loss reaches the transformer and both heads.
    for prefix in ("transformer.", "policy_head.", "value_head."):
        assert any(name.startswith(prefix) and grad.abs().sum() > 0. for name, grad in one_pass.items()), prefix

    assert one_pass.keys() == chunked.keys()
    for name, grad in one_pass.items():
        assert torch.allclose(grad, chunked[name], rtol = 1e-4, atol = 1e-6), name

    assert metrics["joint/learner_input_error"] < 1e-5
    if objective == "ppo":
        assert "imagination/prior_kl_mean" not in metrics
        return

    # The prior is the full model at the start of the loop: the KL is 0 until a parameter changes, and a change of the
    # transformer alone moves the policy away from the prior.
    assert metrics["imagination/prior_kl_mean"] == pytest.approx(0., abs = 1e-6)
    with torch.no_grad():
        for param in world_model.transformer.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    _, metrics = gradients(None)
    assert metrics["imagination/prior_kl_mean"] > 1e-4


def test_fixed_full_transformer_finetuning_trains_all_parameters_with_one_optimizer(tmp_path, capsys):
    from train.ablations import NEXT_STATE_REWARD_TARGET

    config = small_cpu_config(
        tmp_path, **NEXT_STATE_REWARD_TARGET,
        run_name = "joint", num_loops = 2, total_envs = 4, checkpoint_every = 1,
        fixed_full_transformer_finetuning = True, fixed_full_transformer_finetuning_steps = 2,
        world_model_micro_batch_size = 2, fixed_full_transformer_finetuning_dream_micro_batch_size = 2,
    )

    with pytest.raises(ValueError, match = "objective='pmpo'"):
        train_gym.main(**dict(config, objective = "ppo"))
    with pytest.raises(ValueError, match = "imagination_use_time_cache"):
        train_gym.main(**dict(config, imagination_use_time_cache = False, compile_generate_with_time_cache = False))
    with pytest.raises(ValueError, match = "exclude each other"):
        train_gym.main(**dict(config, full_transformer_finetuning = True))

    train_gym.main(**config)

    accumulator = EventAccumulator(str(tmp_path / "logs" / "joint"), size_guidance = {"scalars": 0}).Reload()
    scalars = {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}
    # Each joint step logs the world-model loss and the RL loss. The gradient diagnostic runs one time for each loop.
    for tag, count in (
        ("world_model/loss", 2 + 4), ("imagination/dream_return", 4), ("imagination/prior_kl_mean", 4),
        ("joint_diag/trunk_grad_norm_policy", 2), ("world_model/value_head_grad_norm", 4),
    ):
        assert len(scalars[tag]) == count, tag
    assert max(event.value for event in scalars["joint/learner_input_error"]) < 1e-4
    # The prior is the policy at the start of each loop.
    assert [abs(event.value) < 1e-6 for event in scalars["imagination/prior_kl_mean"]] == [True, False, True, False]

    # One optimizer: no agent optimizer, and the heads are the last group of the world-model optimizer.
    checkpoint = tmp_path / "logs" / "joint" / "checkpoints" / "loop_2.pt"
    package = torch.load(checkpoint, weights_only = True)
    assert package["agent_optimizer"] is None
    groups = package["world_optimizer"]["param_groups"]
    assert len(groups) == 3
    stepped = [package["world_optimizer"]["state"][index]["steps"] for index in groups[2]["params"] if index in package["world_optimizer"]["state"]]
    assert len(stepped) > 0 and min(stepped) == 4

    # A resume loads the one optimizer.
    capsys.readouterr()
    train_gym.main(**dict(config, num_loops = 3, checkpoint_path = str(checkpoint)))
    assert "skipping world optimizer state" not in capsys.readouterr().out


def test_fixed_full_transformer_finetuning_with_closer_ppo_uses_one_optimizer(tmp_path):
    from train.ablations import CLEANRL_PPO_CLOSER, NEXT_STATE_REWARD_TARGET

    config = small_cpu_config(
        tmp_path, **{**CLEANRL_PPO_CLOSER, **NEXT_STATE_REWARD_TARGET},
        run_name = "joint_ppo", num_loops = 2, total_envs = 4, imagination_horizon = 9, checkpoint_every = 1,
        fixed_full_transformer_finetuning = True, fixed_full_transformer_finetuning_steps = 2,
        world_model_micro_batch_size = 2, fixed_full_transformer_finetuning_dream_micro_batch_size = 2,
    )
    train_gym.main(**config)

    accumulator = EventAccumulator(str(tmp_path / "logs" / "joint_ppo"), size_guidance = {"scalars": 0}).Reload()
    scalars = {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}
    for tag, count in (("world_model/loss", 2 + 4), ("imagination/policy_loss", 4), ("joint_diag/trunk_grad_norm_value", 2)):
        assert len(scalars[tag]) == count, tag
    # Closer PPO has no policy prior, and its value loss is a squared error.
    assert "imagination/prior_kl_mean" not in scalars
    assert "extra/value_mse" in scalars
    assert max(event.value for event in scalars["joint/learner_input_error"]) < 1e-4

    # One optimizer: the heads are its last group, with the agent learning rate of closer PPO and its linear decay
    # (the last of the 2 loops has the factor 1 - 1 / 2).
    package = torch.load(tmp_path / "logs" / "joint_ppo" / "checkpoints" / "loop_2.pt", weights_only = True)
    assert package["agent_optimizer"] is None
    groups = package["world_optimizer"]["param_groups"]
    assert groups[-1]["lr"] == pytest.approx(config["agent_learning_rate"] * .5)
    assert groups[0]["lr"] == pytest.approx(config["world_model_learning_rate"])
    stepped = [package["world_optimizer"]["state"][index]["steps"] for index in groups[-1]["params"] if index in package["world_optimizer"]["state"]]
    assert len(stepped) > 0 and min(stepped) == 4
