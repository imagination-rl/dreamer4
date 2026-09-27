from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.nn import functional as F

from dreamer4.dreamer4 import DynamicsWorldModel, Experience
from train import ablations, train_gym
from train import offline_pretraining as offline

CONFIG_PATH = Path(__file__).resolve().parents[1] / "train" / "config.yaml"
ACTION_RANGE = (-1., 1.)


def tiny_world_model(**kwargs):
    config = dict(
        dim = 32,
        dim_latent = 8,
        max_steps = 64,
        num_latent_tokens = 2,
        num_spatial_tokens = 2,
        num_register_tokens = 1,
        dim_critic_state = 17,
        depth = 1,
        time_block_every = 1,
        attn_heads = 2,
        attn_dim_head = 16,
        num_discrete_actions = 0,
        num_continuous_actions = 6,
        continuous_dist_type = "beta",
        continuous_target_action_range = ACTION_RANGE,
        reward_encoder_kwargs = dict(reward_range = (-20., 20.), num_bins = 11),
        value_encoder_kwargs = dict(reward_range = (-100., 100.), num_bins = 11),
        predict_terminals = True,
        policy_head_mlp_depth = 1,
        value_head_mlp_depth = 1,
    )
    config.update(kwargs)
    return DynamicsWorldModel(**config)


def tiny_tokenizer():
    return train_gym.ObservationTokenizer(obs_dim = 17, num_latent_tokens = 2, dim_latent = 8, hidden_dim = 16)


def write_fake_expert(path: Path, obs_dim = 17, action_dim = 6):
    torch.manual_seed(0)
    actor = {
        "fc1.weight": torch.randn(256, obs_dim), "fc1.bias": torch.randn(256),
        "fc2.weight": torch.randn(256, 256) / 16, "fc2.bias": torch.randn(256),
        "fc_mu.weight": torch.randn(action_dim, 256) / 16, "fc_mu.bias": torch.randn(action_dim),
        "action_scale": torch.ones(1, action_dim), "action_bias": torch.zeros(1, action_dim),
    }
    qf = {"fc1.weight": torch.randn(4, obs_dim + action_dim)}
    torch.save((actor, qf, qf), path)
    return actor


def offline_episode(num_transitions: int, reward: float):
    return offline.offline_episode_experience(
        latents = torch.randn(num_transitions + 1, 2, 8),
        observations = np.random.randn(num_transitions + 1, 17).astype(np.float32),
        env_actions = np.zeros((num_transitions, 6), dtype = np.float32),
        rewards = np.full(num_transitions, reward, dtype = np.float32),
        terminated = False,
        truncated = True,
        action_range = ACTION_RANGE,
        step_size = 16,
    )


def replay_of(folder: Path, episodes: list[Experience]):
    replay = Experience.create_memmap_replay_buffer(
        episodes[0],
        folder,
        max_episodes = len(episodes),
        max_timesteps = 16,
        circular = False,
    )
    for episode in episodes:
        offline.store_experience(replay, episode)
    return replay


def test_offline_episode_has_the_online_replay_layout():
    torch.manual_seed(0)
    world_model = tiny_world_model().eval()
    tokenizer = tiny_tokenizer().eval()
    env = train_gym.make_env("HalfCheetah-v4", 0, num_envs = 1, env_vectorization_mode = "sync")

    try:
        online = world_model.interact_with_env(
            env,
            seed = 0,
            max_timesteps = 5,
            env_is_vectorized = True,
            store_agent_embed = False,
            store_old_action_unembeds = False,
            obs_to_latents_fn = train_gym.obs_to_latents_fn(tokenizer),
        ).cpu().unbind()[0]
    finally:
        env.close()

    # Rebuild the episode from what the env saw: observations, env-range actions, and rewards.
    observations = online.critic_state[0].numpy()
    env_actions = world_model.action_embedder.rescale_for_env(online.actions.continuous[0, :5]).clamp(*ACTION_RANGE)
    rebuilt = offline.offline_episode_experience(
        latents = offline.encode_observations(tokenizer, observations, torch.device("cpu")),
        observations = observations,
        env_actions = env_actions.numpy(),
        rewards = online.rewards[0, :5].numpy(),
        terminated = False,
        truncated = True,
        action_range = ACTION_RANGE,
        step_size = offline.interact_step_size(world_model),
    )

    online_fields = {**online.to_buffer_dict()[0], **online.to_buffer_dict()[1]}
    rebuilt_fields = {**rebuilt.to_buffer_dict()[0], **rebuilt.to_buffer_dict()[1]}

    assert online_fields.keys() == rebuilt_fields.keys()
    for key, value in online_fields.items():
        if torch.is_tensor(value):
            assert (value.dtype, value.shape) == (rebuilt_fields[key].dtype, rebuilt_fields[key].shape), key
        else:
            assert (type(value), value) == (type(rebuilt_fields[key]), rebuilt_fields[key]), key

    # Row t pairs latent t with action t and reward t. The last row is the zero-padded bootstrap row.
    assert rebuilt.lens.tolist() == online.lens.tolist() == [6]
    assert torch.allclose(rebuilt.latents, online.latents, atol = 1e-5)
    assert torch.equal(rebuilt.critic_state, online.critic_state)
    assert torch.equal(rebuilt.rewards, online.rewards)
    assert torch.allclose(
        rebuilt.actions.continuous[:, :5],
        online.actions.continuous[:, :5].clamp(offline.NATIVE_ACTION_EPS, 1. - offline.NATIVE_ACTION_EPS),
        atol = 1e-5,
    )
    assert torch.equal(rebuilt.actions.continuous[0, -1], torch.zeros(6))
    assert (rebuilt.is_truncated.item(), rebuilt.terminals.item()) == (True, False)
    assert torch.allclose(rebuilt.episode_return, online.episode_return)


def test_offline_actions_round_trip_through_rescale_for_env():
    world_model = tiny_world_model()
    env_actions = np.random.default_rng(0).uniform(-1., 1., size = (7, 6)).astype(np.float32)
    env_actions[0], env_actions[1] = -1., 1.

    episode = offline.offline_episode_experience(
        latents = torch.zeros(8, 2, 8),
        observations = np.zeros((8, 17), dtype = np.float32),
        env_actions = env_actions,
        rewards = np.zeros(7, dtype = np.float32),
        terminated = False,
        truncated = True,
        action_range = ACTION_RANGE,
        step_size = 16,
    )
    native = episode.actions.continuous[0, :7]

    assert ((native > 0.) & (native < 1.)).all()
    # The endpoint actions move by the clamp margin, 2e-4 in env units.
    assert torch.allclose(world_model.action_embedder.rescale_for_env(native), torch.from_numpy(env_actions), atol = 3e-4)


def test_expert_loader_reproduces_the_cleanrl_actor(tmp_path):
    state = write_fake_expert(tmp_path / "expert.cleanrl_model")

    path, digest = offline.download_expert(
        repo = "unused",
        revision = "unused",
        filename = "unused",
        cache_dir = tmp_path / "cache",
        expert_path = str(tmp_path / "expert.cleanrl_model"),
    )
    actor = offline.load_expert_actor(path, obs_dim = 17, action_dim = 6)

    obs = torch.randn(5, 17)
    hidden = F.relu(F.linear(F.relu(F.linear(obs, state["fc1.weight"], state["fc1.bias"])), state["fc2.weight"], state["fc2.bias"]))
    expected = torch.tanh(F.linear(hidden, state["fc_mu.weight"], state["fc_mu.bias"])) * state["action_scale"] + state["action_bias"]

    assert digest == offline.sha256_file(path)
    assert not (tmp_path / "cache").exists()
    assert torch.allclose(actor(obs), expected, atol = 1e-6)

    with pytest.raises(ValueError, match = "obs dim"):
        offline.load_expert_actor(path, obs_dim = 11, action_dim = 6)


def test_pretrained_world_model_loads_into_a_different_value_head():
    torch.manual_seed(0)
    pretrained = tiny_world_model()
    torch.manual_seed(1)
    target = tiny_world_model(
        use_symexp_value_head = True,
        use_hl_gauss_actor_critic = True,
        value_encoder_kwargs = dict(reward_range = (-10., 10.), num_bins = 7),
    )

    agent_keys = offline.agent_state_keys(target)
    target_agent_state = {key: value.clone() for key, value in target.state_dict().items() if key in agent_keys}
    saved = offline.world_model_state_without_agent_keys(pretrained)

    # A strict load fails, because the value heads differ in shape and in keys.
    with pytest.raises(RuntimeError):
        deepcopy(target).load_state_dict(pretrained.state_dict())

    offline.load_world_model_without_agent_keys(target, saved)
    target_state = target.state_dict()

    assert any(key.startswith("actor_critic_value_head.") for key in agent_keys)
    assert all(torch.equal(target_state[key], value) for key, value in saved.items())
    assert all(torch.equal(target_state[key], value) for key, value in target_agent_state.items())

    with pytest.raises(ValueError, match = "unexpected keys"):
        offline.load_world_model_without_agent_keys(target, {**saved, "not_a_key": torch.zeros(1)})


def test_pretrained_optimizer_state_keeps_the_configured_learning_rate():
    # Optimizer.load_state_dict() also restores the saved lr. The signature does not
    # include world_model_learning_rate, so the rl stage must keep its own value.
    model = torch.nn.Linear(4, 4)
    pretrain_optimizer = train_gym.make_optimizer(
        list(model.parameters()), lr = 1e-3, weight_decay = 0., use_muon = True, muon_params = [model.weight],
    )
    model(torch.ones(1, 4)).sum().backward()
    pretrain_optimizer.step()
    state = train_gym.optimizer_state_dict_for_checkpoint(pretrain_optimizer)

    optimizer = train_gym.make_optimizer(
        list(model.parameters()), lr = 5e-4, weight_decay = 0.01, use_muon = True, muon_params = [model.weight],
    )

    assert offline.load_optimizer_moments(optimizer, state, "test")
    assert len(optimizer.state) == len(pretrain_optimizer.state)
    assert [(group["lr"], group["weight_decay"]) for group in optimizer.param_groups] == [(5e-4, 0.01)] * 2


def test_pretrain_signature_rejects_a_different_reward_target():
    config = dict(yaml.safe_load(CONFIG_PATH.read_text()), obs_dim = 17)
    saved = offline.pretrain_signature(config)

    # YAML lists and Python tuples compare equal.
    offline.check_pretrain_signature(saved, offline.pretrain_signature(dict(config, reward_range = (-20., 20.))), source = "test")

    with pytest.raises(ValueError, match = "old_reward_shifting"):
        offline.check_pretrain_signature(
            saved,
            offline.pretrain_signature(dict(config, old_reward_shifting = not config["old_reward_shifting"])),
            source = "test",
        )


def test_rehearsal_replaces_part_of_each_world_model_batch(tmp_path):
    assert offline.rehearsal_batch_sizes(128, 0.25) == (96, 32)

    online = replay_of(tmp_path / "online", [offline_episode(5, reward = 1.) for _ in range(3)])
    rehearsal = replay_of(tmp_path / "offline", [offline_episode(5, reward = 2.) for _ in range(2)])

    class FakeWorldModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        @property
        def device(self):
            return self.weight.device

    world_model = FakeWorldModel()
    batch_rewards = []

    def loss_fn(latents, rewards, *_args):
        batch_rewards.append(rewards[:, 0].tolist())
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
        return zero, losses

    train_gym.train_world_model(
        world_model,
        torch.optim.AdamW(world_model.parameters()),
        online,
        world_model_loss_fn = loss_fn,
        world_model_loss_timing = None,
        world_model_step_timing = None,
        steps = 3,
        batch_size = 8,
        sequence_length = 4,
        static_batch_shape = True,
        max_grad_norm = 1.,
        global_step = 0,
        writer = None,
        training_profiler = None,
        shortcut_train_mode = "combined",
        desc = "test",
        rehearsal_replay = rehearsal,
        rehearsal_fraction = 0.25,
    )

    assert batch_rewards == [[1.] * 6 + [2.] * 2] * 3

    with pytest.raises(ValueError, match = "different keys"):
        offline.concat_replay_batches({"a": torch.zeros(1)}, {"b": torch.zeros(1)})


def test_offline_ablation_arms_agree_with_their_pretrain_signature():
    specs = {
        spec.train_kwargs["run_name"]: spec.train_kwargs
        for spec in ablations.build_ablation_specs(ablation_indices = [
            index
            for index, ablation in enumerate(ablations.ABLATIONS, 1)
            if ablation["run_name"].startswith("offline_pretrain")
        ])
    }
    pretrain = specs.pop("offline_pretrain_a2_next_state_reward")
    checkpoint = Path(pretrain["offline_pretrain_dir"]) / pretrain["offline_pretrain_name"] / offline.PRETRAINED_FILE

    def signature(kwargs):
        return offline.pretrain_signature(dict(kwargs, obs_dim = 17))

    assert len(specs) == 4
    for kwargs in specs.values():
        assert Path(kwargs["offline_pretrained_checkpoint"]) == checkpoint
        assert signature(kwargs) == signature(pretrain)


def test_main_runs_the_three_offline_stages_on_cpu(tmp_path):
    write_fake_expert(tmp_path / "expert.cleanrl_model")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config.update(
        log_dir = str(tmp_path / "logs"),
        run_name = "offline",
        run_details = "offline integration test",
        cpu = True,
        require_cuda = False,
        compile = False,
        compile_interact = False,
        compile_world_model = False,
        compile_generate = False,
        compile_learn = False,
        num_loops = 1,
        num_envs = 2,
        env_vectorization_mode = "sync",
        max_timesteps = 48,
        replay_size = 2,
        model_dim = 32,
        attn_heads = 2,
        attn_dim_head = 16,
        pretrain_tokenizer_steps = 2,
        tokenizer_batch_size = 32,
        world_model_train_steps = 1,
        world_model_batch_size = 4,
        world_model_train_sequence_length = 16,
        imagination_batch_size = 4,
        imagination_horizon = 8,
        imagination_prompt_length = 4,
        imagination_train_steps = 1,
        preservation_sequence_length = 16,
        checkpoint_every = 0,
        tokenizer_eval_every = 0,
        collect_timings = False,
        offline_pretraining = True,
        offline_expert_path = str(tmp_path / "expert.cleanrl_model"),
        offline_expert_min_return = -1e9,
        offline_data_dir = str(tmp_path / "data"),
        offline_pretrain_dir = str(tmp_path / "pretrain"),
        offline_data_steps = 4000,
        offline_data_mix = {"expert": 0.25, "expert_noise": 0.5, "random": 0.25},
        offline_data_num_envs = 4,
        offline_validation_fraction = 0.5,
        offline_world_model_steps = 2,
        offline_pretrain_checkpoint_every = 1,
        offline_rehearsal_fraction = 0.5,
    )

    train_gym.main(**config)

    dataset = offline.load_offline_dataset(tmp_path / "data" / "HalfCheetah-v4" / config["offline_dataset_name"])
    pretrained = tmp_path / "pretrain" / "offline" / offline.PRETRAINED_FILE
    accumulator = EventAccumulator(str(tmp_path / "logs" / "offline")).Reload()
    scalars = accumulator.Tags()["scalars"]

    assert dataset.env_steps == 4000
    assert sorted(dataset.manifest["episodes_per_mode"].values()) == [1, 1, 2]
    assert (tmp_path / "pretrain" / "offline" / "pretrained_step_1.pt").is_file()
    assert "offline_pretrain/val/world_model/loss" in scalars
    assert "offline_pretrain/train/world_model/loss" in scalars
    assert accumulator.Scalars("charts/episodic_return_offline_inclusive")[0].step == 4000 + 2 * 48

    # A second run loads the checkpoint from disk and does not collect again.
    train_gym.main(**dict(
        config,
        run_name = "rl_only",
        offline_stages = ["rl"],
        offline_pretrained_checkpoint = str(pretrained),
    ))

    accumulator = EventAccumulator(str(tmp_path / "logs" / "rl_only")).Reload()
    hyperparameters = json.loads(
        accumulator.Tensors("run/hyperparameters/text_summary")[0]
        .tensor_proto.string_val[0].decode()
        .strip("`").replace("json", "", 1)
    )

    assert hyperparameters["offline_env_steps"] == 4000
    assert hyperparameters["offline_effective_overrides"]["pretrain_world_model_combined_steps"] == 0
    assert "world_model/loss" in accumulator.Tags()["scalars"]

    # A pretrained checkpoint for another reward target fails before any setup.
    with pytest.raises(ValueError, match = "old_reward_shifting"):
        train_gym.main(**dict(
            config,
            run_name = "wrong_target",
            offline_stages = ["rl"],
            offline_pretrained_checkpoint = str(pretrained),
            old_reward_shifting = not config["old_reward_shifting"],
        ))
