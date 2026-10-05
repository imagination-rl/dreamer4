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


def online_run_replay(
    folder: Path,
    episodes: list[tuple[int, bool]],
    seed: int,
    returns: list[float] | None = None,
    flip_steps: list[int | None] | None = None,
):
    """A replay of an online run: (transitions, terminated) for each episode, with random actions and rewards.

    With returns, each episode gets a constant reward per step with that return.
    With flip_steps, the root angle is 0 (upright), and an episode with a step t is flipped (angle pi) from row t.
    """

    rng = np.random.default_rng(seed)

    def observations(index, length):
        obs = rng.standard_normal((length + 1, 17)).astype(np.float32)
        if flip_steps is not None:
            obs[:, offline.HALF_CHEETAH_ROOT_ANGLE_INDEX] = 0.
            if flip_steps[index] is not None:
                obs[flip_steps[index]:, offline.HALF_CHEETAH_ROOT_ANGLE_INDEX] = np.pi
        return obs

    experiences = [
        offline.offline_episode_experience(
            latents = torch.zeros(length + 1, 2, 8),
            observations = observations(index, length),
            env_actions = rng.uniform(-0.9, 0.9, size = (length, 6)).astype(np.float32),
            rewards = (
                rng.standard_normal(length) if returns is None else np.full(length, returns[index] / length)
            ).astype(np.float32),
            terminated = terminated,
            truncated = not terminated,
            action_range = ACTION_RANGE,
            step_size = 16,
        )
        for index, (length, terminated) in enumerate(episodes)
    ]
    replay = Experience.create_memmap_replay_buffer(
        experiences[0], folder, max_episodes = len(experiences), max_timesteps = 1001, circular = False,
    )
    for experience in experiences:
        offline.store_experience(replay, experience)
    return replay


def test_past_runs_episodes_copy_full_episodes_of_online_replays(tmp_path):
    # Each folder: two full 1000-step episodes, one short episode, and one terminated episode.
    folders = [tmp_path / "run_a", tmp_path / "run_b"]
    replays = [
        online_run_replay(folder, [(1000, False), (400, False), (1000, False), (1000, True)], seed = index)
        for index, folder in enumerate(folders)
    ]
    kwargs = dict(
        env_name = "HalfCheetah-v4", obs_dim = 17, action_dim = 6, action_range = ACTION_RANGE,
        data_dir = tmp_path / "data", data_steps = 4000, data_mix = dict(random = 0.25, past_runs = 0.75),
        noise_stds = [0.1], data_seed = 3, num_envs = 1, expert_repo = "unused", expert_revision = "unused",
        expert_filename = "unused", expert_cache_dir = tmp_path / "experts", expert_path = None, expert_min_return = 0.,
        past_run_replays = [str(folder) for folder in folders],
    )
    # The queues delete the latents of finished runs. The collector must not create a new (zero) latents file.
    (folders[1] / "latents.data.npy").unlink()

    dataset = offline.load_offline_dataset(offline.collect_offline_dataset(dataset_name = "past", **kwargs))
    assert not (folders[1] / "latents.data.npy").exists()
    past = np.flatnonzero(dataset.mode == offline.OFFLINE_MODE_IDS["past_runs"])
    source = np.load(dataset.path / offline.PAST_RUN_SOURCE_FILE)

    # Three past_runs episodes: two from the first folder, one from the second, only full truncated episodes.
    assert len(past) == 3 and (dataset.mode == offline.OFFLINE_MODE_IDS["random"]).sum() == 1
    assert sorted(source[past, 0].tolist()) == [0, 0, 1]
    assert set(source[past, 1].tolist()) <= {0, 2}
    assert (source[dataset.mode != offline.OFFLINE_MODE_IDS["past_runs"]] == -1).all()
    assert [entry["full_episodes"] for entry in dataset.manifest["past_runs"]] == [2, 2]

    # The dataset rows are the replay rows: obs_t, the env-range action a_t, and r_t, then the final observation.
    for index in past:
        replay = replays[source[index, 0]]
        episode = source[index, 1]
        assert np.array_equal(dataset.obs[index], replay.data["critic_state"][episode, :1001])
        assert np.array_equal(dataset.rewards[index], replay.data["rewards"][episode, :1000])
        native = offline.env_to_native_actions(dataset.actions[index], ACTION_RANGE)
        assert np.allclose(native, replay.data["actions_continuous"][episode, :1000], atol = 1e-5)
        assert (dataset.lens[index], dataset.truncated[index], dataset.terminated[index]) == (1000, True, False)

    # The same config skips the collection. Another replay list is a different dataset.
    assert offline.collect_offline_dataset(dataset_name = "past", **kwargs) == dataset.path
    with pytest.raises(ValueError, match = "past_run_replays"):
        offline.collect_offline_dataset(dataset_name = "past", **{**kwargs, "past_run_replays": [str(folders[0])]})


def test_past_runs_return_buckets_split_the_episodes_equally(tmp_path):
    # Run a: three poor episodes and one good episode. Run b: three good, one too good, one random-like.
    runs = [([100., 200., 300., 900.], 4), ([800., 900., 1000., 2000., -300.], 5)]
    folders = [tmp_path / f"run_{index}" for index in range(len(runs))]
    for index, (folder, (returns, num)) in enumerate(zip(folders, runs)):
        online_run_replay(folder, [(15, False)] * num, seed = index, returns = returns)

    arrays, source, summaries = offline.past_run_episodes(
        folders, 4, rng = np.random.default_rng(0), max_episode_steps = 15, obs_dim = 17, action_dim = 6,
        action_range = ACTION_RANGE, return_buckets = [(-150., 500.), (500., 1500.)],
    )
    returns = arrays["rewards"].sum(-1)

    # Two episodes in each bucket. Only run a has poor episodes; the good bucket takes one from each run.
    assert np.sum(returns < 500.) == 2 and np.sum((returns >= 500.) & (returns < 1500.)) == 2
    assert sorted(source[:, 0].tolist()) == [0, 0, 0, 1]
    assert [summary["episodes_per_bucket"] for summary in summaries] == [
        {"[-150, 500)": 2, "[500, 1500)": 1}, {"[-150, 500)": 0, "[500, 1500)": 1},
    ]
    with pytest.raises(ValueError, match = "overlap"):
        offline.past_run_episodes(
            folders, 2, rng = np.random.default_rng(0), max_episode_steps = 15, obs_dim = 17, action_dim = 6,
            action_range = ACTION_RANGE, return_buckets = [(0., 600.), (500., 1500.)],
        )
    with pytest.raises(ValueError, match = r"\[-150, 500\)"):
        offline.past_run_episodes(
            folders, 8, rng = np.random.default_rng(0), max_episode_steps = 15, obs_dim = 17, action_dim = 6,
            action_range = ACTION_RANGE, return_buckets = [(-150., 500.), (500., 1500.)],
        )


def test_past_runs_flip_share_takes_episodes_with_a_flip_first(tmp_path):
    # One bucket. Run a: two episodes that flip (rows 5 and 9) and four upright ones. Run b: four upright ones.
    folders = [tmp_path / "run_a", tmp_path / "run_b"]
    online_run_replay(folders[0], [(15, False)] * 6, seed = 0, returns = [100.] * 6, flip_steps = [5, None, 9, None, None, None])
    online_run_replay(folders[1], [(15, False)] * 4, seed = 1, returns = [200.] * 4, flip_steps = [None] * 4)

    def select(count, fraction):
        arrays, source, summaries = offline.past_run_episodes(
            folders, count, rng = np.random.default_rng(0), max_episode_steps = 15, obs_dim = 17, action_dim = 6,
            action_range = ACTION_RANGE, return_buckets = [(0., 500.)], flip_episode_fraction = fraction,
        )
        angle = np.abs(arrays["obs"][:, :, offline.HALF_CHEETAH_ROOT_ANGLE_INDEX])
        return int((angle > np.pi / 2).any(axis = 1).sum()), source, summaries

    # Half of 4: both flip episodes (run a), then 2 others, one from each run.
    flips, source, summaries = select(4, 0.5)
    assert flips == 2
    assert sorted(source[:, 0].tolist()) == [0, 0, 0, 1]
    assert [summary["flip_share_episodes_per_bucket"] for summary in summaries] == [{"[0, 500)": 2}, {"[0, 500)": 0}]

    # 0.75 of 8 asks for 6 flip episodes; only 2 exist, so the other episodes fill the bucket.
    flips, source, _ = select(8, 0.75)
    assert flips == 2 and len(source) == 8

    with pytest.raises(ValueError, match = "return buckets"):
        offline.past_run_episodes(
            folders, 2, rng = np.random.default_rng(0), max_episode_steps = 15, obs_dim = 17, action_dim = 6,
            action_range = ACTION_RANGE, flip_episode_fraction = 0.5,
        )


def test_balanced_counts_gives_the_share_of_a_full_entry_to_the_others():
    assert offline.balanced_counts(7, np.array([1, 5, 5])).tolist() == [1, 3, 3]
    assert offline.balanced_counts(4, np.array([0, 9, 9, 9])).tolist() == [0, 2, 1, 1]
    with pytest.raises(ValueError):
        offline.balanced_counts(4, np.array([1, 2]))


def test_datasets_without_past_runs_keep_their_collection_config():
    # The manifests of the datasets collected before the past_runs mode must still agree with their configs.
    config = yaml.safe_load(CONFIG_PATH.read_text())
    collect = lambda **overrides: offline.collection_config(**{**dict(
        env_name = "HalfCheetah-v4", dataset_name = config["offline_dataset_name"],
        data_steps = config["offline_data_steps"], data_mix = config["offline_data_mix"],
        noise_stds = config["offline_expert_noise_stds"], data_seed = config["offline_data_seed"],
        num_envs = config["offline_data_num_envs"], expert_repo = config["offline_expert_repo"],
        expert_revision = config["offline_expert_revision"], expert_filename = config["offline_expert_filename"],
        expert_path = None,
    ), **overrides})

    before = collect()
    assert "past_run_replays" not in before and set(before["data_mix"]) == {"expert", "expert_noise", "random"}
    # A replay list without past_runs episodes does not change the config.
    assert collect(
        past_run_replays = ["runs/some_run/replay_buffer"], past_run_return_buckets = [(0., 1.)], expert_action_scales = [0.5],
    ) == before
    assert collect(data_mix = {**config["offline_data_mix"], "past_runs": 0.}) == before


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


def test_expert_scaled_episodes_scale_the_expert_and_keep_the_other_episodes(tmp_path):
    write_fake_expert(tmp_path / "expert.cleanrl_model")
    actor = offline.load_expert_actor(tmp_path / "expert.cleanrl_model", obs_dim = 17, action_dim = 6)
    ids = offline.OFFLINE_MODE_IDS

    def run(first_mode):
        env = train_gym.make_env("HalfCheetah-v4", 0, num_envs = 2, env_vectorization_mode = "sync")
        try:
            return offline.run_episode_batch(
                env, actor, modes = np.array([first_mode, ids["random"]], dtype = np.int8),
                noise_stds = np.zeros(2, dtype = np.float32), rng = np.random.default_rng(0), seed = 0,
                action_range = ACTION_RANGE, max_episode_steps = 5, action_dim = 6,
                action_scales = np.array([0.5, 1.], dtype = np.float32),
            )
        finally:
            env.close()

    expert, scaled = run(ids["expert"]), run(ids["expert_scaled"])

    # The first action is half the expert action (the same start state). The random episode does not change.
    assert np.allclose(scaled["actions"][0, 0], 0.5 * expert["actions"][0, 0], atol = 1e-6)
    assert not np.allclose(scaled["actions"][0, 0], expert["actions"][0, 0])
    for key in ("obs", "actions", "rewards"):
        assert np.array_equal(scaled[key][1], expert[key][1]), key


def test_expert_scaled_scale_changes_every_segment_and_can_add_noise(tmp_path):
    write_fake_expert(tmp_path / "expert.cleanrl_model")
    actor = offline.load_expert_actor(tmp_path / "expert.cleanrl_model", obs_dim = 17, action_dim = 6)
    ids = offline.OFFLINE_MODE_IDS

    def run(noise_std):
        env = train_gym.make_env("HalfCheetah-v4", 0, num_envs = 2, env_vectorization_mode = "sync")
        try:
            return offline.run_episode_batch(
                env, actor, modes = np.array([ids["expert_scaled"], ids["random"]], dtype = np.int8),
                noise_stds = np.array([noise_std, 0.], dtype = np.float32), rng = np.random.default_rng(0), seed = 0,
                action_range = ACTION_RANGE, max_episode_steps = 6, action_dim = 6,
                action_scales = np.array([0.5, 1.], dtype = np.float32),
                scale_change_steps = 2, scale_range = (0.3, 0.8), scale_rng = np.random.default_rng(1),
            )
        finally:
            env.close()

    plain = run(0.)
    # The scale of each step: the recorded action over the expert action of the recorded state.
    expert = np.stack([offline.expert_actions(actor, plain["obs"][0, t][None], 6)[0] for t in range(6)])
    scales = (plain["actions"][0] / expert).mean(-1)
    assert np.allclose(plain["actions"][0], scales[:, None] * expert, atol = 1e-5)
    assert np.isclose(scales[0], 0.5, atol = 1e-5) and np.isclose(scales[0], scales[1], atol = 1e-5)
    assert np.isclose(scales[2], scales[3], atol = 1e-5) and np.isclose(scales[4], scales[5], atol = 1e-5)
    assert not np.isclose(scales[2], scales[0]) and ((scales[2:] >= 0.3) & (scales[2:] <= 0.8)).all()

    # Noise changes the first action of the scaled env; the random env does not change.
    noisy = run(0.1)
    assert not np.allclose(noisy["actions"][0, 0], plain["actions"][0, 0])
    assert np.array_equal(noisy["actions"][1], plain["actions"][1])


def test_bang_bang_episodes_hold_bound_actions_and_keep_the_other_episodes(tmp_path):
    write_fake_expert(tmp_path / "expert.cleanrl_model")
    actor = offline.load_expert_actor(tmp_path / "expert.cleanrl_model", obs_dim = 17, action_dim = 6)
    ids = offline.OFFLINE_MODE_IDS
    steps = 200

    def run(first_mode, hold):
        env = train_gym.make_env("HalfCheetah-v4", 0, num_envs = 2, env_vectorization_mode = "sync")
        try:
            return offline.run_episode_batch(
                env, actor, modes = np.array([first_mode, ids["random"]], dtype = np.int8),
                noise_stds = np.zeros(2, dtype = np.float32), rng = np.random.default_rng(0), seed = 0,
                action_range = ACTION_RANGE, max_episode_steps = steps, action_dim = 6,
                hold_steps = np.array([hold, 1.], dtype = np.float32), bang_rng = np.random.default_rng(1),
            )
        finally:
            env.close()

    expert, short, long = run(ids["expert"], 1.), run(ids["bang_bang"], 1.), run(ids["bang_bang"], 16.)

    # Only bound actions. A longer hold changes the sign less often (about 1 / hold of the steps).
    for out in (short, long):
        assert np.isin(out["actions"][0], ACTION_RANGE).all()
    changes = lambda out: (np.diff(out["actions"][0], axis = 0) != 0.).mean()
    assert 0.35 < changes(short) < 0.65 and 0.02 < changes(long) < 0.12

    # The random episode does not change: bang_bang draws come from their own generator.
    for key in ("obs", "actions", "rewards"):
        assert np.array_equal(short[key][1], expert[key][1]), key


def test_datasets_without_bang_bang_keep_their_collection_config():
    kwargs = dict(
        env_name = "HalfCheetah-v4", dataset_name = "d", data_steps = 1000, noise_stds = [0.2], data_seed = 0,
        num_envs = 1, expert_repo = "r", expert_revision = "v", expert_filename = "f", expert_path = None,
    )
    plain = offline.collection_config(data_mix = {"expert_noise": 1.}, **kwargs)
    assert "bang_bang_hold_steps" not in plain and "bang_bang" not in plain["data_mix"]
    assert offline.collection_config(data_mix = {"expert_noise": 1.}, bang_bang_hold_steps = [4.], **kwargs) == plain

    mixed = offline.collection_config(data_mix = {"expert_noise": 0.5, "bang_bang": 0.5}, bang_bang_hold_steps = [1, 4], **kwargs)
    assert mixed["bang_bang_hold_steps"] == [1., 4.] and mixed["data_mix"]["bang_bang"] == 0.5


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
    pretrains = {
        Path(kwargs["offline_pretrain_dir"]) / kwargs["offline_pretrain_name"]: kwargs
        for kwargs in specs.values()
        if kwargs["offline_stages"] == ["pretrain"]
    }
    rl_arms = [kwargs for kwargs in specs.values() if kwargs["offline_stages"] == ["rl"]]

    def signature(kwargs):
        return offline.pretrain_signature(dict(kwargs, obs_dim = 17))

    # Each RL arm loads a checkpoint of a pretrain entry with the same signature.
    assert pretrains and rl_arms
    for kwargs in rl_arms:
        pretrain = pretrains[Path(kwargs["offline_pretrained_checkpoint"]).parent]
        assert signature(kwargs) == signature(pretrain)


def test_main_runs_the_three_offline_stages_on_cpu(tmp_path, monkeypatch, capsys):
    # Relative output paths with log_storage_dir: each new directory lives in storage/ with a symlink.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "storage").mkdir()
    write_fake_expert(tmp_path / "expert.cleanrl_model")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config.update(
        log_dir = "logs",
        log_storage_dir = str(tmp_path / "storage"),
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
        total_envs = 2,
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
        offline_data_dir = "data",
        offline_pretrain_dir = "pretrain",
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

    for normal_path in ("data/HalfCheetah-v4/" + config["offline_dataset_name"], "pretrain/offline", "logs/offline"):
        assert Path(normal_path).is_symlink()
        assert Path(normal_path).resolve() == (tmp_path / "storage" / normal_path).resolve()

    # A pretrain-only run logs its world-model training to the normal graphs.
    train_gym.main(**dict(config, run_name = "pretrain_only", offline_stages = ["pretrain"], offline_pretrain_name = "pretrain_only"))
    pretrain_scalars = EventAccumulator(str(tmp_path / "logs" / "pretrain_only")).Reload().Tags()["scalars"]
    assert "world_model/loss" in pretrain_scalars and "extra/flow_loss_raw" in pretrain_scalars
    assert not any(tag.startswith("offline_pretrain/train/") for tag in pretrain_scalars)
    assert "offline_pretrain/val/world_model/loss" in pretrain_scalars

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

    # With the frozen world model, only the policy and value heads change.
    train_gym.main(**dict(
        config,
        run_name = "rl_frozen",
        offline_stages = ["rl"],
        offline_pretrained_checkpoint = str(pretrained),
        offline_rehearsal_fraction = 0.,
        freeze_world_model = True,
        num_loops = 2,
        checkpoint_every = 1,
    ))

    def loop_world_model(loop):
        path = tmp_path / "logs" / "rl_frozen" / "checkpoints" / f"loop_{loop}.pt"
        return torch.load(path, map_location = "cpu", weights_only = True)["world_model"]

    # The pretrained package has no agent keys. Its keys keep their values; the agent heads train.
    pretrained_world_model = torch.load(pretrained, map_location = "cpu", weights_only = True)["world_model"]
    loop_1, loop_2 = loop_world_model(1), loop_world_model(2)
    assert all(torch.equal(value, loop_2[key]) for key, value in pretrained_world_model.items())
    changed = {key for key, value in loop_1.items() if not torch.equal(value, loop_2[key])}
    assert changed and changed.isdisjoint(pretrained_world_model)

    # With BC in the world-model updates, the pretrained optimizer moments load, then the policy-head group follows.
    capsys.readouterr()
    train_gym.main(**dict(
        config,
        run_name = "rl_bc",
        offline_stages = ["rl"],
        offline_pretrained_checkpoint = str(pretrained),
        offline_rehearsal_fraction = 0.,
        behavior_cloning_in_world_model_updates = True,
        checkpoint_every = 1,
    ))
    assert "skipping offline pretrained world optimizer state" not in capsys.readouterr().out
    # The pretrained package has a BC weight of 0 in its buffers; the configured weight must replace it.
    bc_accumulator = EventAccumulator(str(tmp_path / "logs" / "rl_bc")).Reload()
    assert min(event.value for event in bc_accumulator.Scalars("world_model/policy_head_grad_norm")) > 0
    package = torch.load(tmp_path / "logs" / "rl_bc" / "checkpoints" / "loop_1.pt", map_location = "cpu", weights_only = True)
    assert package["world_model"]["continuous_action_loss_weight"].item() == 1.
    world_optimizer = package["world_optimizer"]
    groups, state = world_optimizer["param_groups"], world_optimizer["state"]
    assert len(groups) == 3
    # The world-model moments continue from the pretrain steps; the policy-head moments start in this run.
    def group_steps(group):
        return max(state[index]["steps"] for index in group["params"] if index in state)

    assert group_steps(groups[0]) > group_steps(groups[2]) > 0

    with pytest.raises(ValueError, match = "rehearsal"):
        train_gym.main(**dict(
            config,
            run_name = "rl_bc_rehearsal",
            offline_stages = ["rl"],
            offline_pretrained_checkpoint = str(pretrained),
            behavior_cloning_in_world_model_updates = True,
        ))

    # A pretrained checkpoint for another reward target fails before any setup.
    with pytest.raises(ValueError, match = "old_reward_shifting"):
        train_gym.main(**dict(
            config,
            run_name = "wrong_target",
            offline_stages = ["rl"],
            offline_pretrained_checkpoint = str(pretrained),
            old_reward_shifting = not config["old_reward_shifting"],
        ))
