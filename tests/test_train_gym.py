import ast
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

import train.train_gym as train_gym
from dreamer4.dreamer4 import DynamicsWorldModel
from train.compile_runtime import ImaginationGenerateRollout
from train.train_gym import RolloutTimingWrapper, pop_rollout_env_timings, repeat_batch_to_size


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_GYM_PATH = REPO_ROOT / "train" / "train_gym.py"
CONFIG_PATH = REPO_ROOT / "train" / "config.yaml"


def get_function(name):
    module = ast.parse(TRAIN_GYM_PATH.read_text())

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node

    raise AssertionError(f"{name}() not found")


def keyword_map(call):
    return {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None}


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


def test_old_replay_sampling_keeps_partial_world_model_batches_and_replaces_prompts():
    world_model_source = ast.unparse(get_function("train_world_model"))
    imagination_source = ast.unparse(get_function("train_agent_in_imagination"))

    assert "static_batch_shape and (not old_replay_sampling)" in world_model_source
    assert "RandomSampler(prompt_dataset, replacement=True, num_samples=batch_size)" in imagination_source
    assert "resampling_dataloader_iterator(dataloader)" in world_model_source
    assert "resampling_dataloader_iterator(prompt_dataloader)" in imagination_source


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


def test_main_parameters_match_config_exactly():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    main = get_function("main")
    parameters = {argument.arg for argument in main.args.kwonlyargs}

    assert parameters == set(config)
    assert {
        "collect_timings",
        "profile",
        "profile_memory",
        "profile_record_shapes",
        "profile_with_stack",
        "profile_with_flops",
    } <= parameters


def test_config_matches_modded_better_baseline():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["seed"] == 42
    assert config["old_pmpo"] is True
    assert config["old_reward_head"] is True

    for flag in (
        "add_action_embed_to_agent_token",
        "old_continuous_action_encoder",
        "old_reward_shifting",
        "old_terminal_head",
        "old_state_tokenizer",
        "old_dynamics_ff",
        "old_policy_head",
        "old_value_head",
        "old_replay_sampling",
        "old_embedding_init",
    ):
        assert config[flag] is False

    assert config["normalize_advantages"] is False
    assert config["policy_all_tokens"] is True
    assert config["value_range"] == [-1200., 1200.]
    assert config["value_encoder_num_bins"] == 255
    assert config["reward_encoder_num_bins"] == 255
    assert config["pretrain_tokenizer_steps"] == 2000
    assert config["pretrain_tokenizer_observations"] == 16384
    assert config["value_use_symlog"] is False
    assert config["spatial_flex_attention_backend"] == "AUTO"
    assert config["num_envs"] == 64
    assert config["max_timesteps"] == 1000
    assert config["pretrain_world_model_finetune_steps"] == 0
    assert config["pretrain_world_model_combined_steps"] == 256
    assert config["allow_tf32"] is True
    assert config["optimize_generate_compilation"] is False
    assert config["compile_generate_with_time_cache"] is True
    assert config["activation_memory_budget"] == 0.9
    assert config["imagination_use_time_cache"] is True
    assert config["compile_mode"] == "reduce-overhead"
    assert all(config[key] is True for key in (
        "compile_interact",
        "compile_world_model",
        "compile_generate",
        "compile_learn",
    ))


def test_main_forwards_spatial_flex_attention_backend():
    main = get_function("main")
    world_model_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DynamicsWorldModel"
    )
    keywords = keyword_map(world_model_call)

    assert ast.unparse(keywords["transformer_kwargs"]) == (
        "dict(spatial_flex_attention_backend=spatial_flex_attention_backend)"
    )


def test_main_forwards_old_continuous_action_encoder():
    main = get_function("main")
    world_model_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DynamicsWorldModel"
    )
    keywords = keyword_map(world_model_call)

    assert ast.unparse(keywords["old_continuous_action_encoder"]) == "old_continuous_action_encoder"


def test_world_model_training_logs_reward_mae():
    train_world_model = get_function("train_world_model")
    metric_values = {
        key.value: ast.unparse(value)
        for node in ast.walk(train_world_model)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }

    assert metric_values["world_model/reward_mae"] == "losses.reward_mae.item()"


def test_static_training_paths_enforce_fixed_shapes():
    train_world_model = get_function("train_world_model")
    train_imagination = get_function("train_agent_in_imagination")
    main = get_function("main")

    assert "repeat_batch_to_size(batch, batch_size)" in ast.unparse(train_world_model)

    imagination_source = ast.unparse(train_imagination)
    assert "static_generate_shape and prompt_length > 0 and (prompt_probability != 1.0)" in imagination_source
    assert "static compiled generation requires at least one prompt window" in imagination_source
    assert "repeat_batch_to_size(prompt_batch, batch_size)" in imagination_source

    assignments = {
        target.id: ast.unparse(node.value)
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assignments["compile"] == "device.type == 'cuda'"
    assert assignments["static_compile_shapes"] == "compile_dynamic is False"

    calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("train_world_model", "train_agent_in_imagination")
    ]

    world_model_calls = [keyword_map(call) for call in calls if call.func.id == "train_world_model"]
    imagination_call = next(keyword_map(call) for call in calls if call.func.id == "train_agent_in_imagination")

    assert len(world_model_calls) == 3
    assert all(
        ast.unparse(call["static_batch_shape"]) == "compile_world_model and static_compile_shapes"
        for call in world_model_calls
    )
    assert ast.unparse(imagination_call["static_generate_shape"]) == "compile_generate and static_compile_shapes"
