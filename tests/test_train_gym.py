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


def test_interact_uses_host_done_flags_and_injected_forward():
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
