from collections import Counter
from copy import deepcopy
from time import perf_counter

import pytest
import torch
from torch._dynamo.utils import clear_compilation_metrics, get_compilation_metrics
from torch.utils.data import Dataset

import train.train_gym as train_gym
from dreamer4.dreamer4 import DynamicsWorldModel
from train.compile_runtime import (
    DynamicTimeCacheCallable,
    ImaginationGenerateRollout,
    InteractWithEnvForward,
    build_compile_runtime,
    reset_torch_compile_state,
)


@pytest.fixture
def reset_compile_after_test():
    yield
    reset_torch_compile_state(torch.device("cpu"))


class DummyReplayDataset(Dataset):
    """Keep the trainer's DataLoader path without requiring a memmap replay."""

    def __init__(self, window_length):
        self.window_length = window_length

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {
            "latents": torch.zeros(self.window_length, 1, 4),
            "actions_continuous": torch.full((self.window_length, 2), 0.5),
            "rewards": torch.zeros(self.window_length),
        }


class CompileCounterBackend:
    """A CPU backend that records which invocation caused each compilation."""

    def __init__(self):
        self.phase = None
        self.invocation = None
        self.events = []

    def __call__(self, graph_module, example_inputs):
        self.events.append((self.phase, self.invocation, graph_module))
        return graph_module.forward

    def counts_by_invocation(self, phase):
        counts = Counter(invocation for event_phase, invocation, _ in self.events if event_phase == phase)
        return dict(sorted(counts.items()))


class InvocationTracker:
    def __init__(self, fn, backend, phase):
        self.fn = fn
        self.backend = backend
        self.phase = phase
        self.calls = 0
        self.timing = getattr(fn, "timing", None)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.backend.phase = self.phase
        self.backend.invocation = self.calls
        try:
            return self.fn(*args, **kwargs)
        finally:
            self.backend.phase = None
            self.backend.invocation = None


def make_tiny_world_model(*, dim = 8, attn_dim_head = 8, **kwargs):
    return DynamicsWorldModel(
        dim = dim,
        dim_latent = 4,
        max_steps = 8,
        num_latent_tokens = 1,
        num_spatial_tokens = 1,
        depth = 1,
        time_block_every = 1,
        attn_heads = 1,
        attn_dim_head = attn_dim_head,
        num_discrete_actions = 0,
        num_continuous_actions = 2,
        continuous_dist_type = "beta",
        reward_encoder_kwargs = dict(num_bins = 5),
        value_encoder_kwargs = dict(num_bins = 5),
        policy_head_mlp_depth = 1,
        value_head_mlp_depth = 1,
        **kwargs,
    )


def make_tiny_imagination_runtime(monkeypatch, *, denoising_steps):
    reset_torch_compile_state(torch.device("cpu"))
    monkeypatch.setattr(
        train_gym,
        "ReplayDatasetTimeWindow",
        lambda **kwargs: DummyReplayDataset(kwargs["window_length"]),
    )

    world_model = make_tiny_world_model()
    optimizer = torch.optim.Adam(world_model.agent_parameters(), lr = 1e-4)
    backend = CompileCounterBackend()
    runtime = build_compile_runtime(
        world_model,
        device = torch.device("cpu"),
        compile_world_model = False,
        compile_generate = True,
        compile_learn = True,
        compile_backend = backend,
        compile_mode = None,
        compile_fullgraph = False,
        compile_dynamic = False,
        compile_generate_cudagraphs = False,
        track_compile_performance = True,
        imagination_generate_steps = denoising_steps,
        imagination_use_time_cache = True,
        objective = "ppo",
        use_delight_gating = False,
        store_old_action_unembeds = False,
    )
    runtime.generate_rollout = InvocationTracker(runtime.generate_rollout, backend, "generate")
    runtime.learn_from_dream = InvocationTracker(runtime.learn_from_dream, backend, "learn")

    return world_model, optimizer, runtime, backend


def generate_kwargs(time_steps = 2):
    return dict(
        time_steps = time_steps,
        num_steps = 2,
        batch_size = 2,
        return_decoded_video = False,
        return_rewards_per_frame = True,
        return_agent_actions = True,
        return_log_probs_and_values = True,
        return_terminals = False,
        store_agent_embed = True,
        store_old_action_unembeds = False,
        preallocate_outputs = True,
        use_time_cache = True,
    )


def assert_matching_experience(actual, expected, *, tolerance = 0.):
    tensors = (
        (actual.latents, expected.latents),
        (actual.rewards, expected.rewards),
        (actual.agent_embed, expected.agent_embed),
        (actual.actions.continuous, expected.actions.continuous),
        (actual.log_probs.continuous, expected.log_probs.continuous),
        (actual.values, expected.values),
    )
    for actual_tensor, expected_tensor in tensors:
        assert actual_tensor.shape == expected_tensor.shape
        assert torch.allclose(actual_tensor, expected_tensor, atol = tolerance, rtol = tolerance)


def test_interact_forward_compiles_only_world_model_forward_without_cudagraphs(monkeypatch):
    compile_calls = []

    def fake_compile(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = make_tiny_world_model()
    runtime = build_compile_runtime(
        model,
        device = torch.device("cpu"),
        compile_world_model = False,
        compile_generate = False,
        compile_learn = False,
        compile_backend = "eager",
        compile_mode = "reduce-overhead",
        compile_fullgraph = False,
        compile_dynamic = False,
        compile_generate_cudagraphs = False,
        track_compile_performance = False,
        imagination_generate_steps = 2,
        imagination_use_time_cache = True,
        objective = "ppo",
        use_delight_gating = False,
        store_old_action_unembeds = False,
        compile_interact = True,
        interact_max_timesteps = 8,
    )

    assert len(compile_calls) == 1
    assert isinstance(compile_calls[0][0], InteractWithEnvForward)
    assert compile_calls[0][1] == {
        "backend": "eager",
        "fullgraph": False,
        "dynamic": True,
    }
    assert isinstance(runtime.interact_forward, DynamicTimeCacheCallable)
    interact_timing = next(timing for timing in runtime.timings if timing.name == "interact_forward")
    assert interact_timing.compile_mode == "default"
    assert interact_timing.cuda_graphs is False


def test_interact_forward_marks_kv_time_dynamic(monkeypatch):
    marked = []
    tensor = torch.zeros(1, 2, 3, 4, 5, 6)
    time_cache = (
        type("Cache", (), {"next_kv_cache": tensor})(),
        None,
        None,
        None,
        None,
    )

    monkeypatch.setattr(
        torch._dynamo,
        "mark_dynamic",
        lambda value, dim, **kwargs: marked.append((value, dim, kwargs)),
    )
    calls = []
    fn = DynamicTimeCacheCallable(lambda **kwargs: calls.append(kwargs), max_time = 9)
    fn(time_cache = time_cache)

    assert calls == [{"time_cache": time_cache}]
    assert marked == [(tensor, -2, {"min": 1, "max": 9})]


def test_interact_forward_dynamic_cache_compilation_plateaus(reset_compile_after_test):
    backend = CompileCounterBackend()
    model = make_tiny_world_model().eval()
    eager_model = deepcopy(model).eval()
    runtime = build_compile_runtime(
        model,
        device = torch.device("cpu"),
        compile_world_model = False,
        compile_generate = False,
        compile_learn = False,
        compile_backend = backend,
        compile_mode = None,
        compile_fullgraph = False,
        compile_dynamic = False,
        compile_generate_cudagraphs = False,
        track_compile_performance = False,
        imagination_generate_steps = 2,
        imagination_use_time_cache = True,
        objective = "ppo",
        use_delight_gating = False,
        store_old_action_unembeds = False,
        compile_interact = True,
        interact_max_timesteps = 8,
    )
    compiled_cache = eager_cache = None
    compilation_counts = []

    for _ in range(5):
        latents = torch.randn(2, 1, 1, 1, 4)
        kwargs = dict(
            latents = latents,
            signal_levels = 7,
            step_sizes = 2,
            rewards = None,
            discrete_actions = None,
            continuous_actions = None,
            proprio = None,
            latent_is_noised = True,
            latent_has_view_dim = True,
            return_pred_only = True,
            return_intermediates = True,
        )

        with torch.no_grad():
            compiled_pred, (compiled_embeds, compiled_cache) = runtime.interact_forward(
                **kwargs,
                time_cache = compiled_cache,
            )
            eager_pred, (eager_embeds, eager_cache) = eager_model(
                **kwargs,
                time_cache = eager_cache,
            )

        assert torch.allclose(compiled_pred.flow, eager_pred.flow)
        assert torch.allclose(compiled_embeds.agent, eager_embeds.agent)
        compilation_counts.append(len(backend.events))

    assert compilation_counts[-1] == compilation_counts[-2]
    assert compiled_cache.main.next_kv_cache.shape[-2] == 5


def test_attention_softclamp_is_a_nonpersistent_scalar_buffer():
    model = make_tiny_world_model(attn_softclamp_value = 50.)
    softclamp = model.transformer.attn_softclamp_value

    assert torch.is_tensor(softclamp)
    assert softclamp.ndim == 0
    assert softclamp.item() == 50.
    assert dict(model.transformer.named_buffers())["attn_softclamp_value"] is softclamp
    assert "attn_softclamp_value" not in model.transformer.state_dict()

    disabled_model = make_tiny_world_model(attn_softclamp_value = None)
    assert disabled_model.transformer.attn_softclamp_value is None


def test_optimized_generate_is_opt_in_matches_default_and_caches(monkeypatch):
    compile_calls = []
    compiled_forward_calls = 0

    def fake_compile(fn, **kwargs):
        nonlocal compiled_forward_calls
        compile_calls.append((fn, kwargs))

        def compiled(*args, **call_kwargs):
            nonlocal compiled_forward_calls
            compiled_forward_calls += 1
            return fn(*args, **call_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = make_tiny_world_model()
    kwargs = generate_kwargs()

    with torch.no_grad():
        torch.manual_seed(123)
        default_output = model.generate(**kwargs)
        assert model._compiled_generate_forward_fn is None

        torch.manual_seed(123)
        explicit_default_output = model.generate(**kwargs, optimize_generate_compilation = False)
        assert_matching_experience(explicit_default_output, default_output)
        assert model._compiled_generate_forward_fn is None

        torch.manual_seed(123)
        compiled_output = model.generate(**kwargs, optimize_generate_compilation = True)
        cached_fn = model._compiled_generate_forward_fn
        assert_matching_experience(compiled_output, default_output, tolerance = 3e-2)

        torch.manual_seed(123)
        repeated_output = model.generate(**kwargs, optimize_generate_compilation = True)
        assert_matching_experience(repeated_output, default_output, tolerance = 3e-2)

    assert len(compile_calls) == 1
    assert compile_calls[0][1] == {"dynamic": True, "fullgraph": False}
    assert model._compiled_generate_forward_fn is cached_fn
    assert compiled_forward_calls == 12

    model.optimize_generate_compilation = True
    with torch.no_grad():
        model.generate(**kwargs)
        model.generate(**kwargs, optimize_generate_compilation = False)
    assert len(compile_calls) == 1


def test_optimized_generate_compile_cost_does_not_scale_with_configuration(reset_compile_after_test):
    scenarios = (
        dict(name = "small", batch_size = 2, prompt_length = 0, horizon = 2, denoising_steps = 2),
        dict(name = "medium", batch_size = 4, prompt_length = 4, horizon = 4, denoising_steps = 2),
        dict(name = "large", batch_size = 8, prompt_length = 4, horizon = 8, denoising_steps = 4),
    )
    measurements = []

    for scenario in scenarios:
        reset_torch_compile_state(torch.device("cpu"))
        clear_compilation_metrics()
        backend = CompileCounterBackend()
        model = make_tiny_world_model(optimize_generate_compilation = True)
        model._compiled_generate_forward_fn = torch.compile(
            model._forward_for_generate_impl,
            backend = backend,
            dynamic = True,
            fullgraph = False,
        )
        rollout = ImaginationGenerateRollout(
            model,
            generate_steps = scenario["denoising_steps"],
            store_old_action_unembeds = False,
            use_time_cache = True,
            preallocate_outputs = True,
            optimize_generate_compilation = True,
        )
        prompt_length = scenario["prompt_length"]
        prompt_latents = prompt_continuous_actions = prompt_rewards = None

        if prompt_length > 0:
            prompt_latents = torch.randn(
                scenario["batch_size"],
                prompt_length,
                1,
                1,
                4,
            )
            prompt_continuous_actions = torch.rand(
                scenario["batch_size"],
                prompt_length,
                2,
            )
            prompt_rewards = torch.randn(scenario["batch_size"], prompt_length - 1)

        start = perf_counter()
        with torch.no_grad():
            rollout(
                scenario["horizon"] + prompt_length,
                scenario["batch_size"],
                prompt_latents,
                None,
                None,
                prompt_continuous_actions,
                prompt_rewards,
            )
        wall_seconds = perf_counter() - start

        compile_metrics = [
            metric for metric in get_compilation_metrics()
            if metric.co_name == "_forward_for_generate_impl" and metric.has_guarded_code
        ]
        node_counts = [len(list(graph.graph.nodes)) for _, _, graph in backend.events]
        compile_seconds = sum(metric.entire_frame_compile_time_s or 0. for metric in compile_metrics)
        measurements.append((scenario, len(backend.events), node_counts, compile_seconds, wall_seconds))

    print(f"optimized generation compilation: {measurements}")

    compile_counts = [measurement[1] for measurement in measurements]
    max_node_counts = [max(measurement[2]) for measurement in measurements]
    compile_seconds = [measurement[3] for measurement in measurements]

    assert max(compile_counts) - min(compile_counts) <= 1, measurements
    assert max(max_node_counts) - min(max_node_counts) <= 5, measurements
    assert max(max_node_counts) < 1000, measurements
    assert max(compile_seconds) <= max(min(compile_seconds) * 1.5, 1.), measurements


def test_optimized_generate_skips_outer_runtime_compile(monkeypatch):
    compile_calls = []

    def fake_compile(fn, **kwargs):
        compile_calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = make_tiny_world_model()
    runtime = build_compile_runtime(
        model,
        device = torch.device("cpu"),
        compile_world_model = False,
        compile_generate = True,
        compile_learn = False,
        compile_backend = "eager",
        compile_mode = None,
        compile_fullgraph = False,
        compile_dynamic = False,
        compile_generate_cudagraphs = False,
        optimize_generate_compilation = True,
        track_compile_performance = False,
        imagination_generate_steps = 2,
        imagination_use_time_cache = True,
        objective = "ppo",
        use_delight_gating = False,
        store_old_action_unembeds = False,
    )

    assert compile_calls == []

    with torch.no_grad():
        runtime.generate_rollout(2, 2)

    assert len(compile_calls) == 1
    assert compile_calls[0][0].__name__ == "_forward_for_generate_impl"
    assert compile_calls[0][1] == {"dynamic": True, "fullgraph": False}


def test_cpu_imagination_compilations_do_not_scale_with_config_values(monkeypatch, reset_compile_after_test):
    scenarios = (
        dict(
            name = "small_unprompted",
            batch_size = 2,
            prompt_probability = 0.,
            prompt_length = 0,
            horizon = 2,
            denoising_steps = 2,
        ),
        dict(
            name = "medium_prompted",
            batch_size = 4,
            prompt_probability = 1.,
            prompt_length = 2,
            horizon = 4,
            denoising_steps = 2,
        ),
        dict(
            name = "large_prompted",
            batch_size = 8,
            prompt_probability = 1.,
            prompt_length = 4,
            horizon = 8,
            denoising_steps = 4,
        ),
    )
    measurements = {}

    for scenario in scenarios:
        world_model, optimizer, runtime, backend = make_tiny_imagination_runtime(
            monkeypatch,
            denoising_steps = scenario["denoising_steps"],
        )
        global_step = 0

        # Match train_gym.main(): call the trainer once per outer loop. Three
        # calls distinguish cold specialization from persistent recompilation.
        for _ in range(3):
            global_step, _ = train_gym.train_agent_in_imagination(
                world_model,
                optimizer,
                None,
                object(),
                generate_rollout_fn = runtime.generate_rollout,
                learn_from_dream_fn = runtime.learn_from_dream,
                learn_from_dream_timing = runtime.learn_from_dream_timing,
                imagination_step_timing = runtime.imagination_step_timing,
                steps = 1,
                batch_size = scenario["batch_size"],
                horizon = scenario["horizon"],
                prompt_length = scenario["prompt_length"],
                prompt_probability = scenario["prompt_probability"],
                static_generate_shape = True,
                generate_steps = scenario["denoising_steps"],
                max_grad_norm = 1.,
                objective = "ppo",
                use_delight_gating = False,
                global_step = global_step,
                writer = None,
                training_profiler = None,
            )

        compile_counts = {
            "generate": backend.counts_by_invocation("generate"),
            "learn": backend.counts_by_invocation("learn"),
        }
        cumulative_compile_counts = {}
        for phase, counts in compile_counts.items():
            running_total = 0
            cumulative_compile_counts[phase] = []
            for invocation in range(1, 4):
                running_total += counts.get(invocation, 0)
                cumulative_compile_counts[phase].append(running_total)

        measurements[scenario["name"]] = {
            "config": scenario,
            "compile_counts": compile_counts,
            "cumulative_compile_counts": cumulative_compile_counts,
            "compile_totals": {
                phase: sum(counts.values())
                for phase, counts in compile_counts.items()
            },
            "timing_seconds": {
                "generate": runtime.generate_rollout_timing.call_seconds,
                "learn": runtime.learn_from_dream_timing.call_seconds,
                "step": runtime.imagination_step_timing.call_seconds,
            },
        }

        assert global_step == 3
        assert runtime.generate_rollout_timing.calls == 3
        assert runtime.learn_from_dream_timing.calls == 3
        assert runtime.imagination_step_timing.calls == 3
        step_seconds = runtime.imagination_step_timing.call_seconds
        print(
            f"{scenario['name']}: "
            f"compilations={measurements[scenario['name']]['compile_totals']}, "
            f"compilations_by_invocation={compile_counts}, "
            f"imagination_step_seconds="
            f"{{'first': {step_seconds[0]:.6f}, "
            f"'second': {step_seconds[1]:.6f}, "
            f"'later': {[round(value, 6) for value in step_seconds[2:]]}}}"
        )
        # A different cold count is acceptable. What must not happen is one or
        # more new graphs on every outer imagination loop: cumulative counts
        # must plateau by the third invocation.
        assert (
            cumulative_compile_counts["generate"][-1] == cumulative_compile_counts["generate"][-2]
        ), measurements[scenario["name"]]
        assert (
            cumulative_compile_counts["learn"][-1] == cumulative_compile_counts["learn"][-2]
        ), measurements[scenario["name"]]

    def has_positive_linear_growth(values):
        first_delta = values[1] - values[0]
        second_delta = values[2] - values[1]
        return first_delta > 0 and first_delta == second_delta

    scenario_names = [scenario["name"] for scenario in scenarios]
    for phase in ("generate", "learn"):
        cold_totals = [
            measurements[name]["compile_totals"][phase]
            for name in scenario_names
        ]
        assert not has_positive_linear_growth(cold_totals), {
            "phase": phase,
            "cold_totals": cold_totals,
            "measurements": measurements,
        }
