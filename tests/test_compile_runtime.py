from collections import Counter
from copy import deepcopy
from time import perf_counter

import pytest
import torch
from torch._dynamo.utils import clear_compilation_metrics, get_compilation_metrics
from torch.utils.data import Dataset

import train.train_gym as train_gym
import dreamer4.dreamer4 as dreamer4_module
from dreamer4.dreamer4 import Actions, DynamicsWorldModel, LossNormalizer
from train.compile_runtime import (
    DynamicTimeCacheCallable,
    ImaginationGenerateRollout,
    ImaginationLearningLoss,
    InteractWithEnvForward,
    WorldModelTrainingLoss,
    action_pair_or_empty,
    build_compile_runtime,
    configure_sac,
    maybe_compile_and_time,
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


@pytest.mark.parametrize("budget", (-0.1, 1.1, float("nan"), "invalid"))
def test_configure_sac_rejects_invalid_memory_budget(budget):
    with pytest.raises(ValueError, match = "activation_memory_budget"):
        configure_sac(budget)


def make_tiny_world_model(*, dim = 8, attn_dim_head = 8, continuous_dist_type = "beta", **kwargs):
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
        continuous_dist_type = continuous_dist_type,
        reward_encoder_kwargs = dict(num_bins = 5),
        value_encoder_kwargs = dict(num_bins = 5),
        policy_head_mlp_depth = 1,
        value_head_mlp_depth = 1,
        **kwargs,
    )


def test_compiled_world_model_loss_normalization_skips_disabled_action_loss(reset_compile_after_test):
    model = make_tiny_world_model(
        continuous_action_loss_weight = 0.,
        discrete_action_loss_weight = 0.,
        old_reward_head = True,
        old_reward_shifting = True,
        use_loss_normalization = True,
    )
    loss_fn = torch.compile(
        WorldModelTrainingLoss(model),
        backend = "eager",
        fullgraph = False,
        dynamic = False,
    )

    loss, losses = loss_fn(
        torch.randn(2, 4, 1, 4),
        torch.randn(2, 4),
        torch.zeros(2, dtype = torch.bool),
        torch.rand(2, 4, 2),
        torch.tensor((4, 3)),
        "finetune",
        True,
    )
    loss.backward()

    assert loss.ndim == 0
    assert losses.continuous_actions.ndim == 0


def test_compiled_paper_shortcut_objective_supports_mixed_batches(reset_compile_after_test):
    torch.manual_seed(42)
    model = make_tiny_world_model(
        continuous_action_loss_weight = 0.,
        discrete_action_loss_weight = 0.,
        use_loss_normalization = True,
        use_paper_aligned_shortcut_objective = True,
    )
    loss_fn = torch.compile(
        WorldModelTrainingLoss(model),
        backend = "eager",
        fullgraph = False,
        dynamic = False,
    )

    loss, losses = loss_fn(
        torch.randn(16, 4, 1, 4),
        torch.randn(16, 4),
        torch.zeros(16, dtype = torch.bool),
        torch.rand(16, 4, 2),
        torch.full((16,), 4),
        "combined",
        True,
    )
    loss.backward()

    assert loss.ndim == 0
    assert losses.diagnostics.raw_flow > 0.
    assert losses.diagnostics.raw_shortcut > 0.


def test_grouped_dynamics_loss_normalization_keeps_the_flow_to_bootstrap_ratio():
    torch.manual_seed(42)
    model = make_tiny_world_model(
        continuous_action_loss_weight = 0.,
        discrete_action_loss_weight = 0.,
        prob_shortcut_train = 1.,
        use_loss_normalization = True,
        group_dynamics_loss_normalization = True,
    )
    loss_fn = WorldModelTrainingLoss(model)
    batch = (
        torch.randn(16, 4, 1, 4),
        torch.randn(16, 4),
        torch.zeros(16, dtype = torch.bool),
        torch.rand(16, 4, 2),
        torch.full((16,), 4),
        "combined",
        True,
    )

    # The statistic starts at 1. The first call moves it to the scale of the flow and bootstrap sum.
    loss_fn(*batch)
    rms = model.dynamics_loss_normalizer.exp_avg_sq.sqrt().item()
    assert rms < 1.

    _, losses = loss_fn(*batch)
    diagnostics = losses.diagnostics

    assert diagnostics.raw_shortcut > 0.
    assert losses.flow.item() == pytest.approx(diagnostics.raw_flow.item() / rms, rel = 1e-5)
    assert losses.shortcut.item() == pytest.approx(diagnostics.raw_shortcut.item() / rms, rel = 1e-5)
    assert diagnostics.normalized_dynamics.item() == pytest.approx(diagnostics.raw_dynamics.item() / rms, rel = 1e-5)

    # The separate statistics of the default path stay unused.
    assert model.flow_loss_normalizer.exp_avg_sq.item() == 1.
    assert model.shortcut_flow_loss_normalizer.exp_avg_sq.item() == 1.


def test_loss_normalization_warmup_gives_the_plain_sum_and_updates_the_statistics():
    batch = (
        torch.randn(16, 4, 1, 4),
        torch.randn(16, 4),
        torch.zeros(16, dtype = torch.bool),
        torch.rand(16, 4, 2),
        torch.full((16,), 4),
        "combined",
        True,
    )

    def loss_of(apply_loss_normalization = None):
        torch.manual_seed(42)
        normalize = apply_loss_normalization is not None
        model = make_tiny_world_model(
            continuous_action_loss_weight = 0.,
            discrete_action_loss_weight = 0.,
            prob_shortcut_train = 1.,
            **(dict(use_loss_normalization = True, group_dynamics_loss_normalization = True) if normalize else dict()),
        )
        if normalize:
            model.apply_loss_normalization = apply_loss_normalization
            # Statistics away from their start value 1, so the normalized loss differs from the plain sum.
            for module in model.modules():
                if isinstance(module, LossNormalizer):
                    module.exp_avg_sq.fill_(.25)
        torch.manual_seed(0)
        loss, _ = WorldModelTrainingLoss(model)(*batch)
        return model, loss

    _, plain_loss = loss_of()
    _, normalized_loss = loss_of(apply_loss_normalization = True)
    model, warmup_loss = loss_of(apply_loss_normalization = False)

    assert normalized_loss.item() != pytest.approx(plain_loss.item(), rel = 1e-3)
    assert warmup_loss.item() == pytest.approx(plain_loss.item(), rel = 1e-6)
    assert model.dynamics_loss_normalizer.exp_avg_sq.item() != .25
    assert (model.agent_embed_reward_loss_normalizer.exp_avg_sq != .25).all()


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


def apply_imagination_loss(loss_module, dream, preservation = None):
    actions = action_pair_or_empty(dream.actions)
    log_probs = action_pair_or_empty(dream.log_probs)
    old_action_unembeds = action_pair_or_empty(dream.old_action_unembeds)
    preservation_actions = action_pair_or_empty(
        preservation.actions if preservation is not None else None
    )

    return loss_module(
        dream.latents,
        dream.proprio,
        dream.critic_state,
        dream.agent_embed,
        dream.policy_input,
        dream.rewards,
        dream.terminals,
        actions.discrete,
        actions.continuous,
        log_probs.discrete,
        log_probs.continuous,
        old_action_unembeds.discrete,
        old_action_unembeds.continuous,
        dream.values,
        dream.step_size,
        dream.lens,
        dream.is_truncated,
        dream.is_from_world_model,
        dream.episode_return,
        preservation.latents if preservation is not None else None,
        preservation.proprio if preservation is not None else None,
        preservation.rewards if preservation is not None else None,
        preservation_actions.discrete,
        preservation_actions.continuous,
        preservation.lens if preservation is not None else None,
    )


@pytest.mark.parametrize("objective", ("pmpo", "ppo", "spo", "reinforce"))
def test_imagination_diagnostics_log_policy_signal_and_on_policy_ratio(objective):
    model = make_tiny_world_model()
    dream_kwargs = generate_kwargs(time_steps = 4)
    dream_kwargs["store_old_action_unembeds"] = True

    with torch.no_grad():
        dream = model.generate(**dream_kwargs)

    loss_module = ImaginationLearningLoss(
        model,
        objective = objective,
        use_delight_gating = False,
    )
    _, _, diagnostics = apply_imagination_loss(loss_module, dream)

    expected = {
        "extra/advantage_mean",
        "extra/advantage_abs_mean",
        "extra/policy_entropy",
        "extra/importance_ratio_clip_frac",
    }
    assert expected <= diagnostics.keys()
    assert all(torch.isfinite(diagnostics[key]) for key in expected)
    assert diagnostics["extra/importance_ratio_clip_frac"].item() == 0.


def test_reinforce_uses_reward_to_go_without_a_behavior_ratio(monkeypatch):
    model = make_tiny_world_model()
    model.policy_entropy_weight = 0.

    with torch.no_grad():
        dream = model.generate(**generate_kwargs(time_steps = 4))

    captured_lambdas = []
    original_calc_gae = dreamer4_module.calc_gae

    def capture_calc_gae(*args, **kwargs):
        captured_lambdas.append(kwargs["lam"])
        return original_calc_gae(*args, **kwargs)

    monkeypatch.setattr(dreamer4_module, "calc_gae", capture_calc_gae)
    loss_module = ImaginationLearningLoss(
        model,
        objective = "reinforce",
        use_delight_gating = False,
    )
    policy_loss, _, _ = apply_imagination_loss(loss_module, dream)

    shifted_dream = deepcopy(dream)
    shifted_dream.log_probs = Actions(
        dream.log_probs.discrete,
        dream.log_probs.continuous + 5.,
    )
    shifted_policy_loss, _, _ = apply_imagination_loss(loss_module, shifted_dream)

    assert captured_lambdas == [1., 1.]
    assert torch.allclose(policy_loss, shifted_policy_loss)


def test_dreamerv3_reinforce_controls_set_targets_and_return_scale(monkeypatch):
    discount_factor = 1. - 1. / 333.
    model = make_tiny_world_model(
        reinforce_discount_factor = discount_factor,
        reinforce_gae_lambda = 0.95,
        reinforce_return_normalization = "percentile",
        reinforce_use_discounted_trajectory_weights = True,
        policy_entropy_weight = 0.0003,
    )

    with torch.no_grad():
        dream = model.generate(**generate_kwargs(time_steps = 4))
        dream.rewards.copy_(torch.linspace(0., 3., dream.rewards.numel()).reshape_as(dream.rewards))
        dream.values.zero_()

    captured = {}
    original_calc_gae = dreamer4_module.calc_gae

    def capture_calc_gae(*args, **kwargs):
        captured.update(gamma = kwargs["gamma"], lam = kwargs["lam"])
        return original_calc_gae(*args, **kwargs)

    monkeypatch.setattr(dreamer4_module, "calc_gae", capture_calc_gae)
    policy_loss, value_loss, _ = apply_imagination_loss(
        ImaginationLearningLoss(
            model,
            objective = "reinforce",
            use_delight_gating = False,
        ),
        dream,
    )

    assert captured == {"gamma": discount_factor, "lam": 0.95}
    assert model.reinforce_return_high > model.reinforce_return_low
    assert model.policy_entropy_weight == 0.0003
    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)


def test_dreamerv3_reinforce_actor_loss_supports_squashed_gaussian():
    model = make_tiny_world_model(
        continuous_dist_type = "squashed_gaussian",
        reinforce_discount_factor = 1. - 1. / 333.,
        reinforce_gae_lambda = 0.95,
        reinforce_return_normalization = "percentile",
        reinforce_use_discounted_trajectory_weights = True,
        policy_entropy_weight = 0.0003,
    )

    with torch.no_grad():
        dream = model.generate(**generate_kwargs(time_steps = 4))

    policy_loss, value_loss, _ = apply_imagination_loss(
        ImaginationLearningLoss(
            model,
            objective = "reinforce",
            use_delight_gating = False,
        ),
        dream,
    )
    (policy_loss + value_loss).backward()

    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)
    assert any(parameter.grad is not None for parameter in model.policy_head.parameters())


def test_default_model_checkpoint_has_no_reinforce_normalizer_state():
    baseline = make_tiny_world_model()
    aligned = make_tiny_world_model(reinforce_return_normalization = "percentile")

    assert not any(key.startswith("reinforce_return_") for key in baseline.state_dict())
    assert "reinforce_return_low" in aligned.state_dict()
    assert "reinforce_return_high" in aligned.state_dict()


def test_compiled_reinforce_loss_backpropagates(reset_compile_after_test):
    model = make_tiny_world_model()

    with torch.no_grad():
        dream = model.generate(**generate_kwargs(time_steps = 4))

    loss_module = torch.compile(
        ImaginationLearningLoss(
            model,
            objective = "reinforce",
            use_delight_gating = False,
        ),
        backend = "eager",
        fullgraph = False,
        dynamic = False,
    )
    policy_loss, value_loss, diagnostics = apply_imagination_loss(loss_module, dream)
    (policy_loss + value_loss).backward()

    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(diagnostics["extra/advantage_mean"])
    assert any(parameter.grad is not None for parameter in model.policy_head.parameters())


def test_full_transformer_finetuning_is_opt_in_and_preserves_world_model_functions():
    source_model = make_tiny_world_model()
    dream_kwargs = generate_kwargs(time_steps = 4)
    dream_kwargs["store_old_action_unembeds"] = True

    with torch.no_grad():
        dream = source_model.generate(**dream_kwargs)

    disabled_model = deepcopy(source_model)
    disabled_loss = ImaginationLearningLoss(
        disabled_model,
        objective = "pmpo",
        use_delight_gating = False,
    )
    policy_loss, value_loss, diagnostics = apply_imagination_loss(disabled_loss, dream)
    (policy_loss + value_loss).backward()

    assert all(parameter.grad is None for parameter in disabled_model.transformer.parameters())
    assert all(parameter.grad is None for parameter in disabled_model.to_latent_pred.parameters())
    assert all(parameter.grad is None for parameter in disabled_model.to_agent_embed_reward_pred.parameters())
    assert not any(key.startswith("preservation/") for key in diagnostics)

    enabled_model = deepcopy(source_model)
    enabled_loss = ImaginationLearningLoss(
        enabled_model,
        objective = "pmpo",
        use_delight_gating = False,
        full_transformer_finetuning = True,
    )
    policy_loss, value_loss, diagnostics = apply_imagination_loss(enabled_loss, dream, dream)
    (policy_loss + value_loss).backward()

    assert any(parameter.grad is not None for parameter in enabled_model.transformer.parameters())
    assert any(parameter.grad is not None for parameter in enabled_model.to_latent_pred.parameters())
    assert any(parameter.grad is not None for parameter in enabled_model.to_agent_embed_reward_pred.parameters())
    assert {
        "preservation/dynamics_loss",
        "preservation/reward_loss",
        "preservation/loss",
    } <= diagnostics.keys()


def test_full_transformer_finetuning_requires_policy_prior_loss():
    model = make_tiny_world_model()

    with pytest.raises(ValueError, match = "objective='pmpo'"):
        ImaginationLearningLoss(
            model,
            objective = "ppo",
            use_delight_gating = False,
            full_transformer_finetuning = True,
        )

    model.pmpo_kl_div_loss_weight = 0.
    with pytest.raises(ValueError, match = "pmpo_kl_div_loss_weight > 0"):
        ImaginationLearningLoss(
            model,
            objective = "pmpo",
            use_delight_gating = False,
            full_transformer_finetuning = True,
        )


def test_full_transformer_finetuning_uses_replay_preservation_batch(monkeypatch):
    preservation_window_lengths = []
    dataloader_batch_sizes = []
    original_dataloader = train_gym.DataLoader

    def make_replay_dataset(**kwargs):
        preservation_window_lengths.append(kwargs["window_length"])
        return DummyReplayDataset(kwargs["window_length"])

    def make_dataloader(dataset, **kwargs):
        dataloader_batch_sizes.append(kwargs["batch_size"])
        return original_dataloader(dataset, **kwargs)

    monkeypatch.setattr(
        train_gym,
        "ReplayDatasetTimeWindow",
        make_replay_dataset,
    )
    monkeypatch.setattr(train_gym, "DataLoader", make_dataloader)
    world_model = make_tiny_world_model()
    policy_prior = train_gym.FrozenPolicyPrior(world_model)
    optimizer = torch.optim.Adam(
        train_gym.agent_optimizer_parameters(world_model, True),
        lr = 1e-4,
    )
    runtime = build_compile_runtime(
        world_model,
        device = torch.device("cpu"),
        compile_world_model = False,
        compile_generate = False,
        compile_learn = False,
        compile_backend = "eager",
        compile_mode = None,
        compile_fullgraph = False,
        compile_dynamic = False,
        compile_generate_cudagraphs = False,
        track_compile_performance = False,
        imagination_generate_steps = 2,
        imagination_use_time_cache = False,
        objective = "pmpo",
        use_delight_gating = False,
        store_old_action_unembeds = False,
        full_transformer_finetuning = True,
    )

    _, metrics = train_gym.train_agent_in_imagination(
        world_model,
        optimizer,
        policy_prior,
        object(),
        generate_rollout_fn = runtime.generate_rollout,
        learn_from_dream_fn = runtime.learn_from_dream,
        learn_from_dream_timing = None,
        imagination_step_timing = None,
        steps = 1,
        batch_size = 2,
        horizon = 4,
        prompt_length = 0,
        prompt_probability = 0.,
        static_generate_shape = False,
        generate_steps = 2,
        max_grad_norm = 1.,
        objective = "pmpo",
        use_delight_gating = False,
        global_step = 0,
        writer = None,
        training_profiler = None,
        full_transformer_finetuning = True,
        preservation_sequence_length = 7,
        preservation_batch_size = 3,
    )

    assert preservation_window_lengths == [0, 7]
    assert dataloader_batch_sizes == [2, 3]
    assert "preservation/dynamics_loss" in metrics
    assert "preservation/reward_loss" in metrics


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


def test_compile_options_are_passed_without_compile_mode(monkeypatch):
    compile_calls = []

    def fake_compile(fn, **kwargs):
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    options = {
        "cpp_wrapper": True,
        "max_autotune_gemm": True,
        "b2b_gemm_pass": True,
        "triton.enable_persistent_tma_matmul": True,
        "epilogue_fusion_first": True,
    }

    _, timing = maybe_compile_and_time(
        lambda value: value,
        name = "custom_options",
        enabled = True,
        track_timing = False,
        device = torch.device("cpu"),
        compile_backend = "inductor",
        compile_mode = "reduce-overhead",
        compile_fullgraph = False,
        compile_dynamic = False,
        compile_options = options,
    )

    assert len(compile_calls) == 1
    assert compile_calls[0]["options"] == options
    assert "mode" not in compile_calls[0]
    assert timing.compile_mode is None
    assert timing.compile_options == tuple(f"{key}={value}" for key, value in options.items())


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


def test_compiled_interact_preserves_legacy_action_time_axis(reset_compile_after_test):
    model = make_tiny_world_model(
        old_continuous_action_encoder = True,
        add_action_embed_to_agent_token = True,
    ).eval()
    eager_model = deepcopy(model).eval()
    compiled = torch.compile(
        InteractWithEnvForward(model),
        backend = "eager",
        dynamic = True,
    )
    kwargs = dict(
        latents = torch.randn(2, 1, 1, 1, 4),
        signal_levels = 7,
        step_sizes = 2,
        rewards = None,
        discrete_actions = None,
        continuous_actions = torch.rand(2, 1, 2),
        proprio = None,
        time_cache = None,
        latent_is_noised = True,
        latent_has_view_dim = True,
        return_pred_only = True,
        return_intermediates = True,
    )

    with torch.no_grad():
        compiled_pred, (compiled_embeds, _) = compiled(**kwargs)
        eager_pred, (eager_embeds, _) = InteractWithEnvForward(eager_model)(**kwargs)

    assert compiled_pred.flow.shape == (2, 1, 1, 1, 4)
    assert compiled_embeds.agent.shape == (2, 1, 1, 8)
    assert torch.allclose(compiled_pred.flow, eager_pred.flow)
    assert torch.allclose(compiled_embeds.agent, eager_embeds.agent)


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


def test_cached_generate_inner_compile_matches_eager_and_plateaus(reset_compile_after_test):
    backend = CompileCounterBackend()
    model = make_tiny_world_model().eval()
    runtime = build_compile_runtime(
        model,
        device = torch.device("cpu"),
        compile_world_model = False,
        compile_generate = True,
        compile_learn = False,
        compile_backend = backend,
        compile_mode = None,
        compile_fullgraph = False,
        compile_dynamic = False,
        compile_generate_cudagraphs = False,
        optimize_generate_compilation = False,
        compile_generate_with_time_cache = True,
        track_compile_performance = False,
        imagination_generate_steps = 2,
        imagination_use_time_cache = True,
        imagination_max_timesteps = 5,
        objective = "ppo",
        use_delight_gating = False,
        store_old_action_unembeds = False,
    )
    kwargs = generate_kwargs(time_steps = 5)
    cumulative_compilations = []

    with torch.no_grad():
        torch.manual_seed(123)
        eager = model.generate(**kwargs)

        for invocation in range(1, 4):
            backend.phase = "generate"
            backend.invocation = invocation
            torch.manual_seed(123)
            actual = runtime.generate_rollout(5, 2)
            backend.phase = backend.invocation = None
            assert_matching_experience(actual, eager)
            cumulative_compilations.append(len(backend.events))

    # An empty-cache graph and cached graph fragments are expected on the cold
    # rollout. Repeating the same outer training loop must not compile again.
    assert cumulative_compilations[-1] == cumulative_compilations[-2]
    assert backend.counts_by_invocation("generate").keys() == {1}


def test_cached_generate_inner_compile_matches_eager_forward_and_backward(
    reset_compile_after_test,
):
    torch.manual_seed(5)
    eager_model = make_tiny_world_model().eval()
    compiled_model = deepcopy(eager_model).eval()

    torch.manual_seed(11)
    latents = torch.randn(2, 3, 1, 1, 4)
    signal_levels = torch.full((2, 3), 7, dtype = torch.long)
    continuous_actions = torch.rand(2, 3, 2)
    rewards = torch.randn(2, 3)

    def call(fn, time, cache = None):
        return fn(
            latents = latents[:, :time],
            signal_levels = signal_levels[:, :time],
            step_sizes = 2,
            rewards = rewards[:, :time],
            tasks = None,
            latent_gene_ids = None,
            discrete_actions = None,
            continuous_actions = continuous_actions[:, :time],
            proprio = None,
            time_cache = cache,
            aug_id = False,
            latent_is_noised = True,
            latent_has_view_dim = True,
            return_pred_only = True,
            return_intermediates = True,
        )

    def output_loss(output):
        prediction, (embeds, cache) = output
        tensors = (prediction.flow, embeds.agent, embeds.actor, embeds.critic)
        loss = sum(tensor.float().square().mean() for tensor in tensors if tensor is not None)
        return prediction, embeds, cache, loss

    # Reference one full causal forward against a compiled incremental forward.
    # Restrict the loss to the newly processed state so their backward contracts
    # are the same even though the cached route first materializes the prefix.
    eager_prediction, eager_embeds, _, _ = output_loss(
        call(eager_model._forward_for_generate_impl, 3)
    )
    eager_tensors = (
        eager_prediction.flow,
        eager_embeds.agent,
        eager_embeds.actor,
        eager_embeds.critic,
    )
    eager_loss = sum(
        tensor[:, -1:].float().square().mean()
        for tensor in eager_tensors
        if tensor is not None
    )
    eager_loss.backward()

    compiled_forward = torch.compile(
        compiled_model._forward_for_generate_impl,
        backend = "eager",
        dynamic = True,
        fullgraph = False,
    )
    compiled_forward = DynamicTimeCacheCallable(compiled_forward, max_time = 3)
    _, _, compiled_cache, _ = output_loss(call(compiled_forward, 2))
    compiled_prediction, compiled_embeds, _, _ = output_loss(
        call(compiled_forward, 3, compiled_cache)
    )
    compiled_tensors = (
        compiled_prediction.flow,
        compiled_embeds.agent,
        compiled_embeds.actor,
        compiled_embeds.critic,
    )
    compiled_loss = sum(
        tensor[:, -1:].float().square().mean()
        for tensor in compiled_tensors
        if tensor is not None
    )
    compiled_loss.backward()

    assert torch.allclose(
        compiled_prediction.flow[:, -1:],
        eager_prediction.flow[:, -1:],
    )
    for compiled_tensor, eager_tensor in zip(compiled_embeds, eager_embeds):
        if compiled_tensor is not None:
            assert torch.allclose(compiled_tensor[:, -1:], eager_tensor[:, -1:])
    assert torch.allclose(compiled_loss, eager_loss)

    for (eager_name, eager_parameter), (compiled_name, compiled_parameter) in zip(
        eager_model.named_parameters(),
        compiled_model.named_parameters(),
    ):
        assert compiled_name == eager_name
        assert (compiled_parameter.grad is None) == (eager_parameter.grad is None)
        if compiled_parameter.grad is not None:
            assert torch.allclose(
                compiled_parameter.grad,
                eager_parameter.grad,
                atol = 2e-4,
                rtol = 2e-5,
            ), eager_name


def test_cached_generate_compile_rejects_incompatible_modes():
    model = make_tiny_world_model()
    common = dict(
        world_model = model,
        device = torch.device("cpu"),
        compile_world_model = False,
        compile_generate = True,
        compile_learn = False,
        compile_backend = "eager",
        compile_mode = None,
        compile_fullgraph = False,
        compile_dynamic = False,
        compile_generate_cudagraphs = False,
        compile_generate_with_time_cache = True,
        track_compile_performance = False,
        imagination_generate_steps = 2,
        objective = "ppo",
        use_delight_gating = False,
        store_old_action_unembeds = False,
    )

    with pytest.raises(ValueError, match = "cannot both be enabled"):
        build_compile_runtime(
            **common,
            optimize_generate_compilation = True,
            imagination_use_time_cache = True,
        )

    with pytest.raises(ValueError, match = "requires imagination_use_time_cache"):
        build_compile_runtime(
            **common,
            optimize_generate_compilation = False,
            imagination_use_time_cache = False,
        )


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


def test_eager_without_time_cache_callable_selects_the_forward_by_the_time_cache():
    from train.compile_runtime import EagerWithoutTimeCacheCallable

    forward = EagerWithoutTimeCacheCallable(
        lambda **kwargs: "compiled",
        lambda **kwargs: "eager",
    )

    # The first generated frame has no time cache. On CUDA the compiled forward is wrong there.
    assert forward(latents = None, time_cache = None) == "eager"
    assert forward(latents = None) == "eager"
    assert forward(latents = None, time_cache = (1, ())) == "compiled"
