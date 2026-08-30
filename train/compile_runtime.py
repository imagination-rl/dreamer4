"""torch.compile wrappers and timing instrumentation for Gym imagination runs."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Literal, Mapping

import torch
from torch import Tensor, nn

from dreamer4.dreamer4 import (
    Actions,
    DynamicsWorldModel,
    Experience,
    ShortcutTrainMode,
    exists,
)

@dataclass
class CallTiming:
    name: str
    compiled: bool
    compile_mode: str | None = None
    compile_options: tuple[str, ...] = ()
    cuda_graphs: bool = False
    # CUDA Graph modes capture on calls 2-3, not just compile on call 1, so
    # cudagraph timings treat the first few calls as cold instead of only the first
    warmup_calls: int = 1
    call_seconds: list[float] = field(default_factory = list)

    def record(self, seconds: float):
        self.call_seconds.append(seconds)

    @property
    def calls(self):
        return len(self.call_seconds)

    @property
    def total_seconds(self):
        return sum(self.call_seconds)

    @property
    def first_call_seconds(self):
        return self.call_seconds[0] if self.calls > 0 else None

    @property
    def cold_calls(self):
        return min(self.calls, max(self.warmup_calls, 1))

    @property
    def cold_seconds(self):
        return sum(self.call_seconds[:max(self.warmup_calls, 1)])

    @property
    def warm_calls(self):
        return self.calls - self.cold_calls

    @property
    def warm_seconds(self):
        return self.total_seconds - self.cold_seconds

    @property
    def warm_average_seconds(self):
        return self.warm_seconds / self.warm_calls if self.warm_calls > 0 else None


class TimedCallable:
    def __init__(
        self,
        fn: Callable,
        timing: CallTiming,
        device: torch.device,
    ):
        self.fn = fn
        self.timing = timing
        self.device = device

    def __call__(self, *args, **kwargs):
        synchronize_if_cuda(self.device)
        start = perf_counter()
        out = self.fn(*args, **kwargs)
        synchronize_if_cuda(self.device)
        self.timing.record(perf_counter() - start)
        return out


def normalize_compile_mode(mode: str | None):
    if not exists(mode):
        return None

    mode = str(mode)

    if mode in ("", "none", "None", "null"):
        return None

    if mode == "default":
        return "default"

    if mode in ("reduce-overhead", "reduce_overhead"):
        return "reduce-overhead"

    if mode in ("max-autotune", "max_autotune"):
        return "max-autotune"

    raise ValueError("compile_mode must be one of None, 'default', 'reduce-overhead', or 'max-autotune'")


def torch_compile_mode(mode: str | None):
    if mode == "max-autotune-no-cudagraphs":
        return mode

    return normalize_compile_mode(mode)


def compile_mode_uses_cuda_graphs(mode: str | None):
    return normalize_compile_mode(mode) in ("reduce-overhead", "max-autotune")


def configure_sac(activation_memory_budget: float):
    """Configure torch.compile's selective activation recomputation budget."""
    try:
        activation_memory_budget = float(activation_memory_budget)
    except (TypeError, ValueError) as error:
        raise ValueError("activation_memory_budget must be a number between 0 and 1") from error

    if not 0. <= activation_memory_budget <= 1.:
        raise ValueError("activation_memory_budget must be between 0 and 1")

    # Importing Dynamo first initializes the compiler config modules before the
    # private functorch config is updated. Keep this compatible with PyTorch
    # builds that do not expose functorch by treating SAC as unavailable.
    _ = torch._dynamo
    try:
        from torch._functorch import config as functorch_config
    except ImportError:
        return False

    functorch_config.activation_memory_budget = activation_memory_budget
    functorch_config.activation_offload_cpu_gpu_bw = 300
    return True


def cudagraph_mark_step_begin():
    mark_step_begin = getattr(torch.compiler, "cudagraph_mark_step_begin", None)

    if callable(mark_step_begin):
        mark_step_begin()


class CompiledCallable:
    def __init__(
        self,
        fn: Callable,
        timing: CallTiming,
    ):
        self.fn = fn
        self.timing = timing

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class DynamicTimeCacheCallable:
    def __init__(self, fn: Callable, max_time: int):
        self.fn = fn
        self.max_time = max_time
        self.timing = getattr(fn, "timing", None)

    def __call__(self, *args, **kwargs):
        time_cache = kwargs.get("time_cache")

        if exists(time_cache):
            for transformer_cache in time_cache[:5]:
                if not exists(transformer_cache):
                    continue

                kv_cache = transformer_cache.next_kv_cache
                if exists(kv_cache):
                    torch._dynamo.mark_dynamic(
                        kv_cache,
                        -2,
                        min = 1,
                        max = self.max_time,
                    )

        return self.fn(*args, **kwargs)


class InteractWithEnvForward(nn.Module):
    def __init__(self, world_model: DynamicsWorldModel):
        super().__init__()
        self.world_model = world_model

    def forward(
        self,
        *,
        latents: Tensor,
        signal_levels: int | Tensor,
        step_sizes: int | Tensor,
        rewards: Tensor | None,
        discrete_actions: Tensor | None,
        continuous_actions: Tensor | None,
        proprio: Tensor | None,
        time_cache,
        latent_is_noised: bool = True,
        latent_has_view_dim: bool = True,
        return_pred_only: bool = True,
        return_intermediates: bool = True,
    ):
        return self.world_model(
            latents = latents,
            signal_levels = signal_levels,
            step_sizes = step_sizes,
            rewards = rewards,
            discrete_actions = discrete_actions,
            continuous_actions = continuous_actions,
            proprio = proprio,
            time_cache = time_cache,
            latent_is_noised = latent_is_noised,
            latent_has_view_dim = latent_has_view_dim,
            return_pred_only = return_pred_only,
            return_intermediates = return_intermediates,
        )


class WorldModelTrainingLoss(nn.Module):
    def __init__(self, world_model: DynamicsWorldModel):
        super().__init__()
        self.world_model = world_model

    def forward(
        self,
        latents: Tensor,
        rewards: Tensor | None,
        terminals: Tensor | None,
        continuous_actions: Tensor | None,
        lens: Tensor | None,
        shortcut_train_mode: ShortcutTrainMode = "combined",
        update_loss_ema: bool = True,
    ):
        return self.world_model(
            latents = latents,
            rewards = rewards,
            terminals = terminals,
            continuous_actions = continuous_actions,
            lens = lens,
            latent_has_view_dim = latents.ndim == 5,
            return_all_losses = True,
            update_loss_ema = update_loss_ema,
            shortcut_train_mode = shortcut_train_mode,
        )


class ImaginationGenerateRollout(nn.Module):
    def __init__(
        self,
        world_model: DynamicsWorldModel,
        *,
        generate_steps: int,
        store_old_action_unembeds: bool,
        use_time_cache: bool,
        preallocate_outputs: bool,
        optimize_generate_compilation: bool = False,
        forward_for_generate: Callable | None = None,
    ):
        super().__init__()
        self.world_model = world_model
        self.generate_steps = generate_steps
        self.store_old_action_unembeds = store_old_action_unembeds
        self.use_time_cache = use_time_cache
        self.preallocate_outputs = preallocate_outputs
        self.optimize_generate_compilation = optimize_generate_compilation
        self.forward_for_generate = forward_for_generate

    def forward(
        self,
        generation_horizon: int,
        batch_size: int,
        prompt_latents: Tensor | None = None,
        prompt_proprio: Tensor | None = None,
        prompt_discrete_actions: Tensor | None = None,
        prompt_continuous_actions: Tensor | None = None,
        prompt_rewards: Tensor | None = None,
    ):
        return self.world_model.generate(
            generation_horizon,
            num_steps = self.generate_steps,
            batch_size = batch_size,
            return_decoded_video = False,
            return_agent_actions = True,
            return_log_probs_and_values = True,
            return_rewards_per_frame = True,
            return_terminals = False,
            use_time_cache = self.use_time_cache,
            store_agent_embed = True,
            store_old_action_unembeds = self.store_old_action_unembeds,
            preallocate_outputs = self.preallocate_outputs,
            optimize_generate_compilation = self.optimize_generate_compilation,
            forward_for_generate = self.forward_for_generate,
            prompt_latents = prompt_latents,
            prompt_proprio = prompt_proprio,
            prompt_discrete_actions = prompt_discrete_actions,
            prompt_continuous_actions = prompt_continuous_actions,
            prompt_rewards = prompt_rewards,
        )


class ImaginationLearningLoss(nn.Module):
    def __init__(
        self,
        world_model: DynamicsWorldModel,
        *,
        objective: Literal["ppo", "pmpo", "spo"],
        use_delight_gating: bool,
        full_transformer_finetuning: bool = False,
    ):
        super().__init__()
        self.world_model = world_model
        self.objective = objective
        self.use_delight_gating = use_delight_gating
        self.full_transformer_finetuning = full_transformer_finetuning

        if full_transformer_finetuning and objective != "pmpo":
            raise ValueError("full_transformer_finetuning requires objective='pmpo' for the policy-prior loss")

        if full_transformer_finetuning and world_model.pmpo_kl_div_loss_weight <= 0.:
            raise ValueError("full_transformer_finetuning requires pmpo_kl_div_loss_weight > 0")

    def forward(
        self,
        latents: Tensor,
        proprio: Tensor | None,
        critic_state: Tensor | None,
        agent_embed: Tensor | None,
        policy_input: Tensor | None,
        rewards: Tensor | None,
        terminals: Tensor | None,
        actions_discrete: Tensor | None,
        actions_continuous: Tensor | None,
        log_probs_discrete: Tensor | None,
        log_probs_continuous: Tensor | None,
        old_action_unembeds_discrete: Tensor | None,
        old_action_unembeds_continuous: Tensor | None,
        values: Tensor | None,
        step_size: int | Tensor | None,
        lens: Tensor | None,
        is_truncated: Tensor | None,
        is_from_world_model: bool | Tensor,
        episode_return: Tensor | None,
        preservation_latents: Tensor | None,
        preservation_proprio: Tensor | None,
        preservation_rewards: Tensor | None,
        preservation_actions_discrete: Tensor | None,
        preservation_actions_continuous: Tensor | None,
        preservation_lens: Tensor | None,
    ):
        if (
            self.full_transformer_finetuning and
            old_action_unembeds_discrete is None and
            old_action_unembeds_continuous is None
        ):
            raise ValueError("full_transformer_finetuning requires frozen policy-prior targets")

        experience = Experience(
            latents = latents,
            proprio = proprio,
            critic_state = critic_state,
            agent_embed = agent_embed,
            policy_input = policy_input,
            rewards = rewards,
            terminals = terminals,
            actions = Actions(actions_discrete, actions_continuous),
            log_probs = Actions(log_probs_discrete, log_probs_continuous),
            old_action_unembeds = Actions(old_action_unembeds_discrete, old_action_unembeds_continuous),
            values = values,
            step_size = step_size,
            lens = lens,
            is_truncated = is_truncated,
            is_from_world_model = is_from_world_model,
            episode_return = episode_return,
        )

        policy_loss, value_loss, diagnostics = self.world_model.learn_from_experience(
            experience,
            only_learn_policy_value_heads = not self.full_transformer_finetuning,
            objective = self.objective,
            use_delight_gating = self.use_delight_gating,
            return_diagnostics = True,
        )

        if not self.full_transformer_finetuning:
            return policy_loss, value_loss, diagnostics

        # Dreamer 4 reports that full-transformer imagination finetuning needs
        # the dynamics, policy-prior, and reward losses to prevent collapse.
        # PMPO supplies the policy-prior loss in learn_from_experience().  Apply
        # the other two losses to real replay targets during the same step.
        assert preservation_latents is not None
        _, preservation_losses = self.world_model(
            latents = preservation_latents,
            proprio = preservation_proprio,
            rewards = preservation_rewards,
            discrete_actions = preservation_actions_discrete,
            continuous_actions = preservation_actions_continuous,
            lens = preservation_lens,
            latent_has_view_dim = preservation_latents.ndim == 5,
            return_all_losses = True,
            update_loss_ema = False,
            shortcut_train_mode = "combined",
        )

        dynamics_loss = (
            preservation_losses.flow * self.world_model.latent_flow_loss_weight +
            preservation_losses.shortcut * self.world_model.shortcut_loss_weight
        )
        reward_loss = (
            (preservation_losses.agent_embed_reward * self.world_model.reward_loss_weight).sum() +
            (preservation_losses.latent_state_reward * self.world_model.reward_loss_weight).sum()
        )
        preservation_loss = dynamics_loss + reward_loss

        diagnostics = dict(diagnostics)
        diagnostics.update({
            "preservation/dynamics_loss": dynamics_loss.detach(),
            "preservation/reward_loss": reward_loss.detach(),
            "preservation/loss": preservation_loss.detach(),
        })

        return policy_loss + preservation_loss, value_loss, diagnostics


@dataclass
class CompileRuntime:
    interact_forward: Callable
    world_model_loss: Callable
    world_model_loss_timing: CallTiming | None
    generate_rollout: Callable
    generate_rollout_timing: CallTiming | None
    learn_from_dream: Callable
    learn_from_dream_timing: CallTiming | None
    world_model_step_timing: CallTiming | None
    imagination_step_timing: CallTiming | None
    timings: list[CallTiming]


def synchronize_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_torch_compile_state(device: torch.device):
    torch.compiler.reset()

    try:
        import torch._inductor.codecache as codecache

        codecache.FxGraphCache.clear()
        codecache.PyCodeCache.cache_clear()
        codecache.CppCodeCache.cache_clear()
    except Exception:
        pass

    # dynamo/inductor state and cudagraph pools are commonly kept alive by
    # reference cycles, so collect before releasing cached blocks

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


def torch_compile_kwargs(
    *,
    backend: str,
    mode: str | None,
    fullgraph: bool,
    dynamic: bool | None,
    options: Mapping[str, Any] | None = None,
):
    kwargs = dict(backend = backend, fullgraph = fullgraph)
    mode = torch_compile_mode(mode)
    options = dict(options or {})

    # PyTorch rejects passing mode and options together. Options take
    # precedence, so the mode is deliberately omitted whenever options are
    # supplied.
    if len(options) > 0:
        kwargs["options"] = options
    elif exists(mode) and mode != "default":
        kwargs["mode"] = mode

    if dynamic is not None:
        kwargs["dynamic"] = dynamic

    return kwargs


def maybe_compile_and_time(
    fn: Callable,
    *,
    name: str,
    enabled: bool,
    track_timing: bool,
    device: torch.device,
    compile_backend: str,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool | None,
    compile_cudagraphs = True,
    compile_options: Mapping[str, Any] | None = None,
    wrap_timing = True,
):
    compile_mode = normalize_compile_mode(compile_mode)
    compile_options = dict(compile_options or {})
    uses_cuda_graphs = (
        enabled and
        device.type == "cuda" and
        compile_cudagraphs and
        (
            bool(compile_options.get("triton.cudagraphs", False))
            if len(compile_options) > 0
            else compile_mode_uses_cuda_graphs(compile_mode)
        )
    )
    effective_compile_mode = compile_mode

    if len(compile_options) > 0:
        effective_compile_mode = None
    elif enabled and compile_mode_uses_cuda_graphs(compile_mode) and not compile_cudagraphs:
        # Dynamic caches and live autograd intermediates are unsafe to replay,
        # so paths that opt out fall back to a non-CUDA-graph mode.
        effective_compile_mode = "max-autotune-no-cudagraphs" if compile_mode == "max-autotune" else "default"

    timing = CallTiming(
        name = name,
        compiled = enabled,
        compile_mode = effective_compile_mode if enabled else None,
        compile_options = tuple(f"{key}={value}" for key, value in compile_options.items()),
        cuda_graphs = uses_cuda_graphs,
        warmup_calls = 3 if uses_cuda_graphs else 1,
    )

    if enabled:
        fn = torch.compile(
            fn,
            **torch_compile_kwargs(
                backend = compile_backend,
                mode = effective_compile_mode,
                fullgraph = compile_fullgraph,
                dynamic = compile_dynamic,
                options = compile_options,
            )
        )
        fn = CompiledCallable(fn, timing)

    if track_timing and wrap_timing:
        return TimedCallable(fn, timing, device), timing

    return fn, timing


def build_compile_runtime(
    world_model: DynamicsWorldModel,
    *,
    device: torch.device,
    compile_world_model: bool,
    compile_generate: bool,
    compile_learn: bool,
    compile_backend: str,
    compile_mode: str | None,
    compile_fullgraph: bool,
    compile_dynamic: bool | None,
    compile_generate_cudagraphs: bool,
    compile_options: Mapping[str, Any] | None = None,
    optimize_generate_compilation: bool = False,
    compile_generate_with_time_cache: bool = False,
    track_compile_performance: bool,
    imagination_generate_steps: int,
    imagination_use_time_cache: bool,
    imagination_max_timesteps: int = 64,
    objective: Literal["ppo", "pmpo", "spo"],
    use_delight_gating: bool,
    store_old_action_unembeds: bool,
    compile_interact: bool = False,
    interact_max_timesteps: int = 16,
    full_transformer_finetuning: bool = False,
) -> CompileRuntime:
    timings = []

    if compile_generate_with_time_cache and optimize_generate_compilation:
        raise ValueError(
            "compile_generate_with_time_cache and optimize_generate_compilation "
            "select different generation implementations and cannot both be enabled"
        )

    if compile_generate_with_time_cache and not imagination_use_time_cache:
        raise ValueError("compile_generate_with_time_cache requires imagination_use_time_cache=true")

    interact_forward, timing = maybe_compile_and_time(
        InteractWithEnvForward(world_model),
        name = "interact_forward",
        enabled = compile_interact,
        track_timing = False,
        device = device,
        compile_backend = compile_backend,
        compile_mode = compile_mode,
        compile_fullgraph = compile_fullgraph,
        compile_dynamic = True,
        compile_cudagraphs = False,
        compile_options = compile_options,
        wrap_timing = False,
    )
    if compile_interact:
        interact_forward = DynamicTimeCacheCallable(
            interact_forward,
            max_time = interact_max_timesteps + 1,
        )
    timings.append(timing)

    world_model_loss, timing = maybe_compile_and_time(
        WorldModelTrainingLoss(world_model),
        name = "world_model_loss",
        enabled = compile_world_model,
        track_timing = track_compile_performance,
        device = device,
        compile_backend = compile_backend,
        compile_mode = compile_mode,
        compile_fullgraph = compile_fullgraph,
        compile_dynamic = compile_dynamic,
        compile_cudagraphs = False,
        compile_options = compile_options,
        wrap_timing = False,
    )
    world_model_loss_timing = timing if track_compile_performance else None
    timings.append(timing)

    cached_generate_forward = None
    if compile_generate and compile_generate_with_time_cache:
        cached_generate_forward, timing = maybe_compile_and_time(
            world_model._forward_for_generate_impl,
            name = "imagination_generate_forward_cached",
            enabled = True,
            track_timing = False,
            device = device,
            compile_backend = compile_backend,
            compile_mode = compile_mode,
            compile_fullgraph = compile_fullgraph,
            # The KV cache grows on every generated state. A dynamic inner
            # forward specializes once for the empty-cache case and then reuses
            # its cached graphs instead of compiling the Python rollout loop.
            compile_dynamic = True,
            compile_cudagraphs = False,
            compile_options = compile_options,
            wrap_timing = False,
        )
        cached_generate_forward = DynamicTimeCacheCallable(
            cached_generate_forward,
            max_time = imagination_max_timesteps,
        )
        timings.append(timing)

    generate_rollout_module = ImaginationGenerateRollout(
        world_model,
        generate_steps = imagination_generate_steps,
        store_old_action_unembeds = store_old_action_unembeds,
        use_time_cache = imagination_use_time_cache,
        preallocate_outputs = compile_generate or optimize_generate_compilation,
        optimize_generate_compilation = optimize_generate_compilation,
        forward_for_generate = cached_generate_forward,
    )

    generate_rollout, timing = maybe_compile_and_time(
        generate_rollout_module,
        name = "imagination_generate",
        enabled = (
            compile_generate and
            not optimize_generate_compilation and
            not compile_generate_with_time_cache
        ),
        track_timing = track_compile_performance,
        device = device,
        compile_backend = compile_backend,
        compile_mode = compile_mode,
        compile_fullgraph = compile_fullgraph,
        compile_dynamic = compile_dynamic,
        compile_cudagraphs = compile_generate_cudagraphs,
        compile_options = compile_options,
    )
    generate_rollout_timing = timing if track_compile_performance else None
    generate_uses_cuda_graphs = timing.cuda_graphs
    timings.append(timing)

    learn_from_dream, timing = maybe_compile_and_time(
        ImaginationLearningLoss(
            world_model,
            objective = objective,
            use_delight_gating = use_delight_gating,
            full_transformer_finetuning = full_transformer_finetuning,
        ),
        name = "imagination_learn",
        enabled = compile_learn,
        track_timing = track_compile_performance,
        device = device,
        compile_backend = compile_backend,
        compile_mode = compile_mode,
        compile_fullgraph = compile_fullgraph,
        compile_dynamic = compile_dynamic,
        compile_cudagraphs = False,
        compile_options = compile_options,
        wrap_timing = False,
    )
    learn_from_dream_timing = timing if track_compile_performance else None
    timings.append(timing)

    world_model_step_timing = CallTiming(name = "world_model_step", compiled = compile_world_model)
    imagination_step_timing = CallTiming(
        name = "imagination_step",
        compiled = compile_learn or compile_generate,
        warmup_calls = 3 if generate_uses_cuda_graphs else 1,
    )

    if track_compile_performance:
        timings.extend([world_model_step_timing, imagination_step_timing])
    else:
        world_model_step_timing = None
        imagination_step_timing = None

    return CompileRuntime(
        interact_forward = interact_forward,
        world_model_loss = world_model_loss,
        world_model_loss_timing = world_model_loss_timing,
        generate_rollout = generate_rollout,
        generate_rollout_timing = generate_rollout_timing,
        learn_from_dream = learn_from_dream,
        learn_from_dream_timing = learn_from_dream_timing,
        world_model_step_timing = world_model_step_timing,
        imagination_step_timing = imagination_step_timing,
        timings = timings,
    )


def print_compile_timing_report(timings: list[CallTiming]):
    active_timings = [timing for timing in timings if timing.calls > 0]

    if len(active_timings) == 0:
        return

    print("\ncompile performance accounting:")

    for timing in active_timings:
        first = timing.first_call_seconds or 0.
        warm_avg = timing.warm_average_seconds
        warm_avg_text = f"{warm_avg:.4f}s" if exists(warm_avg) else "n/a"
        compiled_text = "compiled" if timing.compiled else "eager"
        mode_text = timing.compile_mode or ("options" if timing.compile_options else ("compiled" if timing.compiled else "eager"))
        options_text = f", options={','.join(timing.compile_options)}" if len(timing.compile_options) > 0 else ""

        print(
            f"  {timing.name} ({compiled_text}): "
            f"calls={timing.calls}, first_call={first:.4f}s, "
            f"warm_total={timing.warm_seconds:.4f}s, warm_avg={warm_avg_text}, "
            f"total={timing.total_seconds:.4f}s, "
            f"mode={mode_text}{options_text}, cuda_graphs={timing.cuda_graphs}"
        )


def record_timed_block(timing: CallTiming | None, device: torch.device, fn: Callable):
    if not exists(timing):
        return fn()

    synchronize_if_cuda(device)
    start = perf_counter()
    out = fn()
    synchronize_if_cuda(device)
    timing.record(perf_counter() - start)
    return out


def action_pair_or_empty(actions):
    if not exists(actions):
        return Actions(None, None)

    if isinstance(actions, Actions):
        return actions

    return Actions(*actions)
