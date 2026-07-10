"""torch.compile wrappers and timing instrumentation shared by the HalfCheetah
train and benchmark scripts."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Literal

import torch
from torch import Tensor, nn

from dreamer4.dreamer4 import (
    Actions,
    DynamicsWorldModel,
    Experience,
    exists,
)

@dataclass
class CallTiming:
    name: str
    compiled: bool
    compile_mode: str | None = None
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
    ):
        return self.world_model(
            latents = latents,
            rewards = rewards,
            terminals = terminals,
            continuous_actions = continuous_actions,
            lens = lens,
            latent_has_view_dim = latents.ndim == 5,
            return_all_losses = True,
            update_loss_ema = True,
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
    ):
        super().__init__()
        self.world_model = world_model
        self.generate_steps = generate_steps
        self.store_old_action_unembeds = store_old_action_unembeds
        self.use_time_cache = use_time_cache
        self.preallocate_outputs = preallocate_outputs

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
    ):
        super().__init__()
        self.world_model = world_model
        self.objective = objective
        self.use_delight_gating = use_delight_gating

    def forward(
        self,
        latents: Tensor,
        proprio: Tensor | None,
        critic_state: Tensor | None,
        agent_embed: Tensor | None,
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
    ):
        experience = Experience(
            latents = latents,
            proprio = proprio,
            critic_state = critic_state,
            agent_embed = agent_embed,
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

        return self.world_model.learn_from_experience(
            experience,
            only_learn_policy_value_heads = True,
            objective = self.objective,
            use_delight_gating = self.use_delight_gating,
            return_diagnostics = True,
        )


@dataclass
class CompileRuntime:
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
):
    kwargs = dict(backend = backend, fullgraph = fullgraph)
    mode = torch_compile_mode(mode)

    if exists(mode) and mode != "default":
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
    wrap_timing = True,
):
    compile_mode = normalize_compile_mode(compile_mode)
    uses_cuda_graphs = (
        enabled and
        device.type == "cuda" and
        compile_cudagraphs and
        compile_mode_uses_cuda_graphs(compile_mode)
    )
    effective_compile_mode = compile_mode

    if enabled and compile_mode_uses_cuda_graphs(compile_mode) and not compile_cudagraphs:
        # AOTAutograd keeps forward intermediates live for backward, and CUDA
        # Graph replay can overwrite them before compiled loss wrappers finish.
        effective_compile_mode = "max-autotune-no-cudagraphs" if compile_mode == "max-autotune" else "default"

    timing = CallTiming(
        name = name,
        compiled = enabled,
        compile_mode = effective_compile_mode if enabled else None,
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
    track_compile_performance: bool,
    imagination_generate_steps: int,
    imagination_use_time_cache: bool,
    objective: Literal["ppo", "pmpo", "spo"],
    use_delight_gating: bool,
    store_old_action_unembeds: bool,
) -> CompileRuntime:
    timings = []

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
        wrap_timing = False,
    )
    world_model_loss_timing = timing if track_compile_performance else None
    timings.append(timing)

    generate_rollout_module = ImaginationGenerateRollout(
        world_model,
        generate_steps = imagination_generate_steps,
        store_old_action_unembeds = store_old_action_unembeds,
        use_time_cache = imagination_use_time_cache,
        preallocate_outputs = compile_generate,
    )

    generate_rollout, timing = maybe_compile_and_time(
        generate_rollout_module,
        name = "imagination_generate",
        enabled = compile_generate,
        track_timing = track_compile_performance,
        device = device,
        compile_backend = compile_backend,
        compile_mode = compile_mode,
        compile_fullgraph = compile_fullgraph,
        compile_dynamic = compile_dynamic,
        compile_cudagraphs = compile_generate_cudagraphs,
    )
    generate_rollout_timing = timing if track_compile_performance else None
    generate_uses_cuda_graphs = timing.cuda_graphs
    timings.append(timing)

    learn_from_dream, timing = maybe_compile_and_time(
        ImaginationLearningLoss(
            world_model,
            objective = objective,
            use_delight_gating = use_delight_gating,
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
        mode_text = timing.compile_mode or ("compiled" if timing.compiled else "eager")

        print(
            f"  {timing.name} ({compiled_text}): "
            f"calls={timing.calls}, first_call={first:.4f}s, "
            f"warm_total={timing.warm_seconds:.4f}s, warm_avg={warm_avg_text}, "
            f"total={timing.total_seconds:.4f}s, "
            f"mode={mode_text}, cuda_graphs={timing.cuda_graphs}"
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
