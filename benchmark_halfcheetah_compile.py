# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "tqdm",
#     "dreamer4",
#     "adam-atan2-pytorch>=0.3.6",
#     "torch-einops-utils",
#     "fire",
#     "gymnasium[mujoco]",
#     "tensorboard",
# ]
# [tool.uv.sources]
# dreamer4 = { path = "." }
# ///

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from time import perf_counter
from typing import Literal

import fire
import torch

from compile_runtime import reset_torch_compile_state, synchronize_if_cuda
from dreamer4.dreamer4 import exists
from train_halfcheetah_imagination_rl import main as train_halfcheetah


BENCHMARK_ARMS = (
    ("eager", False, None),
    ("compiled_default", True, "default"),
    ("compiled_reduce_overhead_generated_graphs", True, "reduce-overhead"),
    ("compiled_max_autotune_generated_graphs", True, "max-autotune"),
)


@dataclass
class BenchmarkResult:
    name: str
    wall_seconds: float
    hot_path_seconds: float
    cold_seconds: float
    warm_seconds: float
    warm_average_seconds: float | None
    # sum of per-timing warm averages - warm cost of one world-model step plus one
    # imagination step, comparable across arms even when warmup call counts differ
    warm_step_seconds: float | None
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None


def train_defaults():
    return {
        name: param.default
        for name, param in signature(train_halfcheetah).parameters.items()
        if param.default is not Parameter.empty
    }


def benchmark_common_kwargs(
    train_kwargs: dict,
    *,
    benchmark_num_loops: int,
    benchmark_num_envs: int,
    benchmark_max_timesteps: int,
    benchmark_require_cuda: bool,
    benchmark_preset: Literal["smoke", "perf"],
):
    if benchmark_preset not in ("smoke", "perf"):
        raise ValueError("benchmark_preset must be 'smoke' or 'perf'")

    common_kwargs = train_defaults()
    common_kwargs.update(train_kwargs)

    smoke = benchmark_preset == "smoke"

    pinned_kwargs = dict(
        num_loops = benchmark_num_loops,
        rollouts_per_loop = 1,
        num_envs = benchmark_num_envs,
        max_timesteps = benchmark_max_timesteps,
        replay_size = 4,
        time_block_every = 1,
        imagination_use_time_cache = False,
        tokenizer_eval_every = 0,
        use_muon_optimizer = False,
        use_tensorboard = False,
        checkpoint_every = 0,
        checkpoint_path = None,
        clear_log_dir = True,
        unique_log_dir = False,
        track_compile_performance = True,
        compile_dynamic = False,
        require_cuda = benchmark_require_cuda,
        prob_shortcut_train = 1.,
    )

    conflicting = sorted(
        key
        for key in pinned_kwargs.keys() & train_kwargs.keys()
        if train_kwargs[key] != pinned_kwargs[key]
    )

    if conflicting:
        raise ValueError(
            f"these kwargs are pinned by the benchmark and cannot be overridden: {conflicting} "
            "(use the benchmark_* flags where available)"
        )

    benchmark_kwargs = dict(pinned_kwargs)

    # rollouts only ever produce max_timesteps frames, so longer train windows are pure padding

    benchmark_kwargs.update(
        world_model_train_sequence_length = min(int(common_kwargs["world_model_train_sequence_length"]), benchmark_max_timesteps),
    )

    if smoke:
        benchmark_kwargs.update(
            model_dim = min(int(common_kwargs["model_dim"]), 32),
            depth = min(int(common_kwargs["depth"]), 1),
            world_model_batch_size = 1,
            world_model_train_steps = max(2, min(int(common_kwargs["world_model_train_steps"]), 2)),
            imagination_batch_size = 1,
            imagination_horizon = min(int(common_kwargs["imagination_horizon"]), 4),
            imagination_prompt_length = min(int(common_kwargs["imagination_prompt_length"]), 2),
            imagination_train_steps = max(2, min(int(common_kwargs["imagination_train_steps"]), 2)),
            imagination_generate_steps = 2,
            pretrain_tokenizer_steps = max(1, min(int(common_kwargs["pretrain_tokenizer_steps"]), 1)),
            pretrain_tokenizer_observations = max(32, benchmark_num_envs * benchmark_max_timesteps),
            tokenizer_batch_size = min(int(common_kwargs["tokenizer_batch_size"]), 16),
        )
    else:
        # Keep the performance preset on full, non-repeated batches. The training
        # path can repeat short batches for static compilation, but benchmarking
        # duplicated episodes would distort throughput comparisons.

        max_static_batch = max(benchmark_num_envs, 1)
        world_model_batch_size = int(common_kwargs["world_model_batch_size"])
        imagination_batch_size = int(common_kwargs["imagination_batch_size"])

        if max(world_model_batch_size, imagination_batch_size) > max_static_batch:
            print(
                f"benchmark: clamping batch sizes to {max_static_batch} so the initial replay "
                "provides full batches (raise --benchmark_num_envs for larger batches)"
            )

        benchmark_kwargs.update(
            world_model_batch_size = min(world_model_batch_size, max_static_batch),
            imagination_batch_size = min(imagination_batch_size, max_static_batch),
        )

    common_kwargs.update(benchmark_kwargs)

    return common_kwargs


def run_benchmark_arm(
    name: str,
    *,
    common_kwargs: dict,
    benchmark_root: Path,
    compile_enabled: bool,
    compile_mode: str | None,
):
    device = torch.device("cpu" if common_kwargs.get("cpu") or not torch.cuda.is_available() else "cuda")
    reset_torch_compile_state(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    run_kwargs = dict(common_kwargs)
    run_kwargs.update(
        compile = compile_enabled,
        compile_world_model = False,
        compile_generate = False,
        compile_learn = False,
        compile_mode = compile_mode,
        log_dir = str(benchmark_root / name),
        return_compile_timings = True,
    )

    print(f"\nbenchmark {name}:")
    start = perf_counter()
    timings = train_halfcheetah(**run_kwargs)
    synchronize_if_cuda(device)
    wall_seconds = perf_counter() - start

    step_timings = [
        timing
        for timing in timings
        if timing.name in ("world_model_step", "imagination_step")
    ]

    hot_path_seconds = sum(timing.total_seconds for timing in step_timings)
    cold_seconds = sum(timing.cold_seconds for timing in step_timings)
    warm_seconds = sum(timing.warm_seconds for timing in step_timings)
    warm_calls = sum(timing.warm_calls for timing in step_timings)
    warm_average_seconds = warm_seconds / warm_calls if warm_calls > 0 else None

    warm_step_seconds = None
    if len(step_timings) > 0 and all(exists(timing.warm_average_seconds) for timing in step_timings):
        warm_step_seconds = sum(timing.warm_average_seconds for timing in step_timings)

    peak_allocated_mib = peak_reserved_mib = None
    if device.type == "cuda":
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 2 ** 20
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2 ** 20

    result = BenchmarkResult(
        name = name,
        wall_seconds = wall_seconds,
        hot_path_seconds = hot_path_seconds,
        cold_seconds = cold_seconds,
        warm_seconds = warm_seconds,
        warm_average_seconds = warm_average_seconds,
        warm_step_seconds = warm_step_seconds,
        peak_allocated_mib = peak_allocated_mib,
        peak_reserved_mib = peak_reserved_mib,
    )

    print(f"benchmark {name} wall time: {wall_seconds:.3f}s")
    print(f"benchmark {name} measured hot-path time: {hot_path_seconds:.3f}s")
    print(f"benchmark {name} measured cold time (compile/capture warmup): {cold_seconds:.3f}s")

    if exists(warm_average_seconds):
        print(f"benchmark {name} measured warm time: {warm_seconds:.3f}s total, {warm_average_seconds:.3f}s avg")

    if exists(warm_step_seconds):
        print(f"benchmark {name} warm per-step time (world model + imagination): {warm_step_seconds:.3f}s")

    if exists(peak_allocated_mib) and exists(peak_reserved_mib):
        print(f"benchmark {name} peak CUDA memory: {peak_allocated_mib:.1f} MiB allocated, {peak_reserved_mib:.1f} MiB reserved")

    return result


def print_benchmark_summary(results: list[BenchmarkResult]):
    eager = next(result for result in results if result.name == "eager")
    eager_warm_step = eager.warm_step_seconds

    print("\nbenchmark summary:")
    eager_warm_step_text = f"{eager_warm_step:.3f}s" if exists(eager_warm_step) else "n/a"
    print(f"  eager warm runtime: {eager.warm_seconds:.3f}s total, {eager_warm_step_text} per step")

    if exists(eager.peak_allocated_mib) and exists(eager.peak_reserved_mib):
        print(f"  eager peak CUDA memory: {eager.peak_allocated_mib:.1f} MiB allocated, {eager.peak_reserved_mib:.1f} MiB reserved")

    for compiled in (result for result in results if result.name != "eager"):
        warm_speedup = None
        if exists(eager_warm_step) and exists(compiled.warm_step_seconds) and compiled.warm_step_seconds > 0:
            warm_speedup = eager_warm_step / compiled.warm_step_seconds

        compile_overhead = max(compiled.cold_seconds - eager.cold_seconds, 0.)

        break_even_steps = None
        if (
            exists(eager_warm_step) and
            exists(compiled.warm_step_seconds) and
            compiled.warm_step_seconds < eager_warm_step
        ):
            break_even_steps = compile_overhead / (eager_warm_step - compiled.warm_step_seconds)

        print(f"\n  {compiled.name}:")
        warm_speedup_text = f"{warm_speedup:.3f}x" if exists(warm_speedup) else "n/a"
        print(f"    steady-state warm per-step speedup: {warm_speedup_text}")
        compiled_warm_step_text = f"{compiled.warm_step_seconds:.3f}s" if exists(compiled.warm_step_seconds) else "n/a"
        print(f"    compiled warm runtime: {compiled.warm_seconds:.3f}s total, {compiled_warm_step_text} per step")
        print(f"    compiled cold time (compile/capture warmup): {compiled.cold_seconds:.3f}s")
        print(f"    estimated compile overhead vs eager cold calls: {compile_overhead:.3f}s")

        if exists(compiled.peak_allocated_mib) and exists(compiled.peak_reserved_mib):
            print(f"    peak CUDA memory: {compiled.peak_allocated_mib:.1f} MiB allocated, {compiled.peak_reserved_mib:.1f} MiB reserved")

        if exists(break_even_steps):
            print(f"    estimated break-even warm steps: {break_even_steps:.1f}")
        else:
            print("    estimated break-even warm steps: never (compiled warm runtime is not faster)")

        print(f"    total elapsed, including setup and compile: {compiled.wall_seconds:.3f}s")
        print(f"    measured hot-path elapsed, including compile: {compiled.hot_path_seconds:.3f}s")

    print(f"\n  total eager elapsed, including setup: {eager.wall_seconds:.3f}s")
    print(f"  measured hot-path elapsed, eager: {eager.hot_path_seconds:.3f}s")


def main(
    benchmark_num_loops = 2,
    benchmark_num_envs = 1,
    benchmark_max_timesteps = 8,
    benchmark_require_cuda = True,
    benchmark_preset: Literal["smoke", "perf"] = "smoke",
    **train_kwargs,
):
    common_kwargs = benchmark_common_kwargs(
        train_kwargs,
        benchmark_num_loops = benchmark_num_loops,
        benchmark_num_envs = benchmark_num_envs,
        benchmark_max_timesteps = benchmark_max_timesteps,
        benchmark_require_cuda = benchmark_require_cuda,
        benchmark_preset = benchmark_preset,
    )

    benchmark_root = Path(common_kwargs["log_dir"]) / "compile_benchmark"
    results = [
        run_benchmark_arm(
            name,
            common_kwargs = common_kwargs,
            benchmark_root = benchmark_root,
            compile_enabled = compile_enabled,
            compile_mode = compile_mode,
        )
        for name, compile_enabled, compile_mode in BENCHMARK_ARMS
    ]

    print_benchmark_summary(results)


if __name__ == "__main__":
    fire.Fire(main)
