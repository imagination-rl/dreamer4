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

from train_halfcheetah_imagination_rl import (
    exists,
    main as train_halfcheetah,
    reset_torch_compile_state,
    synchronize_if_cuda,
)


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
    first_call_seconds: float
    warm_seconds: float
    warm_average_seconds: float | None
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

    common_kwargs.update(
        num_loops = benchmark_num_loops,
        rollouts_per_loop = 1,
        num_envs = benchmark_num_envs,
        max_timesteps = benchmark_max_timesteps,
        replay_size = 4,
        model_dim = min(int(common_kwargs["model_dim"]), 32) if smoke else common_kwargs["model_dim"],
        depth = min(int(common_kwargs["depth"]), 1) if smoke else common_kwargs["depth"],
        time_block_every = 1,
        world_model_batch_size = 1 if smoke else common_kwargs["world_model_batch_size"],
        world_model_train_steps = max(2, min(int(common_kwargs["world_model_train_steps"]), 2)),
        world_model_train_sequence_length = min(int(common_kwargs["world_model_train_sequence_length"]), benchmark_max_timesteps) if smoke else common_kwargs["world_model_train_sequence_length"],
        imagination_batch_size = 1 if smoke else common_kwargs["imagination_batch_size"],
        imagination_horizon = min(int(common_kwargs["imagination_horizon"]), 4) if smoke else common_kwargs["imagination_horizon"],
        imagination_prompt_length = min(int(common_kwargs["imagination_prompt_length"]), 2),
        imagination_train_steps = max(2, min(int(common_kwargs["imagination_train_steps"]), 2)),
        imagination_generate_steps = 2,
        imagination_use_time_cache = False,
        pretrain_tokenizer_steps = max(1, min(int(common_kwargs["pretrain_tokenizer_steps"]), 1)) if smoke else common_kwargs["pretrain_tokenizer_steps"],
        pretrain_tokenizer_observations = max(32, benchmark_num_envs * benchmark_max_timesteps) if smoke else common_kwargs["pretrain_tokenizer_observations"],
        tokenizer_batch_size = min(int(common_kwargs["tokenizer_batch_size"]), 16) if smoke else common_kwargs["tokenizer_batch_size"],
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
    first_call_seconds = sum(timing.first_call_seconds or 0. for timing in step_timings)
    warm_seconds = sum(timing.warm_seconds for timing in step_timings)
    warm_calls = sum(timing.warm_calls for timing in step_timings)
    warm_average_seconds = warm_seconds / warm_calls if warm_calls > 0 else None

    peak_allocated_mib = peak_reserved_mib = None
    if device.type == "cuda":
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 2 ** 20
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2 ** 20

    result = BenchmarkResult(
        name = name,
        wall_seconds = wall_seconds,
        hot_path_seconds = hot_path_seconds,
        first_call_seconds = first_call_seconds,
        warm_seconds = warm_seconds,
        warm_average_seconds = warm_average_seconds,
        peak_allocated_mib = peak_allocated_mib,
        peak_reserved_mib = peak_reserved_mib,
    )

    print(f"benchmark {name} wall time: {wall_seconds:.3f}s")
    print(f"benchmark {name} measured hot-path time: {hot_path_seconds:.3f}s")
    print(f"benchmark {name} measured first-call time: {first_call_seconds:.3f}s")

    if exists(warm_average_seconds):
        print(f"benchmark {name} measured warm time: {warm_seconds:.3f}s total, {warm_average_seconds:.3f}s avg")

    if exists(peak_allocated_mib) and exists(peak_reserved_mib):
        print(f"benchmark {name} peak CUDA memory: {peak_allocated_mib:.1f} MiB allocated, {peak_reserved_mib:.1f} MiB reserved")

    return result


def print_benchmark_summary(results: list[BenchmarkResult]):
    eager = next(result for result in results if result.name == "eager")
    eager_warm_average = eager.warm_average_seconds

    print("\nbenchmark summary:")
    eager_warm_average_text = f"{eager_warm_average:.3f}s" if exists(eager_warm_average) else "n/a"
    print(f"  eager warm runtime: {eager.warm_seconds:.3f}s total, {eager_warm_average_text} avg")

    if exists(eager.peak_allocated_mib) and exists(eager.peak_reserved_mib):
        print(f"  eager peak CUDA memory: {eager.peak_allocated_mib:.1f} MiB allocated, {eager.peak_reserved_mib:.1f} MiB reserved")

    for compiled in (result for result in results if result.name != "eager"):
        warm_speedup = eager.warm_seconds / compiled.warm_seconds if compiled.warm_seconds > 0 else None
        compile_overhead = max(compiled.first_call_seconds - eager.first_call_seconds, 0.)

        break_even_calls = None
        if (
            exists(eager_warm_average) and
            exists(compiled.warm_average_seconds) and
            compiled.warm_average_seconds < eager_warm_average
        ):
            break_even_calls = compile_overhead / (eager_warm_average - compiled.warm_average_seconds)

        print(f"\n  {compiled.name}:")
        warm_speedup_text = f"{warm_speedup:.3f}x" if exists(warm_speedup) else "n/a"
        print(f"    steady-state warm runtime speedup: {warm_speedup_text}")
        compiled_warm_average_text = f"{compiled.warm_average_seconds:.3f}s" if exists(compiled.warm_average_seconds) else "n/a"
        print(f"    compiled warm runtime: {compiled.warm_seconds:.3f}s total, {compiled_warm_average_text} avg")
        print(f"    compiled first-call time: {compiled.first_call_seconds:.3f}s")
        print(f"    estimated compile overhead vs eager first call: {compile_overhead:.3f}s")

        if exists(compiled.peak_allocated_mib) and exists(compiled.peak_reserved_mib):
            print(f"    peak CUDA memory: {compiled.peak_allocated_mib:.1f} MiB allocated, {compiled.peak_reserved_mib:.1f} MiB reserved")

        if exists(break_even_calls):
            print(f"    estimated break-even warm calls: {break_even_calls:.1f}")
        else:
            print("    estimated break-even warm calls: never (compiled warm runtime is not faster)")

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
