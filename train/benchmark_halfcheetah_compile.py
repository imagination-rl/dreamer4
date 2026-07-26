# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "pyyaml",
# ]
# ///

"""Benchmark the canonical Gym training pipeline across compile configurations.

The benchmark deliberately uses the configuration in this directory and runs
exactly three complete training loops for every case.  SAC means selective
activation recomputation here; it is controlled by ``activation_memory_budget``
in ``train_gym.main``.

Examples::

    uv run python train/benchmark_halfcheetah_compile.py
    uv run python train/benchmark_halfcheetah_compile.py \
        --benchmark_cases=eager,compiled_max_autotune \
        --sac_values=0.0,0.5,0.9
    uv run python train/benchmark_halfcheetah_compile.py \
        --benchmark_cases=custom

The three default benchmark cases can be selected by their canonical names or
by the shorter aliases ``eager``, ``default``, and ``max-autotune``.  The
optional ``custom`` case expands into one benchmark case per custom performance
option defined below; each case enables exactly one option.
"""

from __future__ import annotations

import math
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from inspect import signature
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import fire
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
BENCHMARK_NUM_LOOPS = 3
BENCHMARK_TOKENIZER_STEPS = 10
BENCHMARK_FINETUNE_STEPS = 0
BENCHMARK_COMBINED_PRETRAIN_STEPS = 0
CUSTOM_PERFORMANCE_OPTIONS = {
    "cpp_wrapper": True,
    "max_autotune_gemm": True,
    "b2b_gemm_pass": True,
    "triton.enable_persistent_tma_matmul": True,
    "epilogue_fusion_first": True,
}


def _case_key(value: str):
    value = str(value).strip().lower()
    return re.sub(r"[\s.\-]+", "_", value)


@dataclass(frozen = True)
class BenchmarkCase:
    name: str
    compile_enabled: bool
    compile_mode: str | None
    compile_generate_cudagraphs: bool
    compile_options: Mapping[str, Any] | None = None


BENCHMARK_CASES = (
    BenchmarkCase(
        name = "eager",
        compile_enabled = False,
        compile_mode = None,
        compile_generate_cudagraphs = False,
    ),
    BenchmarkCase(
        name = "compiled_default",
        compile_enabled = True,
        compile_mode = "default",
        compile_generate_cudagraphs = True,
    ),
    BenchmarkCase(
        name = "compiled_max_autotune",
        compile_enabled = True,
        compile_mode = "max-autotune",
        compile_generate_cudagraphs = True,
    ),
)

CUSTOM_BENCHMARK_CASES = tuple(
    BenchmarkCase(
        name = f"compiled_{_case_key(option)}",
        compile_enabled = True,
        compile_mode = None,
        compile_generate_cudagraphs = False,
        compile_options = {option: value},
    )
    for option, value in CUSTOM_PERFORMANCE_OPTIONS.items()
)

OPTIONAL_BENCHMARK_CASES = CUSTOM_BENCHMARK_CASES
ALL_BENCHMARK_CASES = BENCHMARK_CASES + CUSTOM_BENCHMARK_CASES
_CASE_BY_NAME = {case.name: case for case in ALL_BENCHMARK_CASES}
_CASE_BY_KEY = {_case_key(case.name): case for case in ALL_BENCHMARK_CASES}
_CUSTOM_CASE_BY_KEY = {
    _case_key(option): case
    for option, case in zip(CUSTOM_PERFORMANCE_OPTIONS, CUSTOM_BENCHMARK_CASES)
}


@dataclass(frozen = True)
class BenchmarkRun:
    case: BenchmarkCase
    sac_value: float
    run_name: str
    train_kwargs: dict[str, Any]


@dataclass
class BenchmarkResult:
    name: str
    sac_value: float
    wall_seconds: float
    hot_path_seconds: float
    cold_seconds: float
    warm_seconds: float
    warm_average_seconds: float | None
    warm_step_seconds: float | None
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None
    compile_modes: tuple[str, ...]


def load_benchmark_config(config_path: str | Path = DEFAULT_CONFIG_PATH):
    config_path = Path(config_path)

    with config_path.open("r", encoding = "utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, Mapping):
        raise TypeError(f"{config_path} must contain a top-level mapping")

    config = dict(config)

    for required_key in ("run_name", "run_details", "log_dir"):
        if required_key not in config:
            raise KeyError(f"{config_path} must define {required_key}")

    return config


def _items(value: Any):
    if value is None:
        return None

    if isinstance(value, str):
        return [item for item in re.split(r"[\s,]+", value.strip()) if item]

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)

    return [value]


def select_benchmark_cases(values: str | Sequence[str] | None = None):
    if values is None:
        return BENCHMARK_CASES

    selected = []
    aliases = {
        "eager": "eager",
        "compiled_default": "compiled_default",
        "default": "compiled_default",
        "torch_compile_default": "compiled_default",
        "compiled_max_autotune": "compiled_max_autotune",
        "max_autotune": "compiled_max_autotune",
    }

    custom_group_keys = {
        "compiled_custom_options",
        "custom",
        "custom_options",
        "custom_performance",
        "custom_performance_options",
    }

    for raw_value in _items(values):
        key = _case_key(raw_value)

        if key == "all":
            selected.extend(ALL_BENCHMARK_CASES)
            continue

        if key in custom_group_keys:
            selected.extend(CUSTOM_BENCHMARK_CASES)
            continue

        case = _CASE_BY_KEY.get(key) or _CUSTOM_CASE_BY_KEY.get(key)
        if case is not None:
            selected.append(case)
            continue

        canonical_name = aliases.get(key)
        if canonical_name is None:
            available = ", ".join(case.name for case in ALL_BENCHMARK_CASES)
            raise ValueError(f"unknown benchmark case {raw_value!r}; choose from {available}")

        selected.append(_CASE_BY_NAME[canonical_name])

    unique = []
    seen = set()

    for case in selected:
        if case.name in seen:
            raise ValueError(f"benchmark case {case.name!r} was selected more than once")

        seen.add(case.name)
        unique.append(case)

    if len(unique) == 0:
        raise ValueError("at least one benchmark case must be selected")

    return tuple(unique)


def parse_sac_values(
    values: str | float | Sequence[float] | None,
    *,
    default: float,
):
    raw_values = [default] if values is None else _items(values)

    if len(raw_values) == 0:
        raise ValueError("at least one SAC value must be provided")

    parsed = []

    for raw_value in raw_values:
        if isinstance(raw_value, bool):
            raise TypeError("SAC values must be numbers between 0 and 1")

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid SAC value {raw_value!r}") from error

        if not math.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError(f"SAC value must be finite and between 0 and 1, got {raw_value!r}")

        if value in parsed:
            raise ValueError(f"SAC value {value} was selected more than once")

        parsed.append(value)

    return tuple(parsed)


def _sac_slug(value: float):
    formatted = f"{value:.8g}"
    return formatted.replace("-", "neg").replace(".", "p")


def _validate_train_overrides(
    train_overrides: Mapping[str, Any],
    *,
    valid_keys: set[str] | None = None,
):
    if valid_keys is not None:
        unknown_keys = sorted(set(train_overrides) - valid_keys)
        if unknown_keys:
            raise KeyError(f"unknown train_gym settings: {unknown_keys}")

    if "run_name" in train_overrides or "run_details" in train_overrides:
        raise ValueError("run_name and run_details are controlled by the benchmark")

    compile_controlled_keys = {
        "compile",
        "compile_interact",
        "compile_world_model",
        "compile_generate",
        "compile_learn",
        "compile_mode",
        "compile_generate_cudagraphs",
        "compile_options",
    }
    overridden_compile_keys = sorted(compile_controlled_keys & set(train_overrides))
    if overridden_compile_keys:
        raise ValueError(
            f"compile settings are selected with benchmark_cases, not train overrides: "
            f"{overridden_compile_keys}"
        )

    if "num_loops" in train_overrides and int(train_overrides["num_loops"]) != BENCHMARK_NUM_LOOPS:
        raise ValueError(f"every benchmark test must use exactly {BENCHMARK_NUM_LOOPS} loops")


def _apply_pinned_settings(
    config: Mapping[str, Any],
    train_overrides: Mapping[str, Any],
):
    # These settings make each arm a three-loop, independently timed benchmark
    # while preserving the configured generation strategy.
    pinned = {
        "num_loops": BENCHMARK_NUM_LOOPS,
        "pretrain_world_model_finetune_steps": BENCHMARK_FINETUNE_STEPS,
        "pretrain_world_model_combined_steps": BENCHMARK_COMBINED_PRETRAIN_STEPS,
        "pretrain_tokenizer_steps": BENCHMARK_TOKENIZER_STEPS,
        "rollouts_per_loop": 1,
        "imagination_use_time_cache": True,
        "checkpoint_folder": None,
        "checkpoint_every": 0,
        "checkpoint_path": None,
        "clear_log_dir": False,
        "unique_log_dir": False,
        "use_tensorboard": False,
        "compile_fullgraph": False,
        "compile_dynamic": False,
        "optimize_generate_compilation": False,
        "compile_generate_with_time_cache": True,
        "track_compile_performance": True,
        "return_compile_timings": True,
        "collect_timings": True,
        "profile": False,
        "profile_memory": False,
        "profile_record_shapes": False,
        "profile_with_stack": False,
        "profile_with_flops": False,
    }

    for key, value in pinned.items():
        if key in train_overrides and train_overrides[key] != value:
            raise ValueError(f"{key} is pinned to {value!r} by the benchmark")

    benchmark_kwargs = dict(config)
    benchmark_kwargs.update(train_overrides)
    benchmark_kwargs.update(pinned)

    # Training windows longer than an episode only add padding to this test.
    benchmark_kwargs["world_model_train_sequence_length"] = min(
        int(benchmark_kwargs["world_model_train_sequence_length"]),
        int(benchmark_kwargs["max_timesteps"]),
    )

    # Keep compiled batch shapes stable while retaining the configured number
    # of vectorized environments and full training-loop counts.
    max_static_batch = max(int(benchmark_kwargs["num_envs"]), 1)
    benchmark_kwargs["world_model_batch_size"] = min(
        int(benchmark_kwargs["world_model_batch_size"]),
        max_static_batch,
    )
    benchmark_kwargs["imagination_batch_size"] = min(
        int(benchmark_kwargs["imagination_batch_size"]),
        max_static_batch,
    )

    return benchmark_kwargs


def build_benchmark_runs(
    config: Mapping[str, Any],
    *,
    benchmark_root: str | Path,
    benchmark_cases: str | Sequence[str] | None = None,
    sac_values: str | float | Sequence[float] | None = None,
    train_overrides: Mapping[str, Any] | None = None,
    valid_train_keys: set[str] | None = None,
):
    train_overrides = {} if train_overrides is None else dict(train_overrides)
    _validate_train_overrides(train_overrides, valid_keys = valid_train_keys)

    common_kwargs = _apply_pinned_settings(config, train_overrides)
    cases = select_benchmark_cases(benchmark_cases)
    sac_default = float(common_kwargs.get("activation_memory_budget", 0.9))
    parsed_sac_values = parse_sac_values(sac_values, default = sac_default)
    include_sac_in_name = sac_values is not None
    benchmark_root = Path(benchmark_root)

    runs = []

    for case in cases:
        for sac_value in parsed_sac_values:
            run_name = case.name
            if include_sac_in_name:
                run_name = f"{run_name}__sac_{_sac_slug(sac_value)}"

            run_kwargs = dict(common_kwargs)
            run_kwargs.update(
                compile = case.compile_enabled,
                compile_interact = case.compile_enabled,
                compile_world_model = case.compile_enabled,
                compile_generate = case.compile_enabled,
                compile_learn = case.compile_enabled,
                compile_mode = case.compile_mode,
                compile_generate_cudagraphs = case.compile_generate_cudagraphs,
                compile_options = case.compile_options,
                activation_memory_budget = sac_value,
                log_dir = str(benchmark_root),
                run_name = run_name,
                run_details = (
                    f"Canonical HalfCheetah compile benchmark: {case.name}. "
                    f"SAC activation_memory_budget={sac_value}. "
                    f"Exactly {BENCHMARK_NUM_LOOPS} full training loops."
                ),
            )

            runs.append(
                BenchmarkRun(
                    case = case,
                    sac_value = sac_value,
                    run_name = run_name,
                    train_kwargs = run_kwargs,
                )
            )

    return tuple(runs)


def _load_train_main():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from train.train_gym import main

    return main


def _load_compile_runtime():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from train.compile_runtime import reset_torch_compile_state, synchronize_if_cuda

    return reset_torch_compile_state, synchronize_if_cuda


def _timing_metrics(timings):
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
    if len(step_timings) > 0 and all(timing.warm_average_seconds is not None for timing in step_timings):
        warm_step_seconds = sum(timing.warm_average_seconds for timing in step_timings)

    return (
        hot_path_seconds,
        cold_seconds,
        warm_seconds,
        warm_average_seconds,
        warm_step_seconds,
    )


def run_benchmark_run(
    benchmark_run: BenchmarkRun,
    *,
    train_main: Callable[..., Any] | None = None,
):
    import torch

    reset_torch_compile_state, synchronize_if_cuda = _load_compile_runtime()
    train_main = _load_train_main() if train_main is None else train_main
    train_kwargs = benchmark_run.train_kwargs

    if int(train_kwargs["num_loops"]) != BENCHMARK_NUM_LOOPS:
        raise AssertionError("benchmark runs must execute exactly three full loops")

    device = torch.device(
        "cpu"
        if train_kwargs.get("cpu") or not torch.cuda.is_available()
        else "cuda"
    )
    reset_torch_compile_state(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"\nbenchmark {benchmark_run.run_name}:")
    start = perf_counter()
    timings = train_main(**dict(train_kwargs))
    synchronize_if_cuda(device)
    wall_seconds = perf_counter() - start

    (
        hot_path_seconds,
        cold_seconds,
        warm_seconds,
        warm_average_seconds,
        warm_step_seconds,
    ) = _timing_metrics(timings)

    peak_allocated_mib = peak_reserved_mib = None
    if device.type == "cuda":
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 2 ** 20
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 2 ** 20

    compile_modes = tuple(
        f"{timing.name}={timing.compile_mode or ('options' if timing.compile_options else ('eager' if not timing.compiled else 'compiled'))}"
        f"{', options=' + ','.join(timing.compile_options) if timing.compile_options else ''}"
        f",cuda_graphs={timing.cuda_graphs}"
        for timing in timings
        if timing.compiled or timing.name in ("world_model_step", "imagination_step")
    )

    result = BenchmarkResult(
        name = benchmark_run.run_name,
        sac_value = benchmark_run.sac_value,
        wall_seconds = wall_seconds,
        hot_path_seconds = hot_path_seconds,
        cold_seconds = cold_seconds,
        warm_seconds = warm_seconds,
        warm_average_seconds = warm_average_seconds,
        warm_step_seconds = warm_step_seconds,
        peak_allocated_mib = peak_allocated_mib,
        peak_reserved_mib = peak_reserved_mib,
        compile_modes = compile_modes,
    )

    print(f"benchmark {result.name} wall time: {wall_seconds:.3f}s")
    print(f"benchmark {result.name} measured hot-path time: {hot_path_seconds:.3f}s")
    print(f"benchmark {result.name} measured cold time: {cold_seconds:.3f}s")
    if warm_average_seconds is not None:
        print(f"benchmark {result.name} measured warm time: {warm_seconds:.3f}s total, {warm_average_seconds:.3f}s avg")
    if warm_step_seconds is not None:
        print(f"benchmark {result.name} warm per-step time: {warm_step_seconds:.3f}s")
    if peak_allocated_mib is not None:
        print(f"benchmark {result.name} peak CUDA memory: {peak_allocated_mib:.1f} MiB allocated, {peak_reserved_mib:.1f} MiB reserved")

    return result


def _format_optional(value: float | None, suffix: str = "s"):
    return f"{value:.3f}{suffix}" if value is not None else "n/a"


def print_benchmark_summary(results: Sequence[BenchmarkResult]):
    if len(results) == 0:
        raise ValueError("cannot summarize an empty benchmark result list")

    sac_order = []
    for result in results:
        if result.sac_value not in sac_order:
            sac_order.append(result.sac_value)

    print("\nbenchmark summary:")

    for sac_value in sac_order:
        sac_results = [result for result in results if result.sac_value == sac_value]
        eager = next((result for result in sac_results if result.name.startswith("eager")), None)
        eager_warm_step = eager.warm_step_seconds if eager is not None else None

        print(f"\n  SAC activation_memory_budget={sac_value}:")
        if eager is not None:
            print(
                f"    eager warm runtime: {eager.warm_seconds:.3f}s total, "
                f"{_format_optional(eager_warm_step)} per step"
            )
        else:
            print("    eager baseline: not selected")

        for result in sac_results:
            if eager is not None and result.name == eager.name:
                continue

            warm_speedup = None
            if (
                eager_warm_step is not None and
                result.warm_step_seconds is not None and
                result.warm_step_seconds > 0
            ):
                warm_speedup = eager_warm_step / result.warm_step_seconds

            compile_overhead = max(
                result.cold_seconds - eager.cold_seconds,
                0.,
            ) if eager is not None else None

            break_even_steps = None
            if (
                compile_overhead is not None and
                eager_warm_step is not None and
                result.warm_step_seconds is not None and
                result.warm_step_seconds < eager_warm_step
            ):
                break_even_steps = compile_overhead / (eager_warm_step - result.warm_step_seconds)

            print(f"\n    {result.name}:")
            print(f"      warm per-step speedup: {_format_optional(warm_speedup, suffix='x')}")
            print(f"      warm runtime: {result.warm_seconds:.3f}s total, {_format_optional(result.warm_step_seconds)} per step")
            print(f"      cold time: {result.cold_seconds:.3f}s")
            print(f"      estimated compile overhead: {_format_optional(compile_overhead)}")
            print(f"      estimated break-even warm steps: {_format_optional(break_even_steps, suffix='')}")
            print(f"      total elapsed: {result.wall_seconds:.3f}s")
            print(f"      effective compile paths: {', '.join(result.compile_modes)}")

            if result.peak_allocated_mib is not None:
                print(
                    f"      peak CUDA memory: {result.peak_allocated_mib:.1f} MiB allocated, "
                    f"{result.peak_reserved_mib:.1f} MiB reserved"
                )


def _benchmark_id(value: str | None):
    if value is None:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    value = str(value).strip()
    if not value or value in (".", "..") or "/" in value or "\\" in value:
        raise ValueError("benchmark_id must be a non-empty path-safe name")

    return value


class _Tee:
    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, value):
        self.console.write(value)
        self.log_file.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def isatty(self):
        return self.console.isatty()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def _tee_output(output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with output_path.open("w", encoding = "utf-8", buffering = 1) as log_file:
        sys.stdout = _Tee(original_stdout, log_file)
        sys.stderr = _Tee(original_stderr, log_file)

        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _run_main(
    *,
    config_path: str = str(DEFAULT_CONFIG_PATH),
    benchmark_cases: str | Sequence[str] | None = None,
    sac_values: str | float | Sequence[float] | None = None,
    benchmark_id: str,
    train_overrides: Mapping[str, Any] | None = None,
):
    train_overrides = {} if train_overrides is None else dict(train_overrides)

    config = load_benchmark_config(config_path)
    train_main = _load_train_main()
    valid_train_keys = set(signature(train_main).parameters)

    _validate_train_overrides(train_overrides, valid_keys = valid_train_keys)

    unknown_config_keys = sorted(set(config) - valid_train_keys)
    if unknown_config_keys:
        raise KeyError(f"{config_path} contains settings not accepted by train_gym.main: {unknown_config_keys}")

    missing_config_keys = sorted(valid_train_keys - set(config))
    if missing_config_keys:
        raise KeyError(f"{config_path} is missing settings required by train_gym.main: {missing_config_keys}")

    common_config = dict(config)
    common_config.update(train_overrides)
    benchmark_root = Path(common_config["log_dir"]) / "compile_benchmark" / _benchmark_id(benchmark_id)
    runs = build_benchmark_runs(
        config,
        benchmark_root = benchmark_root,
        benchmark_cases = benchmark_cases,
        sac_values = sac_values,
        train_overrides = train_overrides,
        valid_train_keys = valid_train_keys,
    )

    print(f"benchmark root: {benchmark_root}")
    print(f"running {len(runs)} benchmark test(s), {BENCHMARK_NUM_LOOPS} full loops per test")
    print(f"cases: {', '.join(run.run_name for run in runs)}")

    results = [run_benchmark_run(run, train_main = train_main) for run in runs]
    print_benchmark_summary(results)
    return [asdict(result) for result in results]


def main(
    config_path: str = str(DEFAULT_CONFIG_PATH),
    benchmark_cases: str | Sequence[str] | None = None,
    sac_values: str | float | Sequence[float] | None = None,
    benchmark_id: str | None = None,
    output_file: str | None = None,
    help: bool = False,
    **train_overrides,
):
    resolved_benchmark_id = _benchmark_id(benchmark_id)
    output_path = (
        Path.cwd() / f"benchmark_halfcheetah_compile_{resolved_benchmark_id}.log"
        if output_file is None
        else Path(output_file)
    )

    with _tee_output(output_path):
        print(f"benchmark output file: {output_path}")

        if help:
            print(__doc__)
            return None

        return _run_main(
            config_path = config_path,
            benchmark_cases = benchmark_cases,
            sac_values = sac_values,
            benchmark_id = resolved_benchmark_id,
            train_overrides = train_overrides,
        )


if __name__ == "__main__":
    fire.Fire(main)
