# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fire",
#     "pyyaml",
# ]
# ///

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Mapping, Sequence

import fire
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
GYM_LOG_ROOT = Path("runs/gym_imagination")

# Each ablation requires seeds 4 and 42.
ABLATIONS = [
    {
        "run_name": "reinforce_baseline",
        "run_details": "prev=[], change=[select the current REINFORCE objective]",
        "seed": 42,
        "objective": "reinforce",
    },
    {
        "run_name": "reinforce_baseline2",
        "run_details": "prev=[], change=[select the current REINFORCE objective]",
        "seed": 4,
        "objective": "reinforce",
    },
    {
        "run_name": "reinforce_dreamerv3_targets",
        "run_details": "prev=[reinforce_baseline], change=[use DreamerV3 discount 1-1/333, lambda 0.95, and percentile return-scale normalization]",
        "seed": 42,
        "objective": "reinforce",
        "reinforce_discount_factor": 1. - 1. / 333.,
        "reinforce_gae_lambda": 0.95,
        "reinforce_return_normalization": "percentile",
    },
    {
        "run_name": "reinforce_dreamerv3_targets2",
        "run_details": "prev=[reinforce_baseline2], change=[use DreamerV3 discount 1-1/333, lambda 0.95, and percentile return-scale normalization]",
        "seed": 4,
        "objective": "reinforce",
        "reinforce_discount_factor": 1. - 1. / 333.,
        "reinforce_gae_lambda": 0.95,
        "reinforce_return_normalization": "percentile",
    },
    {
        "run_name": "reinforce_dreamerv3_actor_loss",
        "run_details": "prev=[reinforce_dreamerv3_targets], change=[add discounted trajectory weights, use squashed-Gaussian bounded actions, and reduce entropy weight from 0.01 to 0.0003]",
        "seed": 42,
        "objective": "reinforce",
        "reinforce_discount_factor": 1. - 1. / 333.,
        "reinforce_gae_lambda": 0.95,
        "reinforce_return_normalization": "percentile",
        "reinforce_use_discounted_trajectory_weights": True,
        "continuous_dist_type": "squashed_gaussian",
        "policy_entropy_weight": 0.0003,
    },
    {
        "run_name": "reinforce_dreamerv3_actor_loss2",
        "run_details": "prev=[reinforce_dreamerv3_targets2], change=[add discounted trajectory weights, use squashed-Gaussian bounded actions, and reduce entropy weight from 0.01 to 0.0003]",
        "seed": 4,
        "objective": "reinforce",
        "reinforce_discount_factor": 1. - 1. / 333.,
        "reinforce_gae_lambda": 0.95,
        "reinforce_return_normalization": "percentile",
        "reinforce_use_discounted_trajectory_weights": True,
        "continuous_dist_type": "squashed_gaussian",
        "policy_entropy_weight": 0.0003,
    },
]

# This list uses 1-based indexing; it is not 0 indexed.
# Keep exactly two runs for each ablation here.
ABLATIONS_TO_RUN = [5]


RUN_PARALLEL = True
MAX_PARALLEL_WORKERS: int | None = 2
CONTINUE_ON_ERROR = False
PARALLEL_LAUNCH_DELAY_SECONDS = 60


@dataclass(frozen = True)
class AblationSpec:
    index: int
    label: str
    overrides: dict[str, Any]
    train_kwargs: dict[str, Any]
    run_log_dir: str | None


@dataclass(frozen = True)
class AblationResult:
    index: int
    label: str
    run_name: str
    run_log_dir: str | None
    success: bool
    elapsed_seconds: float
    error: str | None = None


def ensure_repo_root_on_sys_path():
    repo_root = str(REPO_ROOT)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def load_baseline_config(config_path: str | Path = DEFAULT_CONFIG_PATH):
    config_path = Path(config_path)

    with config_path.open("r", encoding = "utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a top-level mapping")

    if "run_name" not in config:
        raise KeyError(f"{config_path} must define run_name")

    if "run_details" not in config:
        raise KeyError(f"{config_path} must define run_details")

    return config


def format_override_value(value: Any):
    if isinstance(value, str):
        return value

    return json.dumps(value, sort_keys = True, separators = (",", ":"))


def format_ablation_label(overrides: Mapping[str, Any]):
    if len(overrides) == 0:
        return "baseline"

    return ",".join(
        f"{key}={format_override_value(value)}"
        for key, value in sorted(overrides.items())
    )


def slugify(value: str):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "baseline"


def build_run_details(
    base_details: str,
    *,
    index: int,
    label: str,
    overrides: Mapping[str, Any],
    extra_details: str | None,
):
    parts = [str(base_details).strip()] if str(base_details).strip() else []
    parts.append(f"Ablation {index}: {label}")
    parts.append(f"Overrides: {json.dumps(dict(sorted(overrides.items())), sort_keys = True)}")

    if extra_details:
        parts.append(str(extra_details).strip())

    return "\n\n".join(part for part in parts if part)


def plan_run_log_dir(train_kwargs: Mapping[str, Any]):
    checkpoint_path = train_kwargs.get("checkpoint_path")

    if checkpoint_path:
        return None

    return str(GYM_LOG_ROOT / str(train_kwargs["run_name"]))


def plan_checkpoint_folder(train_kwargs: dict[str, Any]):
    """Give each new ablation run its own checkpoint directory."""
    if train_kwargs.get("checkpoint_path"):
        return

    checkpoint_folder = train_kwargs.get("checkpoint_folder")

    if checkpoint_folder is None:
        return

    run_name = str(train_kwargs["run_name"])
    checkpoint_folder = Path(str(checkpoint_folder))

    if checkpoint_folder.name != run_name:
        checkpoint_folder /= run_name

    train_kwargs["checkpoint_folder"] = str(checkpoint_folder)


def merge_small_model_config(
    baseline: Mapping[str, Any],
    overrides: Mapping[str, Any],
):
    """Merge state-only overrides without dropping unspecified defaults."""

    merged = deepcopy(dict(baseline))
    merged.update(overrides)

    if "small_model" not in overrides:
        return merged

    base_small_model = baseline.get("small_model")
    override_small_model = overrides["small_model"]
    if not isinstance(base_small_model, Mapping):
        raise TypeError("baseline small_model must be a mapping")
    if not isinstance(override_small_model, Mapping):
        raise TypeError("small_model ablation override must be a mapping")

    merged_small_model = deepcopy(dict(base_small_model))
    merged_small_model.update(override_small_model)
    merged["small_model"] = merged_small_model
    return merged


def select_ablation_indices(
    ablation_count: int,
    indices: Sequence[int] | None = None,
):
    selected_indices = list(ABLATIONS_TO_RUN if indices is None else indices)

    if len(selected_indices) == 0:
        raise ValueError("ABLATIONS_TO_RUN must contain at least one ablation index")

    seen = set()

    for index in selected_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(
                "ABLATIONS_TO_RUN must contain only integer indices; indices are 1-based"
            )

        if index < 1 or index > ablation_count:
            raise IndexError(
                f"ablation index {index} is out of range; choose values from 1 to "
                f"{ablation_count} (indices are 1-based)"
            )

        if index in seen:
            raise ValueError(f"ablation index {index} was selected more than once")

        seen.add(index)

    return selected_indices


def build_ablation_specs(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    ablations: Sequence[Mapping[str, Any]] | None = None,
    ablation_indices: Sequence[int] | None = None,
):
    baseline = load_baseline_config(config_path)
    raw_ablations = list(ABLATIONS if ablations is None else ablations)

    if len(raw_ablations) == 0:
        raise ValueError(
            "ABLATIONS is empty. Edit train/ablations.py and add one or more override dicts, "
            "for example ABLATIONS = [{'depth': 8}]"
        )

    if ablations is not None and ablation_indices is None:
        selected_indices = list(range(1, len(raw_ablations) + 1))
    else:
        selected_indices = select_ablation_indices(len(raw_ablations), ablation_indices)
    allowed_override_keys = set(baseline.keys())
    specs = []

    for index in selected_indices:
        ablation = raw_ablations[index - 1]

        if not isinstance(ablation, Mapping):
            raise TypeError(f"ablation #{index} must be a mapping, got {type(ablation).__name__}")

        overrides = dict(ablation)
        unknown_keys = sorted(set(overrides) - allowed_override_keys)

        if unknown_keys:
            raise KeyError(
                f"ablation #{index} contains keys not present in {config_path}: {unknown_keys}"
            )

        run_name = str(overrides.get("run_name", "")).strip()
        run_details = str(overrides.get("run_details", "")).strip()

        if not run_name:
            raise KeyError(
                f"ablation #{index} must define a non-empty run_name because each ablation needs "
                "its own log directory under runs/gym_imagination/<run_name>"
            )

        if not run_details:
            raise KeyError(f"ablation #{index} must define a non-empty run_details")

        label = run_name

        train_kwargs = merge_small_model_config(baseline, overrides)
        train_kwargs["run_name"] = run_name
        train_kwargs["log_dir"] = str(GYM_LOG_ROOT)
        plan_checkpoint_folder(train_kwargs)
        train_kwargs["run_details"] = build_run_details(
            str(baseline["run_details"]),
            index = index,
            label = label,
            overrides = {key: value for key, value in overrides.items() if key not in ("run_name", "run_details")},
            extra_details = run_details,
        )

        specs.append(
            AblationSpec(
                index = index,
                label = label,
                overrides = dict(overrides),
                train_kwargs = train_kwargs,
                run_log_dir = plan_run_log_dir(train_kwargs),
            )
        )

    return specs


def run_train_gym(train_kwargs: Mapping[str, Any]):
    ensure_repo_root_on_sys_path()
    from train.train_gym import main as train_gym_main

    return train_gym_main(**dict(train_kwargs))


def log_ablation_error(
    spec: AblationSpec,
    error: str,
    *,
    source: str,
):
    """Write a failure traceback before the launcher waits for other runs."""
    print(
        f"\n[ablation {spec.index}] {source} for {spec.train_kwargs['run_name']}:\n{error}",
        file = sys.stderr,
        flush = True,
    )


def failed_ablation_result(
    spec: AblationSpec,
    *,
    elapsed_seconds: float,
    error: str,
    source: str,
):
    log_ablation_error(spec, error, source = source)

    return AblationResult(
        index = spec.index,
        label = spec.label,
        run_name = str(spec.train_kwargs["run_name"]),
        run_log_dir = spec.run_log_dir,
        success = False,
        elapsed_seconds = elapsed_seconds,
        error = error,
    )


def execute_ablation(spec: AblationSpec, *, dry_run: bool):
    start = perf_counter()

    print(f"[ablation {spec.index}] starting {spec.label}", flush = True)
    print(f"[ablation {spec.index}] run_name={spec.train_kwargs['run_name']}", flush = True)

    if spec.run_log_dir:
        print(f"[ablation {spec.index}] log_dir={spec.run_log_dir}", flush = True)

    if spec.overrides:
        print(
            f"[ablation {spec.index}] overrides={json.dumps(spec.overrides, sort_keys = True)}",
            flush = True,
        )
    else:
        print(f"[ablation {spec.index}] overrides={{}}", flush = True)

    try:
        if not dry_run:
            run_train_gym(spec.train_kwargs)
    except Exception:
        return failed_ablation_result(
            spec,
            elapsed_seconds = perf_counter() - start,
            error = traceback.format_exc(),
            source = "worker error",
        )

    return AblationResult(
        index = spec.index,
        label = spec.label,
        run_name = str(spec.train_kwargs["run_name"]),
        run_log_dir = spec.run_log_dir,
        success = True,
        elapsed_seconds = perf_counter() - start,
    )


def execute_ablation_in_subprocess(spec: AblationSpec, dry_run: bool):
    return execute_ablation(spec, dry_run = dry_run)


def print_planned_ablations(specs: Sequence[AblationSpec], *, config_path: str | Path):
    print(f"loaded {len(specs)} ablation(s) from {Path(config_path)}")

    for spec in specs:
        print(
            f"- [{spec.index}] {spec.label} -> run_name={spec.train_kwargs['run_name']} "
            f"overrides={json.dumps(spec.overrides, sort_keys = True)}"
        )


def summarize_results(results: Sequence[AblationResult]):
    ordered = sorted(results, key = lambda result: result.index)
    success_count = sum(result.success for result in ordered)

    print("\nablation summary:")
    for result in ordered:
        status = "ok" if result.success else "failed"
        print(
            f"- [{result.index}] {status} {result.run_name} "
            f"elapsed={result.elapsed_seconds:.2f}s"
        )

    print(f"completed {success_count}/{len(ordered)} ablation(s) successfully")


def run_ablation_specs(
    specs: Sequence[AblationSpec],
    *,
    parallel: bool,
    max_workers: int | None,
    continue_on_error: bool,
    dry_run: bool,
):
    if len(specs) == 0:
        return []

    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be at least 1 when provided")

    results: list[AblationResult] = []

    def run_one_in_fresh_process(spec: AblationSpec):
        with ProcessPoolExecutor(max_workers = 1, mp_context = get_context("spawn")) as executor:
            future = executor.submit(execute_ablation_in_subprocess, spec, dry_run)

            try:
                return future.result()
            except Exception:
                return failed_ablation_result(
                    spec,
                    elapsed_seconds = 0.,
                    error = traceback.format_exc(),
                    source = "worker process error",
                )

    if not parallel or len(specs) == 1:
        for spec in specs:
            result = execute_ablation(spec, dry_run = dry_run) if dry_run else run_one_in_fresh_process(spec)
            results.append(result)

            if not result.success and not continue_on_error:
                break

        return results

    worker_count = max_workers or min(len(specs), os.cpu_count() or 1)
    mp_context = get_context("spawn")

    with ProcessPoolExecutor(max_workers = worker_count, mp_context = mp_context) as executor:
        future_to_spec = {}

        for index, spec in enumerate(specs):
            if index > 0:
                sleep(PARALLEL_LAUNCH_DELAY_SECONDS)

            future_to_spec[executor.submit(execute_ablation_in_subprocess, spec, dry_run)] = spec

        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]

            try:
                result = future.result()
            except Exception:
                result = failed_ablation_result(
                    spec,
                    elapsed_seconds = 0.,
                    error = traceback.format_exc(),
                    source = "worker process error",
                )

            results.append(result)

    return results


def main(
    config_path: str = str(DEFAULT_CONFIG_PATH),
    parallel: bool = RUN_PARALLEL,
    max_workers: int | None = MAX_PARALLEL_WORKERS,
    continue_on_error: bool = CONTINUE_ON_ERROR,
    dry_run: bool = False,
):
    specs = build_ablation_specs(config_path = config_path)
    print_planned_ablations(specs, config_path = config_path)

    results = run_ablation_specs(
        specs,
        parallel = parallel,
        max_workers = max_workers,
        continue_on_error = continue_on_error,
        dry_run = dry_run,
    )

    summarize_results(results)

    failures = [result for result in results if not result.success]
    if failures:
        first_failure = failures[0]

        if first_failure.error:
            print("\nfirst failure traceback:\n")
            print(first_failure.error)

        raise SystemExit(1)

    return [asdict(result) for result in sorted(results, key = lambda result: result.index)]


if __name__ == "__main__":
    fire.Fire(main)
