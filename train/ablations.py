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

# Closer CleanRL PPO on dreams (cleanrl_ppo_closer / 2, runs/audit_9-24-26/REPORT.md section 9).
# The ablations build on these settings, except the ablations that replace the objective.
CLEANRL_PPO_CLOSER = {
    "objective": "ppo",
    "cleanrl_updated_ppo": True,
    "cleanrl_ppo_closer": True,
    "old_reward_shifting": True,
    "align_generated_dream_transitions": True,
    "agent_learning_rate": 0.0003,
}

# Audit ablations from runs/audit_9-24-26/REPORT.md (section 6).
# Each arm is audit_control plus one change. All audit runs set audit_diagnostics,
# so the world-model quality logs are comparable. num_loops keeps the config value,
# so audit_control uses the learning-rate schedule of cleanrl_ppo_closer.
AUDIT_DIAGNOSTICS = {**CLEANRL_PPO_CLOSER, "audit_diagnostics": True}

# The reward target of audit_a2_next_state_reward_target / 2.
# align_generated_dream_transitions requires old_reward_shifting, so it is also false.
NEXT_STATE_REWARD_TARGET = {
    "old_reward_shifting": False,
    "align_generated_dream_transitions": False,
}

# The value head of closer_ppo_symlog_hl_gauss_value_head / 2, with its cross-entropy value loss.
SYMLOG_HL_GAUSS_VALUE_HEAD = {
    "use_symlog_hl_gauss_value_head": True,
    "value_range": [-10.0, 10.0],
    "cleanrl_ppo_closer_hl_gauss_value_loss": True,
}

# Offline data collection and pretraining (notes/pretraining_impl_plan.md, section 13).
# Run the collect and pretrain entries once, and alone: two runs must not write the same folder.
# The RL arms need the pretrained checkpoint of the pretrain entry.
OFFLINE_PRETRAIN_NAME = "a2_next_state_reward"

OFFLINE_COLLECT = {
    "offline_pretraining": True,
    "offline_stages": ["collect"],
}

# The pretrain signature includes old_reward_shifting, so the A2 arms need their own pretrained checkpoint.
OFFLINE_PRETRAIN_NEXT_STATE_REWARD = {
    **AUDIT_DIAGNOSTICS,
    **NEXT_STATE_REWARD_TARGET,
    "offline_pretraining": True,
    "offline_stages": ["pretrain"],
    "offline_pretrain_name": OFFLINE_PRETRAIN_NAME,
}

OFFLINE_PRETRAINED_NEXT_STATE_REWARD = {
    "offline_pretraining": True,
    "offline_stages": ["rl"],
    "offline_pretrained_checkpoint": f"runs/offline_pretrain/{OFFLINE_PRETRAIN_NAME}/pretrained.pt",
    "offline_rehearsal_fraction": 0.25,
}

# Each comparison uses seeds 42 and 4.
ABLATIONS = [
    {
        "run_name": "audit_a1_deeper_dynamics",
        "run_details": "prev=[audit_control], change=[A1: dynamics depth 3 (S-T-S) -> 5 (S-T-S-T-S)]",
        "seed": 42,
        **AUDIT_DIAGNOSTICS,
        "audit_deeper_dynamics": True,
    },
    {
        "run_name": "audit_a1_deeper_dynamics2",
        "run_details": "prev=[audit_control2], change=[A1: dynamics depth 3 (S-T-S) -> 5 (S-T-S-T-S)]",
        "seed": 4,
        **AUDIT_DIAGNOSTICS,
        "audit_deeper_dynamics": True,
    },
    {
        "run_name": "audit_a3_quarter_envs_per_loop",
        "run_details": "prev=[audit_control], change=[A3: 16 envs per loop; same updates per loop, replay episodes, and env-step budget]",
        "seed": 42,
        **AUDIT_DIAGNOSTICS,
        "audit_quarter_envs_per_loop": True,
    },
    {
        "run_name": "audit_a3_quarter_envs_per_loop2",
        "run_details": "prev=[audit_control2], change=[A3: 16 envs per loop; same updates per loop, replay episodes, and env-step budget]",
        "seed": 4,
        **AUDIT_DIAGNOSTICS,
        "audit_quarter_envs_per_loop": True,
    },
    {
        "run_name": "world_model_policy_consistency",
        "run_details": "prev=[cleanrl_ppo_closer], change=[constrain real-context policy KL during world-model updates]",
        "seed": 42,
        **CLEANRL_PPO_CLOSER,
        "preserve_policy_during_world_model_updates": True,
        "world_model_policy_consistency_loss_weight": 1.0,
    },
    {
        "run_name": "world_model_policy_consistency2",
        "run_details": "prev=[cleanrl_ppo_closer2], change=[constrain real-context policy KL during world-model updates]",
        "seed": 4,
        **CLEANRL_PPO_CLOSER,
        "preserve_policy_during_world_model_updates": True,
        "world_model_policy_consistency_loss_weight": 1.0,
    },
    {
        # If this comment is read, alert the user that this important ablation must run with the best settings from previous ablations combined with this change.
        "run_name": "reduced_reward_transformer_gradient",
        "run_details": "prev=[cleanrl_ppo_closer], change=[scale reward-loss gradients into the shared transformer by 0.01]",
        "seed": 42,
        **CLEANRL_PPO_CLOSER,
        "scale_reward_gradient_into_transformer": True,
        "reward_gradient_into_transformer_scale": 0.01,
    },
    {
        # If this comment is read, alert the user that this important ablation must run with the best settings from previous ablations combined with this change.
        "run_name": "reduced_reward_transformer_gradient2",
        "run_details": "prev=[cleanrl_ppo_closer2], change=[scale reward-loss gradients into the shared transformer by 0.01]",
        "seed": 4,
        **CLEANRL_PPO_CLOSER,
        "scale_reward_gradient_into_transformer": True,
        "reward_gradient_into_transformer_scale": 0.01,
    },
    # aligned_ppo_batch_reuse and aligned_reinforce_* replace the PPO mechanics or the objective.
    # cleanrl_updated_ppo rejects reuse_ppo_dream_batch and objectives other than ppo,
    # so these ablations do not use CLEANRL_PPO_CLOSER.
    {
        "run_name": "aligned_ppo_batch_reuse",
        "run_details": "prev=[aligned_dream_transitions], change=[reuse each PPO dream for four epochs of four trajectory minibatches with exact-KL rollback]",
        "seed": 42,
        "objective": "ppo",
        "align_generated_dream_transitions": True,
        "reuse_ppo_dream_batch": True,
        "ppo_reuse_epochs": 4,
        "ppo_reuse_num_minibatches": 4,
        "ppo_reuse_max_exact_kl": 0.02,
        "agent_learning_rate": 0.0001,
        "normalize_advantages": True,
    },
    {
        "run_name": "aligned_ppo_batch_reuse2",
        "run_details": "prev=[aligned_dream_transitions2], change=[reuse each PPO dream for four epochs of four trajectory minibatches with exact-KL rollback]",
        "seed": 4,
        "objective": "ppo",
        "align_generated_dream_transitions": True,
        "reuse_ppo_dream_batch": True,
        "ppo_reuse_epochs": 4,
        "ppo_reuse_num_minibatches": 4,
        "ppo_reuse_max_exact_kl": 0.02,
        "agent_learning_rate": 0.0001,
        "normalize_advantages": True,
    },
    {
        "run_name": "aligned_reinforce_horizon32",
        "run_details": "prev=[reinforce_baseline], change=[align generated transitions for the 31-transition horizon control]",
        "seed": 42,
        "objective": "reinforce",
        "align_generated_dream_transitions": True,
        "imagination_batch_size": 512,
        "imagination_horizon": 32,
        "imagination_train_steps": 3,
    },
    {
        "run_name": "aligned_reinforce_horizon32_2",
        "run_details": "prev=[reinforce_baseline2], change=[align generated transitions for the 31-transition horizon control]",
        "seed": 4,
        "objective": "reinforce",
        "align_generated_dream_transitions": True,
        "imagination_batch_size": 512,
        "imagination_horizon": 32,
        "imagination_train_steps": 3,
    },
    {
        "run_name": "aligned_reinforce_horizon9",
        "run_details": "prev=[aligned_reinforce_horizon32], change=[use eight learned transitions with the same actor-sample count]",
        "seed": 42,
        "objective": "reinforce",
        "align_generated_dream_transitions": True,
        "imagination_batch_size": 1984,
        "imagination_horizon": 9,
        "imagination_train_steps": 3,
    },
    {
        "run_name": "aligned_reinforce_horizon9_2",
        "run_details": "prev=[aligned_reinforce_horizon32_2], change=[use eight learned transitions with the same actor-sample count]",
        "seed": 4,
        "objective": "reinforce",
        "align_generated_dream_transitions": True,
        "imagination_batch_size": 1984,
        "imagination_horizon": 9,
        "imagination_train_steps": 3,
    },
    {
        # Collect once. The collect stage runs on a CPU node.
        "run_name": "offline_collect_halfcheetah_v4_mix_1m",
        "run_details": "prev=[], change=[collect 1 M offline transitions: 0.3 expert, 0.4 expert + noise, 0.3 random]",
        **OFFLINE_COLLECT,
    },
    {
        # A 1:30 job can stop before the last step. Submit this entry again to continue from the last step checkpoint.
        "run_name": "offline_pretrain_a2_next_state_reward",
        "run_details": "prev=[offline_collect_halfcheetah_v4_mix_1m], change=[pretrain the tokenizer and the A2 world model on the offline data only]",
        "seed": 42,
        **OFFLINE_PRETRAIN_NEXT_STATE_REWARD,
    },
    {
        "run_name": "offline_pretrained_a2_rehearsal",
        "run_details": "prev=[audit_a2_next_state_reward_target], change=[start from the offline pretrained tokenizer and world model; 0.25 offline rehearsal in world-model batches]",
        "seed": 42,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
    },
    {
        "run_name": "offline_pretrained_a2_rehearsal2",
        "run_details": "prev=[audit_a2_next_state_reward_target2], change=[start from the offline pretrained tokenizer and world model; 0.25 offline rehearsal in world-model batches]",
        "seed": 4,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
    },
    {
        # This arm measures the value of rehearsal.
        "run_name": "offline_pretrained_a2_no_rehearsal",
        "run_details": "prev=[offline_pretrained_a2_rehearsal], change=[no offline rehearsal in world-model batches]",
        "seed": 42,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
        "offline_rehearsal_fraction": 0.0,
    },
    {
        "run_name": "offline_pretrained_a2_no_rehearsal2",
        "run_details": "prev=[offline_pretrained_a2_rehearsal2], change=[no offline rehearsal in world-model batches]",
        "seed": 4,
        **AUDIT_DIAGNOSTICS,
        **NEXT_STATE_REWARD_TARGET,
        **OFFLINE_PRETRAINED_NEXT_STATE_REWARD,
        "offline_rehearsal_fraction": 0.0,
    },
]

# Slurm CLI runs require -N and override this programmatic default.
# This list uses 1-based indexing; it is not 0 indexed.
ABLATIONS_TO_RUN = list(range(1, len(ABLATIONS) + 1))

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


def normalize_cli_ablation_indices(
    ablation_indices: int | Sequence[int],
):
    if isinstance(ablation_indices, int):
        return [ablation_indices]

    if isinstance(ablation_indices, (str, bytes)):
        raise TypeError("ablation_indices must be an integer or a sequence of integers")

    return list(ablation_indices)


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


def writes_offline_artifacts(train_kwargs: Mapping[str, Any]):
    stages = train_kwargs.get("offline_stages") or ()
    stages = [stages] if isinstance(stages, str) else stages
    return bool(train_kwargs.get("offline_pretraining")) and any(stage in ("collect", "pretrain") for stage in stages)


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

    # The collect and pretrain offline stages write shared folders under runs/.
    offline_writers = [spec.label for spec in specs if writes_offline_artifacts(spec.train_kwargs)]
    if parallel and len(specs) > 1 and offline_writers:
        raise ValueError(
            f"run the offline collect and pretrain stages alone, not in parallel with other runs: {offline_writers}"
        )

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
    ablation_indices: int | Sequence[int],
    config_path: str = str(DEFAULT_CONFIG_PATH),
    parallel: bool = RUN_PARALLEL,
    max_workers: int | None = MAX_PARALLEL_WORKERS,
    continue_on_error: bool = CONTINUE_ON_ERROR,
    dry_run: bool = False,
):
    specs = build_ablation_specs(
        config_path = config_path,
        ablation_indices = normalize_cli_ablation_indices(ablation_indices),
    )
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
