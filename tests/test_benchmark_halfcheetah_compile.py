from __future__ import annotations

import sys

import pytest

from train.benchmark_halfcheetah_compile import (
    BENCHMARK_NUM_LOOPS,
    CUSTOM_BENCHMARK_CASES,
    CUSTOM_PERFORMANCE_OPTIONS,
    _tee_output,
    build_benchmark_runs,
    load_benchmark_config,
    parse_sac_values,
    select_benchmark_cases,
)


def test_benchmark_output_is_teeed_to_file(tmp_path, capsys):
    output_path = tmp_path / "benchmark.log"

    with _tee_output(output_path):
        print("stdout line")
        print("stderr line", file = sys.stderr)

    output = output_path.read_text(encoding = "utf-8")
    captured = capsys.readouterr()
    assert "stdout line" in output
    assert "stderr line" in output
    assert "stdout line" in captured.out
    assert "stderr line" in captured.err


def test_default_benchmark_uses_all_requested_cases_and_three_loops(tmp_path):
    config = load_benchmark_config()
    runs = build_benchmark_runs(config, benchmark_root = tmp_path / "benchmark")

    assert [run.case.name for run in runs] == [
        "eager",
        "compiled_default",
        "compiled_max_autotune",
    ]

    for run in runs:
        assert run.train_kwargs["num_loops"] == BENCHMARK_NUM_LOOPS
        assert run.train_kwargs["pretrain_tokenizer_steps"] == 10
        assert run.train_kwargs["pretrain_world_model_finetune_steps"] == 0
        assert run.train_kwargs["pretrain_world_model_combined_steps"] == 0
        assert run.train_kwargs["use_tensorboard"] is False
        assert run.train_kwargs["checkpoint_every"] == 0
        assert run.train_kwargs["imagination_use_time_cache"] is True
        assert run.train_kwargs["compile_generate_with_time_cache"] is True
        assert run.train_kwargs["compile_dynamic"] is False
        assert run.train_kwargs["activation_memory_budget"] == config["activation_memory_budget"]


def test_benchmark_selects_multiple_cases_and_sac_values(tmp_path):
    config = load_benchmark_config()
    runs = build_benchmark_runs(
        config,
        benchmark_root = tmp_path / "benchmark",
        benchmark_cases = "eager,max-autotune",
        sac_values = "0.0,0.5,0.9",
    )

    assert len(runs) == 6
    assert [run.case.name for run in runs[:3]] == ["eager"] * 3
    assert [run.sac_value for run in runs[:3]] == [0., 0.5, 0.9]
    assert runs[3].case.name == "compiled_max_autotune"
    assert runs[3].case.compile_mode == "max-autotune"
    assert runs[3].case.compile_generate_cudagraphs is True
    assert runs[3].train_kwargs["activation_memory_budget"] == 0.
    assert runs[3].run_name.endswith("__sac_0")


def test_custom_performance_options_run_individually_and_are_opt_in(tmp_path):
    config = load_benchmark_config()
    runs = build_benchmark_runs(
        config,
        benchmark_root = tmp_path / "benchmark",
        benchmark_cases = "custom",
    )

    assert len(runs) == len(CUSTOM_PERFORMANCE_OPTIONS) == len(CUSTOM_BENCHMARK_CASES)
    assert [run.case.name for run in runs] == [
        f"compiled_{option.replace('.', '_')}"
        for option in CUSTOM_PERFORMANCE_OPTIONS
    ]
    assert [run.train_kwargs["compile_options"] for run in runs] == [
        {option: value}
        for option, value in CUSTOM_PERFORMANCE_OPTIONS.items()
    ]
    assert all(run.train_kwargs["compile_mode"] is None for run in runs)


def test_individual_custom_option_can_be_selected(tmp_path):
    config = load_benchmark_config()
    runs = build_benchmark_runs(
        config,
        benchmark_root = tmp_path / "benchmark",
        benchmark_cases = "cpp_wrapper",
    )

    assert len(runs) == 1
    assert runs[0].case.name == "compiled_cpp_wrapper"
    assert runs[0].train_kwargs["compile_options"] == {"cpp_wrapper": True}


def test_perf_benchmark_keeps_full_config_steps_but_stabilizes_batch_shapes(tmp_path):
    config = load_benchmark_config()
    runs = build_benchmark_runs(
        config,
        benchmark_root = tmp_path / "benchmark",
        benchmark_cases = "eager",
        train_overrides = {"num_envs": 8},
    )
    kwargs = runs[0].train_kwargs

    assert kwargs["num_loops"] == 3
    assert kwargs["pretrain_world_model_finetune_steps"] == 0
    assert kwargs["pretrain_world_model_combined_steps"] == 0
    assert kwargs["pretrain_tokenizer_steps"] == 10
    assert kwargs["world_model_train_steps"] == config["world_model_train_steps"]
    assert kwargs["imagination_train_steps"] == config["imagination_train_steps"]
    assert kwargs["world_model_batch_size"] == 8
    assert kwargs["imagination_batch_size"] == 8


def test_benchmark_rejects_non_three_loop_override(tmp_path):
    config = load_benchmark_config()

    with pytest.raises(ValueError, match = "exactly 3 loops"):
        build_benchmark_runs(
            config,
            benchmark_root = tmp_path / "benchmark",
            train_overrides = {"num_loops": 2},
        )


def test_sac_values_are_validated_and_case_selection_rejects_duplicates():
    with pytest.raises(ValueError, match = "between 0 and 1"):
        parse_sac_values("-0.1,0.5", default = 0.9)

    with pytest.raises(ValueError, match = "selected more than once"):
        select_benchmark_cases("eager,eager")
