from __future__ import annotations

import pytest
import yaml

from train import ablations
from train.ablations import AblationSpec, build_ablation_specs, execute_ablation


def write_config(path, config):
    path.write_text(yaml.safe_dump(config), encoding = "utf-8")


def test_build_ablation_specs_uses_baseline_config_and_overrides(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(
        config_path,
        {
            "run_name": "halfcheetah_baseline",
            "run_details": "Baseline config",
            "log_dir": "runs/gym_imagination",
            "depth": 3,
            "model_dim": 128,
            "checkpoint_path": None,
        },
    )

    specs = build_ablation_specs(
        config_path = config_path,
        ablations = [
            {"run_name": "depth8", "run_details": "Depth-8 ablation", "depth": 8},
            {"run_name": "wider", "run_details": "Wider model ablation", "model_dim": 256},
        ],
    )

    assert len(specs) == 2

    first = specs[0]
    assert first.overrides == {"run_name": "depth8", "run_details": "Depth-8 ablation", "depth": 8}
    assert first.train_kwargs["depth"] == 8
    assert first.train_kwargs["model_dim"] == 128
    assert first.train_kwargs["run_name"] == "depth8"
    assert first.train_kwargs["log_dir"] == "runs/gym_imagination"
    assert first.run_log_dir == "runs/gym_imagination/depth8"
    assert "Ablation 1: depth8" in first.train_kwargs["run_details"]
    assert "Depth-8 ablation" in first.train_kwargs["run_details"]

    second = specs[1]
    assert second.label == "wider"
    assert second.train_kwargs["model_dim"] == 256
    assert second.train_kwargs["depth"] == 3
    assert second.train_kwargs["run_name"] == "wider"
    assert second.run_log_dir == "runs/gym_imagination/wider"


def test_build_ablation_specs_selects_requested_one_based_indices(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(
        config_path,
        {
            "run_name": "baseline",
            "run_details": "details",
            "log_dir": "runs/test",
        },
    )

    specs = build_ablation_specs(
        config_path = config_path,
        ablations = [
            {"run_name": "first", "run_details": "first run"},
            {"run_name": "second", "run_details": "second run"},
            {"run_name": "third", "run_details": "third run"},
        ],
        ablation_indices = [1, 3],
    )

    assert [spec.index for spec in specs] == [1, 3]
    assert [spec.label for spec in specs] == ["first", "third"]


def test_build_ablation_specs_rejects_zero_based_or_out_of_range_indices(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(
        config_path,
        {
            "run_name": "baseline",
            "run_details": "details",
            "log_dir": "runs/test",
        },
    )

    with pytest.raises(IndexError, match = "1-based"):
        build_ablation_specs(
            config_path = config_path,
            ablations = [{"run_name": "first", "run_details": "first run"}],
            ablation_indices = [0],
        )

    with pytest.raises(IndexError, match = "1-based"):
        build_ablation_specs(
            config_path = config_path,
            ablations = [{"run_name": "first", "run_details": "first run"}],
            ablation_indices = [2],
        )


def test_build_ablation_specs_rejects_unknown_override_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(
        config_path,
        {
            "run_name": "baseline",
            "run_details": "details",
            "log_dir": "runs/test",
            "depth": 3,
        },
    )

    with pytest.raises(KeyError, match = "not present"):
        build_ablation_specs(
            config_path = config_path,
            ablations = [{"run_name": "bad", "run_details": "bad", "does_not_exist": 123}],
        )


def test_build_ablation_specs_requires_run_name_and_run_details(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(
        config_path,
        {
            "run_name": "baseline",
            "run_details": "details",
            "log_dir": "runs/test",
            "depth": 3,
            "checkpoint_path": None,
        },
    )

    with pytest.raises(KeyError, match = "run_name"):
        build_ablation_specs(
            config_path = config_path,
            ablations = [{"run_details": "details only"}],
        )

    with pytest.raises(KeyError, match = "run_details"):
        build_ablation_specs(
            config_path = config_path,
            ablations = [{"run_name": "name-only"}],
        )


def test_new_ablation_runs_get_distinct_checkpoint_folders(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(
        config_path,
        {
            "run_name": "baseline",
            "run_details": "details",
            "log_dir": "runs/test",
            "checkpoint_folder": "checkpoints_gym_imagination",
            "checkpoint_path": None,
        },
    )

    specs = build_ablation_specs(
        config_path = config_path,
        ablations = [
            {"run_name": "first", "run_details": "first run"},
            {"run_name": "second", "run_details": "second run"},
        ],
    )

    assert specs[0].train_kwargs["checkpoint_folder"] == "checkpoints_gym_imagination/first"
    assert specs[1].train_kwargs["checkpoint_folder"] == "checkpoints_gym_imagination/second"


def test_resume_ablation_preserves_explicit_checkpoint_folder(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_config(
        config_path,
        {
            "run_name": "baseline",
            "run_details": "details",
            "log_dir": "runs/test",
            "checkpoint_folder": "existing_checkpoints",
            "checkpoint_path": "existing_checkpoints/loop_25.pt",
        },
    )

    spec = build_ablation_specs(
        config_path = config_path,
        ablations = [{"run_name": "resume", "run_details": "resume run"}],
    )[0]

    assert spec.train_kwargs["checkpoint_folder"] == "existing_checkpoints"
    assert spec.train_kwargs["checkpoint_path"] == "existing_checkpoints/loop_25.pt"


def test_execute_ablation_logs_worker_exception_immediately(monkeypatch, capsys):
    spec = AblationSpec(
        index = 7,
        label = "broken",
        overrides = {},
        train_kwargs = {"run_name": "broken"},
        run_log_dir = "runs/gym_imagination/broken",
    )

    def raise_error(_):
        raise RuntimeError("test failure")

    monkeypatch.setattr(ablations, "run_train_gym", raise_error)

    result = execute_ablation(spec, dry_run = False)

    captured = capsys.readouterr()

    assert not result.success
    assert "RuntimeError: test failure" in result.error
    assert "[ablation 7] worker error for broken:" in captured.err
    assert "RuntimeError: test failure" in captured.err


def test_selected_reinforce_ablations_cover_three_paired_concepts():
    selected = [ablations.ABLATIONS[index - 1] for index in ablations.ABLATIONS_TO_RUN]
    specs = build_ablation_specs()

    assert len(specs) == 6
    assert [item["seed"] for item in selected] == [42, 4, 42, 4, 42, 4]
    assert all(item["objective"] == "reinforce" for item in selected)
    assert all(item["run_details"].startswith("prev=[") for item in selected)
    assert all(", change=[" in item["run_details"] for item in selected)

    baseline, targets, actor_loss = selected[:2], selected[2:4], selected[4:]
    assert all("reinforce_gae_lambda" not in item for item in baseline)
    assert all(item["reinforce_gae_lambda"] == 0.95 for item in targets)
    assert all(item["reinforce_return_normalization"] == "percentile" for item in targets)
    assert all(item["reinforce_use_discounted_trajectory_weights"] is True for item in actor_loss)
    assert all(item["continuous_dist_type"] == "squashed_gaussian" for item in actor_loss)
    assert all(item["policy_entropy_weight"] == 0.0003 for item in actor_loss)
