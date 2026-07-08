from __future__ import annotations

import pytest
import yaml

from train.ablations import build_ablation_specs


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
    assert "Ablation 0: depth8" in first.train_kwargs["run_details"]
    assert "Depth-8 ablation" in first.train_kwargs["run_details"]

    second = specs[1]
    assert second.label == "wider"
    assert second.train_kwargs["model_dim"] == 256
    assert second.train_kwargs["depth"] == 3
    assert second.train_kwargs["run_name"] == "wider"
    assert second.run_log_dir == "runs/gym_imagination/wider"


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
