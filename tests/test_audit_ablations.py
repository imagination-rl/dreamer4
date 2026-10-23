from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from train import ablations, train_gym
from train.audit_ablations import (
    AUDIT_ABLATION_FLAGS,
    AuditDiagnostics,
    isolated_rng,
    resolve_audit_ablations,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "train" / "config.yaml"


def resolve_from(config, **flags):
    return resolve_audit_ablations(
        audit_deeper_dynamics = flags.get("audit_deeper_dynamics", config["audit_deeper_dynamics"]),
        audit_arrival_reward_target = flags.get("audit_arrival_reward_target", config["audit_arrival_reward_target"]),
        audit_quarter_envs_per_loop = flags.get("audit_quarter_envs_per_loop", config["audit_quarter_envs_per_loop"]),
        depth = config["depth"],
        time_block_every = config["time_block_every"],
        old_reward_shifting = config["old_reward_shifting"],
        align_generated_dream_transitions = config["align_generated_dream_transitions"],
        num_envs = config["total_envs"],
        replay_size = config["replay_size"],
        num_loops = config["num_loops"],
        checkpoint_every = config["checkpoint_every"],
    )


def test_default_config_resolves_no_audit_overrides():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert resolve_from(config) == {}
    assert not config["audit_diagnostics"]


def test_each_audit_flag_replaces_only_its_own_values():
    config = dict(
        yaml.safe_load(CONFIG_PATH.read_text()),
        depth = 3,
        time_block_every = 2,
        old_reward_shifting = False,
        align_generated_dream_transitions = False,
        num_envs = 64,
        replay_size = 256,
        num_loops = 500,
        checkpoint_every = 25,
    )

    assert resolve_from(config, audit_deeper_dynamics = True) == {"depth": 5}
    assert resolve_from(config, audit_arrival_reward_target = True) == {
        "old_reward_shifting": True,
        "align_generated_dream_transitions": True,
    }
    assert resolve_from(config, audit_quarter_envs_per_loop = True) == {
        "num_envs": 16,
        "replay_size": 1024,
        "num_loops": 2000,
        "checkpoint_every": 100,
    }


def test_audit_flags_reject_layouts_they_cannot_extend():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    # time_block_every=1 has no spatial layer, and an even depth ends on a temporal layer.
    with pytest.raises(ValueError, match = "odd depth"):
        resolve_from(dict(config, time_block_every = 1), audit_deeper_dynamics = True)

    with pytest.raises(ValueError, match = "odd depth"):
        resolve_from(dict(config, depth = 4), audit_deeper_dynamics = True)

    with pytest.raises(ValueError, match = "divisible by 4"):
        resolve_from(dict(config, total_envs = 30), audit_quarter_envs_per_loop = True)


def test_each_audit_arm_differs_from_its_control_by_one_audit_flag():
    audit_indices = [
        index
        for index, ablation in enumerate(ablations.ABLATIONS, 1)
        if ablation["run_name"].startswith("audit_")
    ]
    specs = {
        spec.train_kwargs["run_name"]: spec.train_kwargs
        for spec in ablations.build_ablation_specs(
            ablation_indices = audit_indices,
        )
    }
    ignored = {"run_name", "run_details", "checkpoint_folder"}

    # A2 removes the arrival reward target of the PPO control, so it changes the two keys that the A2 flag sets.
    arms = {
        "audit_a1_deeper_dynamics": {"audit_deeper_dynamics"},
        "audit_a2_next_state_reward_target": {"old_reward_shifting", "align_generated_dream_transitions"},
        "audit_a3_quarter_envs_per_loop": {"audit_quarter_envs_per_loop"},
    }

    for arm, flags in arms.items():
        for suffix, seed in (("", 42), ("2", 4)):
            # A completed arm seed can leave the run list.
            if arm + suffix not in specs:
                continue

            arm_kwargs = specs[arm + suffix]
            control_kwargs = specs["audit_control" + suffix]
            changed = {
                key
                for key in arm_kwargs.keys() | control_kwargs.keys()
                if key not in ignored and arm_kwargs.get(key) != control_kwargs.get(key)
            }

            assert changed == flags
            assert arm_kwargs["seed"] == control_kwargs["seed"] == seed
            assert arm_kwargs["audit_diagnostics"]
            assert control_kwargs["cleanrl_ppo_closer"]
            assert not any(control_kwargs[name] for name in AUDIT_ABLATION_FLAGS)

    # The A2 flag on the A2 arm gives back the control reward target.
    a2 = specs["audit_a2_next_state_reward_target"]
    assert resolve_from(a2, audit_arrival_reward_target = True) == {
        key: specs["audit_control"][key] for key in arms["audit_a2_next_state_reward_target"]
    }


def test_aligned_reward_slices_pair_the_same_transition():
    def diagnostics(old_reward_shifting):
        return AuditDiagnostics(
            env_name = "HalfCheetah-v4",
            obs_dim = 17,
            action_range = (-1., 1.),
            prompt_length = 8,
            horizon = 32,
            generate_steps = 4,
            use_time_cache = True,
            old_reward_shifting = old_reward_shifting,
            every_env_steps = 1,
            seed = 0,
            device = torch.device("cpu"),
        )

    # True reward t belongs to transition P-1+t. Generated frame i reports the
    # transition i-1 with shifting and the transition i+1 without it.
    for old_reward_shifting, frame_to_transition in ((True, lambda i: i - 1), (False, lambda i: i + 1)):
        dream_slice, true_slice = diagnostics(old_reward_shifting).aligned_reward_slices(32)
        dream_frames = range(dream_slice.start, dream_slice.stop)
        true_transitions = [7 + t for t in range(true_slice.start, true_slice.stop)]

        assert [frame_to_transition(i) for i in dream_frames] == true_transitions


def test_isolated_rng_leaves_the_training_stream_unchanged():
    torch.manual_seed(0)
    expected = torch.rand(3)

    torch.manual_seed(0)
    with isolated_rng(torch.device("cpu")):
        torch.rand(100)

    assert torch.equal(torch.rand(3), expected)


def test_main_logs_audit_diagnostics_with_all_audit_flags(tmp_path):
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config.update(
        log_dir = str(tmp_path),
        run_name = "audit",
        run_details = "audit integration test",
        cpu = True,
        require_cuda = False,
        compile = False,
        compile_interact = False,
        compile_world_model = False,
        compile_generate = False,
        compile_learn = False,
        num_loops = 1,
        total_envs = 8,
        env_vectorization_mode = "sync",
        max_timesteps = 48,
        replay_size = 2,
        model_dim = 32,
        attn_heads = 2,
        attn_dim_head = 16,
        pretrain_tokenizer_steps = 2,
        pretrain_tokenizer_observations = 64,
        tokenizer_batch_size = 32,
        pretrain_world_model_combined_steps = 1,
        world_model_train_steps = 1,
        world_model_batch_size = 2,
        world_model_train_sequence_length = 16,
        imagination_batch_size = 4,
        imagination_horizon = 8,
        imagination_prompt_length = 4,
        imagination_train_steps = 1,
        preservation_sequence_length = 16,
        checkpoint_every = 0,
        tokenizer_eval_every = 0,
        collect_timings = False,
        audit_deeper_dynamics = True,
        audit_arrival_reward_target = True,
        audit_quarter_envs_per_loop = True,
        audit_diagnostics = True,
        audit_diagnostics_every_env_steps = 1,
    )

    # num_loops = 1 becomes 4 loops with 2 envs each.
    train_gym.main(**config)

    accumulator = EventAccumulator(str(tmp_path / "audit")).Reload()
    scalars = accumulator.Tags()["scalars"]
    hyperparameters = json.loads(
        accumulator.Tensors("run/hyperparameters/text_summary")[0]
        .tensor_proto.string_val[0].decode()
        .strip("`").replace("json", "", 1)
    )

    # The default config already sets old_reward_shifting=true.
    assert hyperparameters["audit_effective_overrides"] == {
        "depth": 5,
        "align_generated_dream_transitions": True,
        "num_envs": 2,
        "replay_size": 8,
        "num_loops": 4,
        "checkpoint_every": 0,
    }
    assert [event.step for event in accumulator.Scalars("audit/progress/loop")] == [96, 192, 288, 384]
    assert [event.value for event in accumulator.Scalars("audit/sim/restore_ok")] == [1.] * 3

    for tag in (
        "audit/rollout/flipped_frac",
        "audit/world_model/one_step_norm_mae",
        "audit/world_model/reward_head_corr",
        "audit/world_model/ranking_pairwise_sign_agreement",
        "audit/closed_loop/reward_gap_per_step",
        "audit/closed_loop/state_norm_mae_h8",
    ):
        assert tag in scalars
