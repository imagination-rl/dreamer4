"""Closed-loop dream against MuJoCo, split by flipped and upright start states.

Each model dreams 32 steps with its own policy from replay prompts: the loop-100 replay of the collapsed repro run
(many flipped states), or the model's own loop-100 replay. The dream actions then run in MuJoCo from the restored start state.
For each group of start states, the script reports the dream and true reward per step and the flip rate at the end.

Output: report/9-30-26-repro/data/probe_flip_dreams.csv
Usage:  python report/9-30-26-repro/scripts/probe_flip_dreams.py [--generate-steps 64] [--only <label part>]
        [--model 'label|run|checkpoint|replay folder' ...] [--out csv]   (--model replaces MODELS)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "reports" / "9-28-26" / "scripts"))

import numpy as np
import pandas as pd
import torch
from memmap_replay_buffer import ReplayBuffer

from probe_common import build, seed_all
from train.audit_ablations import AuditDiagnostics, HALF_CHEETAH_ROOT_ANGLE_INDEX, _frame_latents, _numpy, _wrap_angle

RUNS = REPO / "runs" / "gym_imagination"
S42_REPLAY = RUNS / "audit_a2_next_state_reward_target_local_s42" / "checkpoints" / "replay_loop_100"
S4_REPLAY = RUNS / "audit_a2_next_state_reward_target_local_s4" / "checkpoints" / "replay_loop_100"
# (label, run whose hyperparameters build the model, checkpoint, replay of the prompts)
MODELS = [
    ("repro s42 loop 50", "audit_a2_next_state_reward_target", RUNS / "audit_a2_next_state_reward_target_local_s42" / "checkpoints" / "loop_50.pt", S42_REPLAY),
    ("repro s42 loop 100", "audit_a2_next_state_reward_target", RUNS / "audit_a2_next_state_reward_target_local_s42" / "checkpoints" / "loop_100.pt", S42_REPLAY),
    ("repro s4 loop 100", "audit_a2_next_state_reward_target2", RUNS / "audit_a2_next_state_reward_target_local_s4" / "checkpoints" / "loop_100.pt", S42_REPLAY),
    ("repro s4 loop 100, own replay", "audit_a2_next_state_reward_target2", RUNS / "audit_a2_next_state_reward_target_local_s4" / "checkpoints" / "loop_100.pt", S4_REPLAY),
    ("repro s4 loop 50, own replay", "audit_a2_next_state_reward_target2", RUNS / "audit_a2_next_state_reward_target_local_s4" / "checkpoints" / "loop_50.pt", S4_REPLAY),
    # The H100 runs kept no replay snapshot, so they use the prompts of the collapsed repro run.
    ("H100 s42 loop 75", "audit_a2_next_state_reward_target", RUNS / "audit_a2_next_state_reward_target" / "checkpoints" / "loop_75.pt", S42_REPLAY),
    ("H100 s42 loop 125", "audit_a2_next_state_reward_target", RUNS / "audit_a2_next_state_reward_target" / "checkpoints" / "loop_125.pt", S42_REPLAY),
]
PROMPT, HORIZON, WINDOWS, CHUNK = 8, 32, 1024, 256
# Denoising steps of each generated state (4 in training). "--generate-steps 64 --only repro" checks whether the
# missing flips come from the coarse 4-step sampling.
GENERATE_STEPS = int(sys.argv[sys.argv.index("--generate-steps") + 1]) if "--generate-steps" in sys.argv else 4
ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
OUT = HERE.parent / "data" / ("probe_flip_dreams.csv" if GENERATE_STEPS == 4 else f"probe_flip_dreams_generate{GENERATE_STEPS}.csv")
OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else OUT
_GIVEN = [sys.argv[i + 1].split("|") for i, arg in enumerate(sys.argv) if arg == "--model"]
if _GIVEN:
    MODELS = [(label, run, REPO / checkpoint, REPO / replay) for label, run, checkpoint, replay in _GIVEN]


def is_flipped(obs: np.ndarray) -> np.ndarray:
    # Both directions of the root angle (the audit metric counts only angle > pi/2).
    return np.abs(_wrap_angle(obs[..., HALF_CHEETAH_ROOT_ANGLE_INDEX])) > np.pi / 2


def main():
    device = torch.device("cuda")
    rows = []

    for label, run, checkpoint, replay_folder in MODELS:
        if ONLY is not None and ONLY not in label:
            continue
        replay = ReplayBuffer.from_folder(replay_folder, read_only = True)
        lengths = np.asarray(replay.episode_lens)
        data = replay.data
        if not checkpoint.is_file():
            print("missing", checkpoint)
            continue
        seed_all(0)
        _, tokenizer, world_model, _, action_range = build(run, str(checkpoint), device = device)
        audit = AuditDiagnostics(
            env_name = "HalfCheetah-v4", obs_dim = 17, action_range = action_range, prompt_length = PROMPT, horizon = HORIZON,
            generate_steps = GENERATE_STEPS, use_time_cache = True, old_reward_shifting = False, every_env_steps = 1, seed = 0, device = device,
        )
        episodes, starts = audit.sample_windows(lengths, WINDOWS, PROMPT)
        dream_step, true_step, start_flip, true_end, dream_end = [], [], [], [], []

        for chunk in range(0, len(episodes), CHUNK):
            e, s = episodes[chunk:chunk + CHUNK], starts[chunk:chunk + CHUNK]
            latents = np.stack([data["latents"][i, j:j + PROMPT] for i, j in zip(e, s)])
            prompt_actions = np.stack([data["actions_continuous"][i, j:j + PROMPT] for i, j in zip(e, s)])
            start_obs = np.stack([data["critic_state"][i, j + PROMPT - 1] for i, j in zip(e, s)])

            with torch.no_grad():
                generated = audit.generate(world_model, PROMPT + HORIZON, _frame_latents(torch.from_numpy(latents)).float(),
                                           torch.from_numpy(prompt_actions).float(), return_agent_actions = True)
            env_actions = audit.to_env_actions(_numpy(generated.actions.continuous)[:, PROMPT - 1:PROMPT - 1 + HORIZON])
            true_rewards, true_obs = audit.resimulate(start_obs, env_actions)
            dream_slice, true_slice = audit.aligned_reward_slices(HORIZON)
            decoded = _numpy(tokenizer.decode(_frame_latents(generated.latents)[:, PROMPT:PROMPT + HORIZON].flatten(0, 1))).reshape(len(e), HORIZON, -1)

            dream_step.append(_numpy(generated.rewards)[:, dream_slice].mean(-1))
            true_step.append(true_rewards[:, true_slice].mean(-1))
            start_flip.append(is_flipped(start_obs))
            true_end.append(is_flipped(true_obs[:, -1]))
            dream_end.append(is_flipped(decoded[:, -1]))

        audit.close()
        dream_step, true_step, start_flip, true_end, dream_end = map(np.concatenate, (dream_step, true_step, start_flip, true_end, dream_end))

        groups = (
            ("all", np.ones_like(start_flip)), ("start upright", ~start_flip), ("start flipped", start_flip),
            # Upright starts split by the MuJoCo outcome: does the dream see the flip that the policy causes?
            ("upright, real flip", ~start_flip & true_end), ("upright, no real flip", ~start_flip & ~true_end),
        )
        for group, mask in groups:
            if mask.sum() < 2:
                continue
            rows.append(dict(
                model = label, start = group, windows = int(mask.sum()),
                dream_reward_per_step = float(dream_step[mask].mean()), true_reward_per_step = float(true_step[mask].mean()),
                gap_per_step = float((dream_step - true_step)[mask].mean()),
                dream_above_true_frac = float((dream_step > true_step)[mask].mean()),
                return_corr = float(np.corrcoef(dream_step[mask], true_step[mask])[0, 1]),
                true_flipped_at_end = float(true_end[mask].mean()), dream_flipped_at_end = float(dream_end[mask].mean()),
            ))
            print(rows[-1], flush = True)

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents = True, exist_ok = True)
    table.to_csv(OUT, index = False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index = False))


if __name__ == "__main__":
    main()
