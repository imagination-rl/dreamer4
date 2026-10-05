"""How each world model rates the real action segments that flip the cheetah, against the segments that do not.

Segments come from the replays of the realistic-data RL runs (entries 58/59), loops 10-29, where the flips start.
Each segment: 8 real prompt frames (upright at the last frame), then the 32 real policy actions, open loop. The script
resimulates the same actions in MuJoCo from the restored start state, and splits the segments by the MuJoCo outcome:
"real flip" (flipped at the end) and "no real flip". For each model it gives the dream reward per step, the true reward
per step, and the dream flip rate at the end (decoded root angle). A model that rates the flip segments above the
no-flip segments gives the policy a reason to flip. Two more groups compare the gaits of the later loops: the flipped
gait of the collapsed s4 run (loops 50-99) and the upright gait of the s42 run (loops 80-99).

Output: report/10-1-26-realistic-data-analysis/data/probe_flip_segments.csv
Usage:  python report/10-1-26-realistic-data-analysis/scripts/probe_with_runs.py \
            report/10-1-26-realistic-data-analysis/scripts/probe_flip_segments.py [--cuda]
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
from train.audit_ablations import (AuditDiagnostics, HALF_CHEETAH_ROOT_ANGLE_INDEX, _frame_latents, _numpy, _wrap_angle,
                                   native_to_env_actions)

G = REPO / "runs" / "gym_imagination"
P = REPO / "runs" / "offline_pretrain"
REPLAYS = [G / "pretrain_rl_realistic_data_s42" / "replay_buffer", G / "pretrain_rl_realistic_data_s4" / "replay_buffer"]
MODELS = [
    ("realistic s42 pretrain", "pretrain_rl_realistic_data_s42", P / "realistic_data_s42" / "pretrained.pt"),
    ("realistic s4 pretrain", "pretrain_rl_realistic_data_s4", P / "realistic_data_s4" / "pretrained.pt"),
    ("scaled40 s42 pretrain", "offline_pretrained_a2_scaled40_no_rehearsal_s42", P / "a2_scaled40_s42" / "pretrained.pt"),
    ("scaled40 s4 pretrain", "offline_pretrained_a2_scaled40_no_rehearsal_s4", P / "a2_scaled40_s4" / "pretrained.pt"),
    ("no pretraining: repro s4 loop 50", "audit_a2_next_state_reward_target2", G / "audit_a2_next_state_reward_target_local_s4" / "checkpoints" / "loop_50.pt"),
]
PROMPT, HORIZON, PER_GROUP, LOOPS, CHUNK = 8, 32, 256, (10, 30), 128
ACTION_RANGE = (-1., 1.)
OUT = HERE.parent / "data" / "probe_flip_segments.csv"


def flipped(obs):
    return np.abs(_wrap_angle(obs[..., HALF_CHEETAH_ROOT_ANGLE_INDEX])) > np.pi / 2


def segments(rng):
    """Equal numbers of flip and no-flip segments from each replay. Returns prompt obs, native actions, label."""

    obs_all, act_all, label_all = [], [], []
    for folder in REPLAYS:
        replay = ReplayBuffer.from_folder(str(folder), read_only = True)
        data = replay.data
        episodes = np.arange(64 * LOOPS[0], 64 * LOOPS[1])
        state = np.asarray(data["critic_state"][episodes[0]:episodes[-1] + 1, :1000])
        flip = flipped(state)
        onset = [np.flatnonzero(~f[:-1] & f[1:]) + 1 for f in flip]
        flip_starts, calm_starts = [], []
        for k, e in enumerate(episodes):
            if len(onset[k]):
                # The prompt ends 1-28 steps before the first flip.
                start = onset[k][0] - PROMPT - rng.integers(1, HORIZON - 3)
                if 0 <= start and start + PROMPT + HORIZON <= 1000:
                    flip_starts.append((k, e, start))
            # A window that stays upright for the prompt and the horizon.
            upright_until = onset[k][0] if len(onset[k]) else 1000
            if upright_until > PROMPT + HORIZON + 1:
                calm_starts.append((k, e, rng.integers(0, upright_until - PROMPT - HORIZON)))
        for picks, label in ((flip_starts, "real flip"), (calm_starts, "no real flip")):
            chosen = rng.choice(len(picks), size = min(PER_GROUP // len(REPLAYS), len(picks)), replace = False)
            for c in chosen:
                k, e, s = picks[c]
                obs_all.append(state[k, s:s + PROMPT])
                act_all.append(np.asarray(data["actions_continuous"][e, s:s + PROMPT - 1 + HORIZON]))
                label_all.append(label)
    # Two gaits of the later loops: the flipped gait of the collapsed s4 run (loops 50-99, flipped for the whole
    # window) and the upright gait of the s42 run (loops 80-99, upright for the whole window).
    for folder, loops, want_flipped, label in ((REPLAYS[1], (50, 100), True, "flipped gait"), (REPLAYS[0], (80, 100), False, "upright gait")):
        data = ReplayBuffer.from_folder(str(folder), read_only = True).data
        episodes = rng.integers(64 * loops[0], 64 * loops[1], size = 4 * PER_GROUP)
        found = 0
        for e in episodes:
            s = int(rng.integers(0, 1000 - PROMPT - HORIZON))
            window = np.asarray(data["critic_state"][e, s:s + PROMPT + HORIZON])
            if (flipped(window).all() if want_flipped else (~flipped(window)).all()):
                obs_all.append(window[:PROMPT])
                act_all.append(np.asarray(data["actions_continuous"][e, s:s + PROMPT - 1 + HORIZON]))
                label_all.append(label)
                found += 1
                if found == PER_GROUP // 2:
                    break
    return np.stack(obs_all).astype(np.float32), np.stack(act_all).astype(np.float32), np.array(label_all)


def main():
    device = torch.device("cuda" if "--cuda" in sys.argv else "cpu")
    rng = np.random.default_rng(0)
    obs, native, labels = segments(rng)
    start_obs = obs[:, -1]
    audit = AuditDiagnostics(
        env_name = "HalfCheetah-v4", obs_dim = 17, action_range = ACTION_RANGE, prompt_length = PROMPT, horizon = HORIZON,
        generate_steps = 4, use_time_cache = True, old_reward_shifting = False, every_env_steps = 1, seed = 0, device = device,
    )
    env_actions = np.clip(native_to_env_actions(native[:, PROMPT - 1:], ACTION_RANGE), -1 + 1e-3, 1 - 1e-3)
    true_rewards, true_obs = audit.resimulate(start_obs, env_actions)
    true_end = flipped(true_obs[:, -1])
    # The MuJoCo resimulation decides the group (the restored state can differ a little from the replay).
    group = np.where(np.isin(labels, ["flipped gait", "upright gait"]), labels, np.where(true_end, "real flip", "no real flip"))
    print("replay label vs resimulated outcome agreement:", float((group == labels).mean()), flush = True)
    print({name: int((group == name).sum()) for name in dict.fromkeys(group)}, flush = True)
    dream_slice, true_slice = audit.aligned_reward_slices(HORIZON)
    true_step = true_rewards[:, true_slice].mean(-1)

    rows = []
    for label, run, checkpoint in MODELS:
        seed_all(0)
        _, tokenizer, world_model, _, _ = build(run, str(checkpoint), device = device)
        dream_step, dream_end = [], []
        for c in range(0, len(obs), CHUNK):
            sl = slice(c, c + CHUNK)
            with torch.no_grad():
                prompt_latents = tokenizer.encode(torch.from_numpy(obs[sl]).to(device)).float()
                generated = world_model.generate(
                    PROMPT + HORIZON, num_steps = 4, batch_size = len(obs[sl]), return_decoded_video = False,
                    return_agent_actions = False, return_rewards_per_frame = True, return_terminals = False,
                    use_time_cache = True, prompt_latents = prompt_latents,
                    prompt_continuous_actions = torch.from_numpy(native[sl]).to(device),
                )
                decoded = _numpy(tokenizer.decode(_frame_latents(generated.latents)[:, PROMPT + HORIZON - 1]))
            dream_step.append(_numpy(generated.rewards)[:, dream_slice].mean(-1))
            dream_end.append(flipped(decoded.reshape(len(obs[sl]), -1)))
        dream_step, dream_end = np.concatenate(dream_step), np.concatenate(dream_end)
        for name in ("real flip", "no real flip", "flipped gait", "upright gait"):
            m = group == name
            rows.append(dict(model = label, segment = name, windows = int(m.sum()),
                             dream_reward_per_step = float(dream_step[m].mean()), true_reward_per_step = float(true_step[m].mean()),
                             gap = float((dream_step - true_step)[m].mean()), dream_flipped_at_end = float(dream_end[m].mean()),
                             true_flipped_at_end = float(true_end[m].mean()),
                             saturated = float((np.abs(env_actions[m]) > 0.95).mean()), vx_start = float(start_obs[m, 8].mean())))
            print(rows[-1], flush = True)
    audit.close()
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index = False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index = False))
    wide = table.pivot(index = "model", columns = "segment", values = "dream_reward_per_step")
    print("\n dream reward per step: real flip minus no real flip")
    print((wide["real flip"] - wide["no real flip"]).round(3).to_string())


if __name__ == "__main__":
    main()
