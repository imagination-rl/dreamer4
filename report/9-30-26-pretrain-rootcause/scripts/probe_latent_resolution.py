"""Tokenizer geometry against dynamics error on online states.

For each model (its own tokenizer and world model), on the same online windows of the repro s4 replay:
  - velocity resolution: the latent L2 change for +0.1 m/s of forward velocity (obs index 8), and the ratio of that
    change to the latent L2 change of one real env step (the signal that the dynamics loss must resolve);
  - one-step prediction with the true prompt actions (8 prompt frames): the latent RMSE against the encoded true next
    state, the latent RMSE relative to the true one-step latent change, and the decoded forward-velocity error in m/s
    (absolute, signed bias, and the slope of predicted on true velocity).

Output: report/9-30-26-pretrain-rootcause/data/probe_latent_resolution.csv
Usage:  python report/9-30-26-pretrain-rootcause/scripts/probe_latent_resolution.py [--cuda] [--model 'label|run|checkpoint' ...] [--out csv]
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
from train.audit_ablations import HALF_CHEETAH_X_VELOCITY_INDEX as VX, _frame_latents, _numpy

G = REPO / "runs" / "gym_imagination"
P = REPO / "runs" / "offline_pretrain"
REPLAY = G / "audit_a2_next_state_reward_target_local_s4" / "checkpoints" / "replay_loop_100"
MODELS = [
    ("pretrained s4 step 12000 (RL start)", "audit_a2_next_state_reward_target2", P / "a2_next_state_reward_s4" / "pretrained_step_12000.pt"),
    ("pretrained s42 step 12000 (RL start)", "audit_a2_next_state_reward_target", P / "a2_next_state_reward_s42" / "pretrained_step_12000.pt"),
    ("repro s4 loop 50", "audit_a2_next_state_reward_target2", G / "audit_a2_next_state_reward_target_local_s4" / "checkpoints" / "loop_50.pt"),
    ("H100 s4 loop 50", "audit_a2_next_state_reward_target2", G / "audit_a2_next_state_reward_target2" / "checkpoints" / "loop_50.pt"),
    ("rehearsal 0.5 s4 loop 50", "audit_a2_next_state_reward_target2", G / "offline_pretrained_a2_rehearsal_s4" / "checkpoints" / "loop_50.pt"),
    ("BC s4 loop 50 (pretrained, no rehearsal)", "audit_a2_next_state_reward_target2", G / "offline_pretrained_a2_bc_no_rehearsal_s4" / "checkpoints" / "loop_50.pt"),
]
SUBSETS = {"online loops 0-9": (0, 10), "online loops 45-54": (45, 55), "online loops 90-99": (90, 100)}
PROMPT, WINDOWS = 8, 512
OUT = HERE.parent / "data" / "probe_latent_resolution.csv"



def models_from_argv(default):
    """--model 'label|run|checkpoint' (repeatable) replaces MODELS. run gives the architecture config."""

    given = [sys.argv[i + 1] for i, arg in enumerate(sys.argv) if arg == "--model"]
    return [tuple(value.split("|")) for value in given] if given else default


def out_from_argv(default):
    return Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else default


def main():
    # CPU by default: a training run can hold most of the GPU memory. "--cuda" uses the GPU.
    device = torch.device("cuda" if "--cuda" in sys.argv else "cpu")
    replay = ReplayBuffer.from_folder(REPLAY, read_only = True)
    lengths = np.asarray(replay.episode_lens)
    obs_all, act_all = replay.data["critic_state"], replay.data["actions_continuous"]
    rng = np.random.default_rng(0)
    samples = {}

    for name, (lo, hi) in SUBSETS.items():
        episodes = rng.integers(64 * lo, 64 * hi, size = WINDOWS)
        starts = np.array([rng.integers(0, lengths[e] - PROMPT - 1) for e in episodes])
        obs = np.stack([obs_all[e, s:s + PROMPT + 1] for e, s in zip(episodes, starts)]).astype(np.float32)
        actions = np.stack([act_all[e, s:s + PROMPT + 1] for e, s in zip(episodes, starts)]).astype(np.float32)
        samples[name] = (obs, actions)

    rows = []
    for label, run, checkpoint in models_from_argv(MODELS):
        seed_all(0)
        _, tokenizer, world_model, _, _ = build(run, str(checkpoint), device = device)

        for name, (obs, actions) in samples.items():
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).to(device)
                latents = tokenizer.encode(obs_t)                                  # (B, T, n, d)

                # Velocity resolution around the states of this subset.
                bumped = obs_t.clone()
                bumped[..., VX] += 0.1
                dv_latent = (tokenizer.encode(bumped) - latents).flatten(-2).norm(dim = -1).mean().item()
                step_latent = (latents[:, 1:] - latents[:, :-1]).flatten(-2).norm(dim = -1).mean().item()

                generated = world_model.generate(
                    PROMPT + 1, num_steps = 4, batch_size = len(obs), return_decoded_video = False,
                    return_agent_actions = False, return_rewards_per_frame = True, return_terminals = False,
                    use_time_cache = True, prompt_latents = latents[:, :PROMPT].float(),
                    prompt_continuous_actions = torch.from_numpy(actions[:, :PROMPT]).to(device),
                )
                predicted = _frame_latents(generated.latents)[:, PROMPT]
                target = latents[:, PROMPT]
                latent_rmse = (predicted - target).flatten(-2).norm(dim = -1).mean().item()
                true_change = (target - latents[:, PROMPT - 1]).flatten(-2).norm(dim = -1).mean().item()
                decoded = _numpy(tokenizer.decode(predicted))

            rows.append(dict(
                model = label, subset = name,
                obs_std_vx = float(tokenizer.obs_std[VX]), obs_mean_vx = float(tokenizer.obs_mean[VX]),
                latent_change_per_0p1_mps = dv_latent, latent_change_per_env_step = step_latent,
                one_step_latent_error = latent_rmse, one_step_latent_error_over_true_change = latent_rmse / max(true_change, 1e-8),
                one_step_vx_error_mps = float(np.abs(decoded[:, VX] - obs[:, PROMPT, VX]).mean()),
                # Signed: > 0 means the model predicts a faster next state than the env gives.
                one_step_vx_bias_mps = float((decoded[:, VX] - obs[:, PROMPT, VX]).mean()),
                # Slope of the predicted on the true next velocity (1 is ideal, 0 ignores the true velocity).
                one_step_vx_slope = float(np.polyfit(obs[:, PROMPT, VX], decoded[:, VX], 1)[0]),
                one_step_vx_error_copy_last_mps = float(np.abs(obs[:, PROMPT - 1, VX] - obs[:, PROMPT, VX]).mean()),
            ))
            print(rows[-1], flush = True)

    table = pd.DataFrame(rows)
    out = out_from_argv(OUT)
    out.parent.mkdir(parents = True, exist_ok = True)
    table.to_csv(out, index = False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index = False))


if __name__ == "__main__":
    main()
