"""How much does each RL policy act like the expert? Cosine similarity of the policy action and the CleanRL TD3
expert action at the same (replay) state, and the size ratio |a_policy| / |a_expert|.

Upright states only (the flipped gait is not comparable). Loops 40-49 of each run's own replay.
Output: report/9-30-26-scaled40-collapse/data/expert_likeness.csv
Usage:  python report/9-30-26-scaled40-collapse/scripts/expert_likeness.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from memmap_replay_buffer.replay_buffer import ReplayBuffer

from train import offline_pretraining as off
from train.audit_ablations import HALF_CHEETAH_ROOT_ANGLE_INDEX as ANGLE, _wrap_angle, native_to_env_actions

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
G = REPO / "runs" / "gym_imagination"
EXPERT = next((REPO / "runs" / "offline_data" / "experts").rglob("*.cleanrl_model"))
RUNS = {
    "scaled40 reh s42": G / "offline_pretrained_a2_scaled40_rehearsal_s42" / "checkpoints" / "replay_loop_50",
    "scaled40 reh s4": G / "offline_pretrained_a2_scaled40_rehearsal_s4" / "checkpoints" / "replay_loop_50",
    "old reh s42": G / "offline_pretrained_a2_rehearsal_s42" / "checkpoints" / "replay_loop_100",
    "old reh s4": G / "offline_pretrained_a2_rehearsal_s4" / "checkpoints" / "replay_loop_100",
    "old no reh s42": G / "offline_pretrained_a2_no_rehearsal_s42" / "replay_buffer",
    "no pretraining repro s4": G / "audit_a2_next_state_reward_target_local_s4" / "checkpoints" / "replay_loop_100",
    "no pretraining repro s42": G / "audit_a2_next_state_reward_target_local_s42" / "checkpoints" / "replay_loop_100",
}
LOOPS, STEPS = (40, 50), 200
OUT = HERE.parent / "data" / "expert_likeness.csv"


def main():
    actor = off.load_expert_actor(EXPERT, obs_dim = 17, action_dim = 6)
    rng = np.random.default_rng(0)
    rows = []
    for label, folder in RUNS.items():
        replay = ReplayBuffer.from_folder(folder, read_only = True)
        episodes = np.arange(64 * LOOPS[0], 64 * LOOPS[1])
        obs = np.asarray(replay.data["critic_state"][episodes, :1000], dtype = np.float32)
        act = native_to_env_actions(np.asarray(replay.data["actions_continuous"][episodes, :1000], dtype = np.float32), (-1., 1.))
        pick = rng.integers(0, 1000, size = (len(episodes), STEPS))
        obs = np.take_along_axis(obs, pick[..., None], 1).reshape(-1, 17)
        act = np.take_along_axis(act, pick[..., None], 1).reshape(-1, 6)
        upright = np.abs(_wrap_angle(obs[:, ANGLE])) <= np.pi / 2
        obs, act = obs[upright], act[upright]
        expert = off.expert_actions(actor, obs, 6)
        cos = (act * expert).sum(-1) / (np.linalg.norm(act, axis = -1) * np.linalg.norm(expert, axis = -1) + 1e-8)
        rows.append(dict(run = label, upright_steps = int(upright.sum()), upright_frac = float(upright.mean()),
                         cos_to_expert_mean = float(cos.mean()), cos_above_0p7_frac = float((cos > 0.7).mean()),
                         size_ratio = float(np.linalg.norm(act, axis = -1).mean() / np.linalg.norm(expert, axis = -1).mean()),
                         mean_abs_action = float(np.abs(act).mean()), x_velocity = float(obs[:, 8].mean())))
        print(rows[-1], flush = True)
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents = True, exist_ok = True)
    table.to_csv(OUT, index = False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index = False))


if __name__ == "__main__":
    main()
