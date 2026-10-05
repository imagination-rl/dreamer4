"""One row for each pretrained RL run and each online-only A2 run (loops 0-99), from TensorBoard.

Output: report/9-30-26-pretrain-analysis/data/{loops.csv, runs.csv}
Usage:  python report/9-30-26-pretrain-analysis/scripts/run_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "report" / "9-29-26-pretrain" / "scripts"))
from collapse_logs import load  # noqa: E402

G = "runs/gym_imagination/"
# name: (path, group, pretrain, rehearsal, seed)
RUNS = {
    "h100_s42": (G + "audit_a2_next_state_reward_target", "online", "-", 0.0, 42),
    "h100_s4": (G + "audit_a2_next_state_reward_target2", "online", "-", 0.0, 4),
    "local928_s42": ("runs/local_9-28-26/attnpool_on_s42", "online", "-", 0.0, 42),
    "local928_s4": ("runs/local_9-28-26/attnpool_on_s4", "online", "-", 0.0, 4),
    "local928_s1": ("runs/local_9-28-26/attnpool_on_s1", "online", "-", 0.0, 1),
    "repro_s42": (G + "audit_a2_next_state_reward_target_local_s42", "online", "-", 0.0, 42),
    "repro_s4": (G + "audit_a2_next_state_reward_target_local_s4", "online", "-", 0.0, 4),
    "1m_a_reh50_s42": (G + "offline_pretrained_a2_rehearsal", "pretrained", "1m_a", 0.5, 42),
    "1m_a_reh50_s4": (G + "offline_pretrained_a2_rehearsal2", "pretrained", "1m_a", 0.5, 4),
    "1m_a_noreh_s42": (G + "offline_pretrained_a2_no_rehearsal", "pretrained", "1m_a", 0.0, 42),
    "1m_a_noreh_s4": (G + "offline_pretrained_a2_no_rehearsal2", "pretrained", "1m_a", 0.0, 4),
    "1m_b_reh50_s42": (G + "offline_pretrained_a2_rehearsal_s42", "pretrained", "1m_b", 0.5, 42),
    "1m_b_reh50_s4": (G + "offline_pretrained_a2_rehearsal_s4", "pretrained", "1m_b", 0.5, 4),
    "1m_b_noreh_s42": (G + "offline_pretrained_a2_no_rehearsal_s42", "pretrained", "1m_b", 0.0, 42),
    "1m_b_noreh_s4": (G + "offline_pretrained_a2_no_rehearsal_s4", "pretrained", "1m_b", 0.0, 4),
    "1m_b_ent_reh25_s42": (G + "offline_pretrained_a2_entropy_rehearsal25_s42", "pretrained", "1m_b", 0.25, 42),
    "1m_b_ent_reh25_s4": (G + "offline_pretrained_a2_entropy_rehearsal25_s4", "pretrained", "1m_b", 0.25, 4),
    "1m_b_bc_noreh_s42": (G + "offline_pretrained_a2_bc_no_rehearsal_s42", "pretrained", "1m_b", 0.0, 42),
    "1m_b_bc_noreh_s4": (G + "offline_pretrained_a2_bc_no_rehearsal_s4", "pretrained", "1m_b", 0.0, 4),
    "s40_reh50_s42": (G + "offline_pretrained_a2_scaled40_rehearsal_s42", "pretrained", "scaled40", 0.5, 42),
    "s40_reh50_s4": (G + "offline_pretrained_a2_scaled40_rehearsal_s4", "pretrained", "scaled40", 0.5, 4),
    "s40_noreh_s42": (G + "offline_pretrained_a2_scaled40_no_rehearsal_s42", "pretrained", "scaled40", 0.0, 42),
    "s40_noreh_s4": (G + "offline_pretrained_a2_scaled40_no_rehearsal_s4", "pretrained", "scaled40", 0.0, 4),
}
OUT = HERE.parent / "data"


def window(df, col, lo, hi):
    part = df[(df["loop"] >= lo) & (df["loop"] <= hi)]
    return part[col].mean() if col in part and part[col].notna().any() else np.nan


def first_loop(ret5, level):
    hit = ret5[ret5 >= level]
    return float(hit.index[0]) if len(hit) else np.nan


def summary(name, df, meta):
    path, group, pretrain, rehearsal, seed = meta
    df = df[df["loop"] <= 99].sort_values("loop")
    ret = df.set_index("loop")["rollout/average_return"].dropna()
    ret5 = ret.rolling(5).mean()
    row = dict(run = name, group = group, pretrain = pretrain, rehearsal = rehearsal, seed = seed, loops = int(ret.index.max()) + 1)
    row.update(
        ret_95_99 = ret.loc[95:99].mean(), best5 = ret5.max(), mean_0_99 = ret.mean(),
        loop_500 = first_loop(ret5, 500), loop_1000 = first_loop(ret5, 1000), loop_1500 = first_loop(ret5, 1500),
        ret_10_19 = ret.loc[10:19].mean(), ret_20_29 = ret.loc[20:29].mean(),
    )
    early, late = (5, 24), (80, 99)
    for key, col in (
        ("sat", "rollout/action_saturation_frac"), ("flip", "rollout/flipped_frac"), ("vx", "rollout/x_velocity"),
        ("vx_mae", "world_model/one_step_x_velocity_mae"), ("rew_mae", "world_model/reward_mae"),
        ("rew_corr", "world_model/reward_head_corr"), ("rank_spear", "world_model/ranking_spearman"),
        ("gap", "closed_loop/reward_gap_per_step"), ("ret_corr", "closed_loop/return_corr"),
        ("dflip", "closed_loop/dream_flipped_at_end_frac"), ("tflip", "closed_loop/true_flipped_at_end_frac"),
        ("flow", "world_model/flow_loss"), ("act_sq", "rollout/action_sq_sum_per_step"),
    ):
        row[f"{key}_early"] = window(df, col, *early)
        row[f"{key}_late"] = window(df, col, *late)
    return row


def main():
    frames, rows = [], []
    for name, meta in RUNS.items():
        df = load(meta[0])
        df.insert(0, "run", name)
        frames.append(df)
        rows.append(summary(name, df, meta))
    pd.concat(frames, ignore_index = True).to_csv(OUT / "loops.csv", index = False)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index = False)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(runs[["run", "loops", "ret_95_99", "best5", "mean_0_99", "loop_500", "loop_1000", "loop_1500", "ret_10_19", "flip_late"]].round(2).to_string(index = False))


if __name__ == "__main__":
    main()
