"""Per-loop metrics of the scaled40 RL runs (entries 47-50) against the earlier pretrained arms and the runs without
pretraining, from TensorBoard.

Output: report/9-30-26-scaled40-collapse/data/loops.csv
Usage:  python report/9-30-26-scaled40-collapse/scripts/collapse_logs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "report" / "9-29-26-pretrain" / "scripts"))
from collapse_logs import load  # noqa: E402

G = "runs/gym_imagination/"
RUNS = {
    "scaled40_reh50_s42": G + "offline_pretrained_a2_scaled40_rehearsal_s42",
    "scaled40_reh50_s4": G + "offline_pretrained_a2_scaled40_rehearsal_s4",
    "scaled40_noreh_s42": G + "offline_pretrained_a2_scaled40_no_rehearsal_s42",
    "scaled40_noreh_s4": G + "offline_pretrained_a2_scaled40_no_rehearsal_s4",
    "old_reh50_s42": G + "offline_pretrained_a2_rehearsal_s42",
    "old_reh50_s4": G + "offline_pretrained_a2_rehearsal_s4",
    "old_noreh_s42": G + "offline_pretrained_a2_no_rehearsal_s42",
    "old_noreh_s4": G + "offline_pretrained_a2_no_rehearsal_s4",
    "repro_s42": G + "audit_a2_next_state_reward_target_local_s42",
    "repro_s4": G + "audit_a2_next_state_reward_target_local_s4",
    "h100_s42": G + "audit_a2_next_state_reward_target",
    "h100_s4": G + "audit_a2_next_state_reward_target2",
}
COLUMNS = [
    "rollout/average_return", "rollout/x_velocity", "rollout/action_saturation_frac", "rollout/flipped_frac",
    "imagination/dream_return", "extra/policy_entropy", "cleanrl_updated_ppo/approx_kl",
    "world_model/reward_mae", "world_model/flow_loss", "world_model/one_step_x_velocity_mae",
    "closed_loop/dream_reward_per_step", "closed_loop/reward_gap_per_step", "closed_loop/return_corr",
    "closed_loop/true_flipped_at_end_frac", "closed_loop/dream_flipped_at_end_frac",
]
OUT = HERE.parent / "data" / "loops.csv"


def main():
    frames = []
    for name, path in RUNS.items():
        df = load(path)
        df.insert(0, "run", name)
        frames.append(df)
    out = pd.concat(frames, ignore_index = True)
    out = out[out["progress/loop"].notna()] if "progress/loop" in out else out
    OUT.parent.mkdir(parents = True, exist_ok = True)
    out.to_csv(OUT, index = False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    short = {c: c.split("/")[-1] for c in COLUMNS}
    rows = []
    for name, df in out.groupby("run", sort = False):
        for lo, hi in ((10, 20), (25, 35), (40, 50), (60, 70), (80, 90), (90, 100)):
            part = df[(df["loop"] >= lo) & (df["loop"] < hi)]
            if part.empty:
                continue
            rows.append({"run": name, "loops": f"{lo}-{hi - 1}", **{short[c]: part[c].mean() for c in COLUMNS if c in part}})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index = False))


if __name__ == "__main__":
    main()
