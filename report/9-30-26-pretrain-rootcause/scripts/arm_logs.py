"""Per-loop metrics of all offline-pretrained RL arms and their no-pretraining baselines, from TensorBoard.

Output: report/9-30-26-pretrain-rootcause/data/arm_loops.csv (one row per run and loop)
Usage:  python report/9-30-26-pretrain-rootcause/scripts/arm_logs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "report" / "9-29-26-pretrain" / "scripts"))
sys.path.insert(0, str(REPO / "reports" / "9-29-26" / "scripts"))
from collapse_logs import load  # noqa: E402

G = "runs/gym_imagination/"
# name: (arm, seed, path)
RUNS = {
    "repro_s42": ("repro (no pretrain)", 42, G + "audit_a2_next_state_reward_target_local_s42"),
    "repro_s4": ("repro (no pretrain)", 4, G + "audit_a2_next_state_reward_target_local_s4"),
    "h100_s42": ("H100 (no pretrain)", 42, G + "audit_a2_next_state_reward_target"),
    "h100_s4": ("H100 (no pretrain)", 4, G + "audit_a2_next_state_reward_target2"),
    "reh50_s42": ("rehearsal 0.5", 42, G + "offline_pretrained_a2_rehearsal_s42"),
    "reh50_s4": ("rehearsal 0.5", 4, G + "offline_pretrained_a2_rehearsal_s4"),
    "noreh_s42": ("no rehearsal", 42, G + "offline_pretrained_a2_no_rehearsal_s42"),
    "noreh_s4": ("no rehearsal", 4, G + "offline_pretrained_a2_no_rehearsal_s4"),
    "ent_s42": ("entropy + rehearsal 0.25", 42, G + "offline_pretrained_a2_entropy_rehearsal25_s42"),
    "ent_s4": ("entropy + rehearsal 0.25", 4, G + "offline_pretrained_a2_entropy_rehearsal25_s4"),
    "bc_s42": ("BC, no rehearsal", 42, G + "offline_pretrained_a2_bc_no_rehearsal_s42"),
    "bc_s4": ("BC, no rehearsal", 4, G + "offline_pretrained_a2_bc_no_rehearsal_s4"),
    # Entries 17-20 (shared pretrain a2_next_state_reward, step 12000).
    "old_reh50_s42": ("old rehearsal 0.5", 42, G + "offline_pretrained_a2_rehearsal"),
    "old_reh50_s4": ("old rehearsal 0.5", 4, G + "offline_pretrained_a2_rehearsal2"),
    "old_noreh_s42": ("old no rehearsal", 42, G + "offline_pretrained_a2_no_rehearsal"),
    "old_noreh_s4": ("old no rehearsal", 4, G + "offline_pretrained_a2_no_rehearsal2"),
}
OUT = HERE.parent / "data" / "arm_loops.csv"


def main():
    frames = []
    for name, (arm, seed, path) in RUNS.items():
        if not (REPO / path).exists():
            print("missing", path)
            continue
        df = load(path)
        if df.empty:
            print("empty", path)
            continue
        df.insert(0, "seed", seed)
        df.insert(0, "arm", arm)
        df.insert(0, "run", name)
        frames.append(df)
    out = pd.concat(frames, ignore_index = True)
    out.to_csv(OUT, index = False)
    print(out.groupby("run").loop.max())


if __name__ == "__main__":
    main()
