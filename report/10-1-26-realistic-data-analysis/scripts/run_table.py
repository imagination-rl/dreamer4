"""Run table of the realistic-data runs (entries 58-61) and the earlier runs, from TensorBoard.

It uses the summary of report/9-30-26-pretrain-analysis/scripts/run_table.py.
Output: report/10-1-26-realistic-data-analysis/data/{loops.csv, runs.csv}
Usage:  python report/10-1-26-realistic-data-analysis/scripts/run_table.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("prev", REPO / "report/9-30-26-pretrain-analysis/scripts/run_table.py")
prev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prev)

G = "runs/gym_imagination/"
RUNS = dict(prev.RUNS)
RUNS.update({
    "real_noreh_s42": (G + "pretrain_rl_realistic_data_s42", "pretrained", "realistic", 0.0, 42),
    "real_noreh_s4": (G + "pretrain_rl_realistic_data_s4", "pretrained", "realistic", 0.0, 4),
    "real_anchor_s42": (G + "pretrain_rl_anchor_realistic_data_s42", "pretrained", "realistic_anchor", 0.0, 42),
    "real_anchor_s4": (G + "pretrain_rl_anchor_realistic_data_s4", "pretrained", "realistic_anchor", 0.0, 4),
})
OUT = HERE.parent / "data"


def main():
    frames, rows = [], []
    for name, meta in RUNS.items():
        df = prev.load(meta[0])
        df.insert(0, "run", name)
        frames.append(df)
        rows.append(prev.summary(name, df, meta))
    pd.concat(frames, ignore_index = True).to_csv(OUT / "loops.csv", index = False)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index = False)
    pd.set_option("display.width", 250, "display.max_columns", 60)
    print(runs[["run", "loops", "ret_95_99", "best5", "mean_0_99", "loop_500", "loop_1000", "ret_10_19", "ret_20_29", "flip_early", "flip_late", "vx_mae_early", "vx_mae_late", "ret_corr_early", "rank_spear_early", "sat_early"]].round(2).to_string(index = False))


if __name__ == "__main__":
    main()
