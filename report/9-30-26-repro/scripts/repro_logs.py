"""Per-loop metrics of the local A2 reproductions (queue 9-29-26, entries 21/22), the H100 A2 runs, and the
earlier local A2 runs (runs/local_9-28-26/attnpool_on_*), from the TensorBoard logs.

Output: report/9-30-26-repro/data/repro_loops.csv (one row per run and loop)
Usage:  python report/9-30-26-repro/scripts/repro_logs.py
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

RUNS = {
    "repro_s42": "runs/gym_imagination/audit_a2_next_state_reward_target_local_s42",
    "repro_s4": "runs/gym_imagination/audit_a2_next_state_reward_target_local_s4",
    "h100_s42": "runs/gym_imagination/audit_a2_next_state_reward_target",
    "h100_s4": "runs/gym_imagination/audit_a2_next_state_reward_target2",
    "local928_s42": "runs/local_9-28-26/attnpool_on_s42",
    "local928_s4": "runs/local_9-28-26/attnpool_on_s4",
    "local928_s1": "runs/local_9-28-26/attnpool_on_s1",
}
OUT = HERE.parent / "data" / "repro_loops.csv"


def main():
    frames = []
    for name, path in RUNS.items():
        df = load(path)
        df.insert(0, "run", name)
        frames.append(df)
    out = pd.concat(frames, ignore_index = True)
    out.to_csv(OUT, index = False)
    print(out.groupby("run").loop.max())


if __name__ == "__main__":
    main()
