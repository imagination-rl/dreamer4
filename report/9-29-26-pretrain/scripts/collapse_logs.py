"""Per-loop metrics of the offline-pretrained RL runs and the A2 baselines, from the TensorBoard logs.

Output: report/9-29-26-pretrain/data/collapse_loops.csv (one row per run and loop)
Usage:  python collapse_logs.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "reports" / "9-29-26" / "scripts"))
from windows import value_of  # noqa: E402

RUNS = {
    "pre_norh_s42": "runs/gym_imagination/offline_pretrained_a2_no_rehearsal",
    "pre_norh_s4": "runs/gym_imagination/offline_pretrained_a2_no_rehearsal2",
    "pre_rh50_s42": "runs/gym_imagination/offline_pretrained_a2_rehearsal",
    "pre_rh50_s4": "runs/gym_imagination/offline_pretrained_a2_rehearsal2",
    "local_a2_s42": "runs/local_9-28-26/attnpool_on_s42",
    "local_a2_s4": "runs/local_9-28-26/attnpool_on_s4",
    "local_a2_s1": "runs/local_9-28-26/attnpool_on_s1",
    "h100_a2_s42": "runs/gym_imagination/audit_a2_next_state_reward_target",
    "h100_a2_s4": "runs/gym_imagination/audit_a2_next_state_reward_target2",
}
PREFIXES = ("audit/rollout/", "audit/env_step/", "audit/closed_loop/", "audit/world_model/", "audit/progress/loop")
OUT = HERE.parent / "data" / "collapse_loops.csv"


def load(path: str) -> pd.DataFrame:
    rows = {}
    for f in glob.glob(str(REPO / path / "events.out.tfevents.*")):
        for ev in EventFileLoader(f).Load():
            if ev.HasField("summary"):
                for v in ev.summary.value:
                    if v.tag.startswith(PREFIXES) and (x := value_of(v)) is not None:
                        rows.setdefault(ev.step, {})[v.tag.replace("audit/", "").replace("env_step/", "")] = x
    df = pd.DataFrame.from_dict(rows, orient = "index").sort_index()
    # The rollout metrics of loop k are logged at env step 64000 (k + 1); the others at the same step.
    df["env_step"] = df.index
    df["loop"] = (df.index // 64000) - 1
    return df.reset_index(drop = True)


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
