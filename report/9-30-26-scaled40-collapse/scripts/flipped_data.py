"""Flipped states in the offline datasets: how many, from which mode, and what reward they get.

Flipped: |wrapped root angle (obs index 1)| > pi / 2. A flip transition: upright at t, flipped at t + 1.
Output: report/9-30-26-scaled40-collapse/data/flipped_data.csv
Usage:  python report/9-30-26-scaled40-collapse/scripts/flipped_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from train import offline_pretraining as off
from train.audit_ablations import HALF_CHEETAH_ROOT_ANGLE_INDEX as ANGLE, _wrap_angle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "runs" / "offline_data"
DATASETS = ["HalfCheetah-v4/halfcheetah_v4_mix_1m", "archive/HalfCheetah-v4/halfcheetah_v4_mix_past_runs_scaled_expert_5p5m",
            "HalfCheetah-v4/halfcheetah_v4_mix_scaled40_5p5m"]
OUT = HERE.parent / "data" / "flipped_data.csv"


def main():
    rows = []
    for name in DATASETS:
        d = off.load_offline_dataset(DATA / name)
        flipped = np.abs(_wrap_angle(d.obs[:, :, ANGLE])) > np.pi / 2          # (E, T + 1)
        state_flipped = flipped[:, :-1]                                        # state of transition t
        flips = (~flipped[:, :-1]) & flipped[:, 1:]
        total_steps = d.rewards.size
        for mode, mode_id in off.OFFLINE_MODE_IDS.items():
            m = d.mode == mode_id
            if not m.any():
                continue
            f = state_flipped[m]
            r = d.rewards[m]
            rows.append(dict(
                dataset = name.split("/")[-1], mode = mode, episodes = int(m.sum()),
                flipped_step_frac = float(f.mean()),
                flipped_steps_share_of_dataset = float(f.sum() / total_steps),
                episodes_mostly_flipped = int((f.mean(-1) > 0.5).sum()),
                flip_transitions = int(flips[m].sum()),
                reward_per_step_flipped = float(r[f].mean()) if f.any() else np.nan,
                reward_per_step_upright = float(r[~f].mean()),
                flipped_steps_reward_above_0_frac = float((r[f] > 0).mean()) if f.any() else np.nan,
            ))
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents = True, exist_ok = True)
    table.to_csv(OUT, index = False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index = False))


if __name__ == "__main__":
    main()
