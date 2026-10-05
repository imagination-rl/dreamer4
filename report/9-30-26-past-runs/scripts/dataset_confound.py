"""Action size against speed in each offline dataset (the confound of report/9-30-26-pretrain-rootcause).

For each dataset and mode: the fraction of action dimensions at the bound (|a| > 0.95), the mean |a|, and the forward
velocity (obs index 8). A dataset without the confound has moderate actions at all speeds, as the online policy does.
The last rows bin all steps of a dataset by velocity and give the saturation of each bin.

Output: report/9-30-26-past-runs/data/dataset_confound.csv, dataset_confound_by_velocity.csv
Usage:  python report/9-30-26-past-runs/scripts/dataset_confound.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from train import offline_pretraining as off

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "runs" / "offline_data"
DATASETS = [
    "HalfCheetah-v4/halfcheetah_v4_mix_1m",
    "archive/HalfCheetah-v4/halfcheetah_v4_mix_past_runs_1m",
    "archive/HalfCheetah-v4/halfcheetah_v4_mix_past_runs_scaled_expert_5p5m",
    "HalfCheetah-v4/halfcheetah_v4_mix_scaled40_5p5m",
]
VX, SATURATED = 8, 0.95
VELOCITY_BINS = [-np.inf, 0., 2., 4., 6., 8., 10., np.inf]
OUT = HERE.parent / "data"


def main():
    rows, binned = [], []
    for name in DATASETS:
        dataset = off.load_offline_dataset(DATA / name)
        # Transition t: obs_t and a_t. The velocity is that of obs_t.
        velocity = dataset.obs[:, :-1, VX]
        size = np.abs(dataset.actions)
        groups = {mode: dataset.mode == mode_id for mode, mode_id in off.OFFLINE_MODE_IDS.items()}
        groups["all"] = np.ones(dataset.num_episodes, dtype = bool)

        for mode, mask in groups.items():
            if not mask.any():
                continue
            rows.append(dict(
                dataset = name.split('/')[-1], mode = mode, episodes = int(mask.sum()),
                saturated_frac = float((size[mask] > SATURATED).mean()),
                mean_abs_action = float(size[mask].mean()),
                velocity_mean = float(velocity[mask].mean()),
                velocity_p10 = float(np.percentile(velocity[mask], 10)),
                velocity_p90 = float(np.percentile(velocity[mask], 90)),
                return_mean = float(dataset.rewards[mask].sum(-1).mean()),
            ))

        steps_velocity = velocity.reshape(-1)
        steps_saturated = (size > SATURATED).mean(-1).reshape(-1)
        steps_size = size.mean(-1).reshape(-1)
        bins = pd.cut(steps_velocity, VELOCITY_BINS)
        for interval, group in pd.DataFrame(dict(bin = bins, sat = steps_saturated, size = steps_size)).groupby("bin", observed = True):
            binned.append(dict(dataset = name.split('/')[-1], velocity_bin = str(interval), steps = len(group),
                               step_frac = len(group) / len(steps_velocity),
                               saturated_frac = float(group.sat.mean()), mean_abs_action = float(group["size"].mean())))

        # Correlation of action size with speed over all steps: the confound in one number.
        rows[-1]["corr_abs_action_velocity"] = float(np.corrcoef(steps_size, steps_velocity)[0, 1])

    OUT.mkdir(parents = True, exist_ok = True)
    table, by_velocity = pd.DataFrame(rows), pd.DataFrame(binned)
    table.to_csv(OUT / "dataset_confound.csv", index = False)
    by_velocity.to_csv(OUT / "dataset_confound_by_velocity.csv", index = False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index = False))
    print()
    print(by_velocity.round(3).to_string(index = False))


if __name__ == "__main__":
    main()
