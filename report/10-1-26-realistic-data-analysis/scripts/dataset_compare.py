"""The realistic-data dataset (entry 55) and the bang-bang dataset (entry 62) against the scaled40 dataset (entry 44), for each mode.

Speed: obs index 8. Flipped: |wrapped root angle| > pi / 2. Saturated: |a| > 0.95.
Output: report/10-1-26-realistic-data-analysis/data/dataset_compare.csv, dataset_speed_bins.csv
Usage:  python report/10-1-26-realistic-data-analysis/scripts/dataset_compare.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from train import offline_pretraining as off

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "runs" / "offline_data" / "HalfCheetah-v4"
DATASETS = {"scaled40": "halfcheetah_v4_mix_scaled40_5p5m", "realistic": "halfcheetah_v4_mix_realistic_7p5m",
            "bangbang": "halfcheetah_v4_mix_bangbang_scaled_7p5m"}
VX, ANGLE = off.HALF_CHEETAH_X_VELOCITY_INDEX, off.HALF_CHEETAH_ROOT_ANGLE_INDEX
SPEED_BINS = [-np.inf, 0., 1., 2., 3., 4., 6., np.inf]
OUT = HERE.parent / "data"


def stats(obs, actions, rewards):
    vx = obs[:, :-1, VX]
    size = np.abs(actions)
    flipped = np.abs(off._wrap_angle(obs[:, :-1, ANGLE])) > np.pi / 2
    up = ~flipped
    flips = ~flipped[:, :-1] & flipped[:, 1:]
    return dict(
        episodes = len(rewards), steps = rewards.size, return_mean = rewards.sum(-1).mean(),
        vx_mean = vx.mean(), vx_p50 = np.median(vx), vx_p90 = np.percentile(vx, 90), vx_p99 = np.percentile(vx, 99),
        frac_vx_gt2 = (vx > 2).mean(), frac_vx_gt4 = (vx > 4).mean(),
        abs_action = size.mean(), saturated = (size > 0.95).mean(),
        corr_abs_action_vx = np.corrcoef(size.mean(-1).ravel(), vx.ravel())[0, 1],
        flipped_frac = flipped.mean(), flip_transitions = int(flips.sum()),
        reward_upright = rewards[up].mean() if up.any() else np.nan,
        reward_flipped = rewards[flipped].mean() if flipped.any() else np.nan,
        vx_flipped = vx[flipped].mean() if flipped.any() else np.nan,
        upright_vx_gt1_steps = int((up & (vx > 1)).sum()), flipped_vx_gt1_steps = int((flipped & (vx > 1)).sum()),
    )


def main():
    rows, bins = [], []
    for label, name in DATASETS.items():
        ds = off.load_offline_dataset(DATA / name)
        modes = np.asarray(ds.modes) if hasattr(ds, "modes") else np.asarray(ds.mode)
        for mode in ["all", *dict.fromkeys(modes.tolist())]:
            keep = np.ones(len(modes), bool) if mode == "all" else modes == mode
            rows.append(dict(dataset = label, mode = mode, **stats(ds.obs[keep], ds.actions[keep], ds.rewards[keep])))
        vx = ds.obs[:, :-1, VX].ravel()
        size = np.abs(ds.actions).reshape(-1, ds.actions.shape[-1])
        flipped = (np.abs(off._wrap_angle(ds.obs[:, :-1, ANGLE])) > np.pi / 2).ravel()
        which = np.digitize(vx, SPEED_BINS[1:-1])
        for b in range(len(SPEED_BINS) - 1):
            k = which == b
            bins.append(dict(dataset = label, vx_bin = f"[{SPEED_BINS[b]:g}, {SPEED_BINS[b + 1]:g})", steps = int(k.sum()),
                             share = k.mean(), saturated = (size[k] > 0.95).mean() if k.any() else np.nan,
                             abs_action = size[k].mean() if k.any() else np.nan, flipped = flipped[k].mean() if k.any() else np.nan))
    out, by_bin = pd.DataFrame(rows), pd.DataFrame(bins)
    out.to_csv(OUT / "dataset_compare.csv", index = False)
    by_bin.to_csv(OUT / "dataset_speed_bins.csv", index = False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(out.round(3).T.to_string())
    print(by_bin.round(3).to_string(index = False))


if __name__ == "__main__":
    main()
