"""Compare the offline datasets on the measures of the earlier root causes.

- corr_abs_action_velocity: the pooled correlation of the mean |a| of a step and the x velocity
  (report/9-30-26-pretrain-rootcause; scaled40 0.661, realistic 0.55, bang-bang data 0.12).
- saturated fraction (|a| > 0.95) for each speed bin: fast data with saturated actions only is the confound.
- vx mean and std: the frozen pretrained tokenizer normalizes the velocity with them (online-only tokenizer: std 0.74).
- flipped steps, flip onsets (upright-to-flipped transitions), and recoveries (flipped-to-upright).
The arrays are read with mmap in chunks of episodes, so one dataset never needs to fit in memory.

Output: report/10-4-26-pretrain-data/data/dataset_compare.csv, dataset_speed_bins.csv
Usage:  python report/10-4-26-pretrain-data/scripts/dataset_compare.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "data"
DATA = REPO / "runs" / "offline_data" / "HalfCheetah-v4"
DATASETS = [
    "halfcheetah_v4_mix_1m", "halfcheetah_v4_mix_scaled40_5p5m", "halfcheetah_v4_mix_realistic_7p5m",
    "halfcheetah_v4_mix_bangbang_scaled_7p5m", "halfcheetah_v4_own_runs_6m",
]
VX, ANGLE, SATURATED, CHUNK = 8, 1, 0.95, 250
BINS = [-np.inf, 0., 1., 2., 3., 4., 5., 6., 8., np.inf]


def main():
    rows, binned = [], []
    for name in DATASETS:
        folder = DATA / name
        obs = np.load(folder / "obs.npy", mmap_mode = "r")
        actions = np.load(folder / "actions.npy", mmap_mode = "r")
        lens = np.load(folder / "lens.npy")
        sums = np.zeros(6)
        n = flipped_steps = onsets = recoveries = onset_episodes = 0
        vx_all = []
        bin_steps, bin_sat, bin_size = (np.zeros(len(BINS) - 1) for _ in range(3))
        for start in range(0, len(lens), CHUNK):
            stop = min(start + CHUNK, len(lens))
            o = np.asarray(obs[start:stop])
            a = np.abs(np.asarray(actions[start:stop]))
            steps = o.shape[1] - 1
            valid = np.arange(steps)[None] < lens[start:stop, None]
            x = a.mean(-1)[valid].astype(np.float64)
            y = o[:, :steps, VX][valid].astype(np.float64)
            sat = (a > SATURATED).mean(-1)[valid]
            sums += [x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum(), 0.]
            n += len(x)
            vx_all.append(y[::10])
            index = np.digitize(y, BINS[1:-1])
            bin_steps += np.bincount(index, minlength = len(BINS) - 1)
            bin_sat += np.bincount(index, weights = sat, minlength = len(BINS) - 1)
            bin_size += np.bincount(index, weights = x, minlength = len(BINS) - 1)
            flip = np.abs(np.angle(np.exp(1j * o[:, :, ANGLE].astype(np.float64)))) > np.pi / 2
            valid_rows = np.arange(steps + 1)[None] <= lens[start:stop, None]
            flipped_steps += int((flip & valid_rows).sum())
            pair = valid_rows[:, 1:]
            up_to_flip = ~flip[:, :-1] & flip[:, 1:] & pair
            onsets += int(up_to_flip.sum())
            onset_episodes += int(up_to_flip.any(1).sum())
            recoveries += int((flip[:, :-1] & ~flip[:, 1:] & pair).sum())
        sx, sy, sxx, syy, sxy, _ = sums / n
        vx = np.concatenate(vx_all)
        rows.append(dict(
            dataset = name, episodes = len(lens), steps = n,
            corr_abs_action_velocity = (sxy - sx * sy) / np.sqrt((sxx - sx ** 2) * (syy - sy ** 2)),
            vx_mean = float(vx.mean()), vx_std = float(vx.std()), vx_p95 = float(np.percentile(vx, 95)),
            saturated = float(bin_sat.sum() / bin_steps.sum()), mean_abs_action = sx,
            flipped_step_frac = flipped_steps / (n + len(lens)), onsets = onsets, onset_episodes = onset_episodes,
            recoveries = recoveries, onsets_per_100k_flipped_steps = 1e5 * onsets / max(flipped_steps, 1),
        ))
        for index in range(len(BINS) - 1):
            if bin_steps[index] > 0:
                binned.append(dict(
                    dataset = name, velocity_bin = f"[{BINS[index]:g}, {BINS[index + 1]:g})",
                    step_frac = bin_steps[index] / n, saturated = bin_sat[index] / bin_steps[index],
                    mean_abs_action = bin_size[index] / bin_steps[index],
                ))
        print(name, flush = True)

    OUT.mkdir(parents = True, exist_ok = True)
    table, by_speed = pd.DataFrame(rows), pd.DataFrame(binned)
    table.to_csv(OUT / "dataset_compare.csv", index = False)
    by_speed.to_csv(OUT / "dataset_speed_bins.csv", index = False)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index = False))
    print(by_speed.pivot(index = "velocity_bin", columns = "dataset", values = "saturated").round(3).to_string())
    print(by_speed.pivot(index = "velocity_bin", columns = "dataset", values = "step_frac").round(3).to_string())


if __name__ == "__main__":
    main()
