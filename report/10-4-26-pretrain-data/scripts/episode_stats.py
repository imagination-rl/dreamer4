"""Statistics of each real episode in the replays of candidate source runs for a new offline dataset.

For each episode: loop, return, velocity, action size, saturation, flips (the onset step and the flipped fraction),
and the moment sums of (mean |a| of the step, x velocity of the step). The moment sums give the exact pooled
correlation of action size and speed (the confound of report/9-30-26-pretrain-rootcause) for any set of episodes.
The script reads the replay files with np.load(mmap_mode="r"), so it does not create or change a file in the run folder
(ReplayBuffer.from_folder creates a zero latents file when a run has no latents.data.npy).

Output: report/10-4-26-pretrain-data/data/episodes.csv
Usage:  python report/10-4-26-pretrain-data/scripts/episode_stats.py [workers]
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE.parent / "data"
RUNS = REPO / "runs" / "gym_imagination"
CANDIDATES = [
    f"a2_grouped_combo_{pair}_s{seed}"
    for pair in ("symexp_warmup15", "cevalue_warmup15", "symlog_warmup15", "rw4_block2", "rw4_depth5", "rw4_fullres")
    for seed in (4, 42, 1)
]
VX, ANGLE = 8, 1
STEPS = 1000
EPISODES_PER_LOOP = 64
SATURATED = 0.95
CHUNK = 256


def wrapped(angle):
    return np.abs(np.angle(np.exp(1j * angle)))


def run_stats(run: str) -> pd.DataFrame:
    folder = RUNS / run / "replay_buffer"
    load = lambda name: np.load(folder / name, mmap_mode = "r")
    lens = load("episode_lens.data.meta.npy")
    from_world_model = load("is_from_world_model.data.meta.npy")
    num = int(np.load(folder / "num_episodes.state.npy"))
    if int(np.load(folder / "episode_index.state.npy")) != num:
        raise ValueError(f"{run}: the replay wrapped around; the loop of an episode is not its index // 64")
    state, actions, rewards = load("critic_state.data.npy"), load("actions_continuous.data.npy"), load("rewards.data.npy")

    rows = []
    for start in range(0, num, CHUNK):
        stop = min(start + CHUNK, num)
        obs = np.asarray(state[start:stop, :STEPS + 1])
        size = np.abs(2. * np.asarray(actions[start:stop, :STEPS]) - 1.)
        reward = np.asarray(rewards[start:stop, :STEPS], dtype = np.float64)
        vx = obs[:, :STEPS, VX].astype(np.float64)
        flipped = wrapped(obs[:, :, ANGLE]) > np.pi / 2
        onsets = ~flipped[:, :-1] & flipped[:, 1:]
        step_size = size.mean(-1).astype(np.float64)
        upright = ~flipped[:, :STEPS]
        for row in range(stop - start):
            episode = start + row
            onset = np.flatnonzero(onsets[row])
            x, y, keep = step_size[row], vx[row], upright[row]
            rows.append(dict(
                run = run, episode = episode, loop = episode // EPISODES_PER_LOOP,
                full = bool(lens[episode] == STEPS + 1 and not from_world_model[episode]),
                ret = float(reward[row].sum()),
                vx_mean = float(y.mean()), vx_p90 = float(np.percentile(y, 90)),
                abs_action = float(x.mean()), saturated = float((size[row] > SATURATED).mean()),
                flipped_frac = float(flipped[row, :STEPS].mean()),
                onsets = int(len(onset)), first_onset = int(onset[0]) if len(onset) else -1,
                start_flipped = bool(flipped[row, 0]),
                # Moment sums over all steps, and over upright steps.
                n = STEPS, sx = float(x.sum()), sy = float(y.sum()), sxx = float((x * x).sum()), syy = float((y * y).sum()),
                sxy = float((x * y).sum()),
                n_up = int(keep.sum()), sx_up = float(x[keep].sum()), sy_up = float(y[keep].sum()),
                sxx_up = float((x[keep] ** 2).sum()), syy_up = float((y[keep] ** 2).sum()), sxy_up = float((x[keep] * y[keep]).sum()),
                sat_up = float((size[row][keep] > SATURATED).mean()) if keep.any() else np.nan,
            ))
    print(run, num, flush = True)
    return pd.DataFrame(rows)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with ProcessPoolExecutor(workers) as pool:
        tables = list(pool.map(run_stats, CANDIDATES))
    OUT.mkdir(parents = True, exist_ok = True)
    pd.concat(tables).to_csv(OUT / "episodes.csv", index = False)


if __name__ == "__main__":
    main()
