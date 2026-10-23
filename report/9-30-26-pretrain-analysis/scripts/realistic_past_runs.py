"""Dry run of the past_runs selection of OFFLINE_DATA_REALISTIC (train/ablations.py), without an env or a collection.

It calls the same function as the collect stage (offline_pretraining.past_run_episodes, same rng) and gives the
diversity of the selected episodes: the return, the speed, the action size, the flipped fraction, and the
upright-to-flipped transitions, for each return bucket and for each source group.

Output: report/9-30-26-pretrain-analysis/data/realistic_past_runs_{buckets,groups}.csv
Usage:  python -m report.9-30-26-pretrain-analysis.scripts.realistic_past_runs [flip_episode_fraction]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from train import offline_pretraining as off
from train.ablations import OFFLINE_DATA_REALISTIC, OFFLINE_DATA_REALISTIC_EPISODES

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data"
VX, ANGLE = off.HALF_CHEETAH_X_VELOCITY_INDEX, off.HALF_CHEETAH_ROOT_ANGLE_INDEX


def group_of(folder: str) -> str:
    if "local_9-2" in folder:
        return "online-only (local 9-28 / 9-29)"
    if "scaled40" in folder:
        return "pretrained scaled40 (47-50)"
    return "pretrained 1M (17-32)"


def stats(obs, actions, rewards):
    angle = np.abs(off._wrap_angle(obs[:, :, ANGLE]))
    flipped = angle > np.pi / 2
    returns = rewards.sum(-1)
    return dict(
        episodes = len(returns), return_mean = returns.mean(), return_p10 = np.percentile(returns, 10),
        return_p90 = np.percentile(returns, 90), speed_mean = obs[:, :-1, VX].mean(),
        speed_p90 = np.percentile(obs[:, :-1, VX], 90), abs_action = np.abs(actions).mean(),
        saturated = (np.abs(actions) > 0.95).mean(), flipped_step_frac = flipped.mean(),
        mostly_flipped_episodes = int((flipped.mean(1) > 0.5).sum()),
        episodes_with_flip = int((~flipped[:, :-1] & flipped[:, 1:]).any(1).sum()),
        upright_to_flipped = int((~flipped[:, :-1] & flipped[:, 1:]).sum()),
    )


def main():
    config = OFFLINE_DATA_REALISTIC
    folders = config["offline_past_run_replays"]
    count = OFFLINE_DATA_REALISTIC_EPISODES["past_runs"]
    # The collect stage uses default_rng([data_seed, 3]) for past_runs.
    arrays, source, summaries = off.past_run_episodes(
        folders, count, rng = np.random.default_rng([10042, 3]), max_episode_steps = 1000, obs_dim = 17,
        action_dim = 6, action_range = (-1., 1.), return_buckets = config["offline_past_run_return_buckets"],
        flip_episode_fraction = float(sys.argv[1]) if len(sys.argv) > 1 else config.get("offline_past_run_flip_episode_fraction", 0.),
    )
    returns = arrays["rewards"].sum(-1)
    buckets = config["offline_past_run_return_buckets"]
    rows = []
    for low, high in buckets:
        keep = (returns >= low) & (returns < high)
        rows.append(dict(bucket = f"[{low:g}, {high:g})", **stats(arrays["obs"][keep], arrays["actions"][keep], arrays["rewards"][keep])))
    rows.append(dict(bucket = "all", **stats(arrays["obs"], arrays["actions"], arrays["rewards"])))
    by_bucket = pd.DataFrame(rows)

    groups = np.array([group_of(str(folders[index])) for index in source[:, 0]])
    by_group = pd.DataFrame([
        dict(group = name, folders = sum(group_of(str(f)) == name for f in folders),
             **stats(arrays["obs"][groups == name], arrays["actions"][groups == name], arrays["rewards"][groups == name]))
        for name in dict.fromkeys(groups)
    ])
    per_folder = pd.DataFrame([dict(folder = s["folder"].split("/")[-2], episodes = s["episodes"], **s["episodes_per_bucket"]) for s in summaries])

    by_bucket.to_csv(OUT / "realistic_past_runs_buckets.csv", index = False)
    by_group.to_csv(OUT / "realistic_past_runs_groups.csv", index = False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(by_bucket.round(3).to_string(index = False))
    print()
    print(by_group.round(3).to_string(index = False))
    print()
    print(per_folder.to_string(index = False))


if __name__ == "__main__":
    main()
