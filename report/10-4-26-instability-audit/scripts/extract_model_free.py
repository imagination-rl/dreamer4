"""Scalars of the two model-free CleanRL PPO runs (Beta policy, entropy weight 0), for comparison.

One model-free iteration is 64 envs x 1000 steps = 64,000 env steps, 10 epochs x 32 minibatches.
One loop of the world-model runs also collects 64,000 env steps, and its PPO update also has 10 epochs x 32 minibatches.
Writes ../data/model_free.csv with the iteration index in the column "loop".
"""

import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parents[1] / "runs" / "gym_imagination"
TAGS = ["rollout/average_return", "cleanrl_updated_ppo/entropy", "cleanrl_updated_ppo/approx_kl",
        "cleanrl_updated_ppo/clip_fraction", "cleanrl_updated_ppo/explained_variance", "charts/learning_rate"]


def main():
    rows = []
    for run in ("sm_cleanrl_updated_PPO", "sm_cleanrl_updated_PPO2"):
        events = EventAccumulator(str(RUNS / run), size_guidance = {"scalars": 0})
        events.Reload()
        for tag in TAGS:
            for event in events.Scalars(tag):
                rows.append({"run": run, "tag": tag, "loop": event.step // 64000 - 1, "env_step": event.step, "value": event.value})
    with open(ROOT / "data" / "model_free.csv", "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
