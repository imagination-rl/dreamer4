"""Window means of the main measures for each run and each pair, from ../data/loop_metrics.csv.

Windows: loops 0-49, 50-99, 100-174, 150-174. The arms and the base have loops 0-99 only.
Writes ../data/windows.csv (one row for each run, tag and window) and prints the pair means (mean of the seeds).
"""

import collections
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"0-49": (0, 50), "50-99": (50, 100), "100-174": (100, 175), "150-174": (150, 175)}
TAGS = [
    ("rollout/average_return", "return", "{:.0f}"),
    ("audit/rollout/x_velocity", "velocity m/s", "{:.2f}"),
    ("audit/rollout/flipped_frac", "flipped frac", "{:.3f}"),
    ("audit/rollout/action_saturation_frac", "action saturation", "{:.3f}"),
    ("world_model/reward_mae", "reward MAE", "{:.3f}"),
    ("extra/flow_loss_raw", "flow loss raw", "{:.4f}"),
    ("cleanrl_updated_ppo/explained_variance", "critic EV", "{:.3f}"),
    ("cleanrl_updated_ppo/entropy", "policy entropy", "{:.2f}"),
    ("cleanrl_updated_ppo/approx_kl", "approx KL", "{:.3f}"),
    ("cleanrl_updated_ppo/clip_fraction", "clip fraction", "{:.3f}"),
    ("imagination/dream_return", "dream return", "{:.0f}"),
    ("audit/closed_loop/return_corr", "dream/true return corr", "{:.2f}"),
    ("audit/closed_loop/reward_gap_per_step", "dream - true reward/step", "{:+.3f}"),
    ("audit/closed_loop/dream_flipped_at_end_frac", "dream flipped at end", "{:.3f}"),
    ("audit/closed_loop/true_flipped_at_end_frac", "true flipped at end", "{:.3f}"),
    ("audit/world_model/ranking_spearman", "action ranking rho", "{:.2f}"),
    ("audit/world_model/one_step_x_velocity_mae", "1-step vx MAE", "{:.3f}"),
    ("audit/closed_loop/state_norm_mae_h32", "state MAE h32", "{:.3f}"),
]
PAIRS = ["cevalue+warmup15", "symlog+warmup15", "symexp+warmup15", "rw4+block2", "rw4+depth5", "rw4+fullres"]
ARMS = {
    "base": "a2_grouped_loss_normalization", "cevalue": "a2_grouped_hl_gauss_value_loss", "symlog": "a2_grouped_symlog_hl_gauss_value_head",
    "symexp": "a2_grouped_symexp_hl_gauss_actor_critic", "warmup15": "a2_grouped_norm_warmup15", "rw4": "a2_grouped_reward_weight4",
    "block2": "a2_grouped_block2_attn_res", "fullres": "a2_grouped_full_attn_res", "depth5": "a2_grouped_no_attn_pool_depth5",
}


def group_of(run):
    if run.startswith("a2_grouped_combo_"):
        return "+".join(run[len("a2_grouped_combo_"):].rsplit("_s", 1)[0].split("_"))
    for name, prefix in ARMS.items():
        if run.rsplit("_s", 1)[0] == prefix:
            return name
    return None


def main():
    data = collections.defaultdict(dict)
    with open(ROOT / "data" / "loop_metrics.csv") as file:
        for row in csv.DictReader(file):
            data[(row["run"], row["tag"])][int(row["loop"])] = float(row["value"])

    out = []
    pair_values = collections.defaultdict(list)
    for (run, tag), points in data.items():
        group = group_of(run)
        if group is None:
            continue
        for window, (lo, hi) in WINDOWS.items():
            values = [value for loop, value in points.items() if lo <= loop < hi]
            if values:
                out.append({"run": run, "group": group, "tag": tag, "window": window, "mean": float(np.mean(values))})
                pair_values[(group, tag, window)].append(float(np.mean(values)))

    with open(ROOT / "data" / "windows.csv", "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    groups = PAIRS + list(ARMS)
    for tag, label, fmt in TAGS:
        print(f"\n== {label} ({tag})")
        print(f"{'':18s}" + "".join(f"{w:>18s}" for w in WINDOWS))
        for group in groups:
            cells = []
            for window in WINDOWS:
                values = pair_values.get((group, tag, window))
                cells.append(f"{fmt.format(np.mean(values))} ({len(values)})" if values else "-")
            print(f"{group:18s}" + "".join(f"{cell:>18s}" for cell in cells))


if __name__ == "__main__":
    main()
