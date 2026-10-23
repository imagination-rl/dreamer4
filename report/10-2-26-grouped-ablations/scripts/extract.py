"""Loop metrics of the 10-2-26 queue (entries 78-97) and their base runs (entries 73 and 77).

It uses read_run of report/10-1-26-grouped-normalization/scripts/extract.py (the same tags and the same summary).
Writes ../data/loop_metrics.csv, ../data/episodes.csv, ../data/run_summary.csv.
"""

import csv
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data"
BASE_EXTRACT = ROOT.parent / "10-1-26-grouped-normalization" / "scripts" / "extract.py"

spec = importlib.util.spec_from_file_location("base_extract", BASE_EXTRACT)
base_extract = importlib.util.module_from_spec(spec)
# The worker processes find the module by this name.
sys.modules["base_extract"] = base_extract
spec.loader.exec_module(base_extract)

ARMS = [
    "loss_normalization", "wm_pretrain640", "reward_weight4", "norm_warmup15", "pmpo", "hl_gauss_value_loss",
    "symlog_hl_gauss_value_head", "symexp_hl_gauss_actor_critic", "no_attn_pool", "full_attn_res", "block2_attn_res",
]
# run: (arm, seed). The base arm is a2_grouped_loss_normalization_s4 / s42.
RUNS = {f"a2_grouped_{arm}_s{seed}": (arm if arm != "loss_normalization" else "base", seed) for arm in ARMS for seed in (4, 42)}
# The plain A2 local runs (no normalization), for reference.
RUNS.update({f"audit_a2_next_state_reward_target_local_s{seed}": ("plain A2", seed) for seed in (4, 42)})


def main():
    OUT.mkdir(parents = True, exist_ok = True)
    with ProcessPoolExecutor(8) as pool:
        results = dict(pool.map(base_extract.read_run, RUNS))

    summaries = []
    with open(OUT / "loop_metrics.csv", "w", newline = "") as metrics_file, open(OUT / "episodes.csv", "w", newline = "") as episodes_file:
        metrics, episodes = csv.writer(metrics_file), csv.writer(episodes_file)
        metrics.writerow(["run", "loop", "tag", "value"])
        episodes.writerow(["run", "loop", "mean", "std", "p10", "min"])
        for name, (arm, seed) in RUNS.items():
            if results[name] is None:
                print("missing:", name, file = sys.stderr)
                continue
            rows, episode_rows, stats = results[name]
            metrics.writerows(rows)
            episodes.writerows(episode_rows)
            summaries.append({"run": name, "arm": arm, "seed": seed, **stats})

    keys = list(dict.fromkeys(key for row in summaries for key in row))
    with open(OUT / "run_summary.csv", "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = keys)
        writer.writeheader()
        writer.writerows(summaries)

    for row in summaries:
        print(
            f"{row['arm'][:28]:28s} s{row['seed']:<2d} loops={row['loops']:3d} last5={row['return_last5']:6.0f} best5={row['return_best5']:6.0f} "
            f"at0={row['loop_at_0']} at500={row['loop_at_500']} at1000={row['loop_at_1000']} drop={row['max_drawdown5']:5.0f} "
            f"noise={row['noise_std_after_takeoff']:4.0f} drops>200={row['loops_with_drop_over_200']} flip_max={row.get('flipped_frac_max', float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
