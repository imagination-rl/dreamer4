"""Loop metrics of the 18 stage-1 combination runs (entries 118-135), all 175 loops.

It uses read_run of report/10-1-26-grouped-normalization/scripts/extract.py with MAX_LOOPS = 175 (the same tags).
The arms and the base (100 loops) come from report/10-3-26-combinations/data/loop_metrics.csv.
Writes ../data/loop_metrics.csv (pairs, arms and base), ../data/episodes.csv and ../data/run_summary.csv (pairs, loops 0-174).
"""

import csv
import importlib.util
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
OUT = ROOT / "data"
sys.path.insert(0, str(REPO))
BASE_EXTRACT = REPO / "report" / "10-1-26-grouped-normalization" / "scripts" / "extract.py"
ARMS_DATA = REPO / "report" / "10-3-26-combinations" / "data"

spec = importlib.util.spec_from_file_location("base_extract", BASE_EXTRACT)
base_extract = importlib.util.module_from_spec(spec)
# The worker processes find the module by this name.
sys.modules["base_extract"] = base_extract
spec.loader.exec_module(base_extract)
# The worker processes are forked after this change, so they see it.
base_extract.MAX_LOOPS = 175

from train.ablations import COMBINATION_PAIRS, COMBINATION_SEEDS, combination_run_name  # noqa: E402

RUNS = {combination_run_name(pair, seed): ("+".join(pair), seed) for pair in COMBINATION_PAIRS for seed in COMBINATION_SEEDS}


def main():
    OUT.mkdir(parents = True, exist_ok = True)
    with ProcessPoolExecutor(9) as pool:
        results = dict(pool.map(base_extract.read_run, RUNS))

    summaries = []
    with open(OUT / "loop_metrics.csv", "w", newline = "") as metrics_file, open(OUT / "episodes.csv", "w", newline = "") as episodes_file:
        metrics, episodes = csv.writer(metrics_file), csv.writer(episodes_file)
        metrics.writerow(["run", "loop", "tag", "value"])
        episodes.writerow(["run", "loop", "mean", "std", "p10", "min"])
        for name, (pair, seed) in RUNS.items():
            if results[name] is None:
                raise RuntimeError(f"missing run {name}")
            rows, episode_rows, stats = results[name]
            metrics.writerows(rows)
            episodes.writerows(episode_rows)
            summaries.append({"run": name, "pair": pair, "seed": seed, **stats})
        # The arms and the base, loops 0-99.
        with open(ARMS_DATA / "loop_metrics.csv") as file:
            reader = csv.reader(file)
            next(reader)
            metrics.writerows(reader)

    keys = list(dict.fromkeys(key for row in summaries for key in row))
    with open(OUT / "run_summary.csv", "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = keys)
        writer.writeheader()
        writer.writerows(summaries)
    shutil.copy(ARMS_DATA / "run_summary.csv", OUT / "arm_summary.csv")

    for row in summaries:
        print(f"{row['pair']:18s} s{row['seed']:<2d} loops={row['loops']} last5={row['return_last5']:6.0f} best5={row['return_best5']:6.0f} "
              f"mean={row['return_mean_all']:6.0f} at0={row['loop_at_0']} at1000={row['loop_at_1000']} drawdown={row['max_drawdown5']:5.0f} "
              f"noise={row['noise_std_after_takeoff']:4.0f} drops>200={row['loops_with_drop_over_200']} worst={row['worst_single_loop_drop']:5.0f} "
              f"flip_max={row.get('flipped_frac_max', float('nan')):.2f} flip_last10={row.get('flipped_frac_last10', float('nan')):.2f} hours={row.get('hours', float('nan')):.2f}")


if __name__ == "__main__":
    main()
