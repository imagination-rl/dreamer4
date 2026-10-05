"""Stage-1 results of the combination queue (entries 118-135) against their two arms and the base.

Loops 0-99 (all runs): the summary of read_run of the 10-1-26 extract.py (MAX_LOOPS = 100), the same numbers as for the
100-loop arm runs in ../data/run_summary.csv. Loops 0-174 (combinations only): the same summary on the full curve.
For each pair and measure: the pair mean, the arm means, and two-sided p values of the pair against each arm and against
the base (pooled t test: the seed variance pooled over the base, all arms and all pairs; and Welch).
Writes ../data/stage1_summary.csv, ../data/stage1_tests.csv, ../data/stage1_returns.csv.
"""

import collections
import csv
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
BASE_EXTRACT = ROOT.parent / "10-1-26-grouped-normalization" / "scripts" / "extract.py"
spec = importlib.util.spec_from_file_location("base_extract", BASE_EXTRACT)
base_extract = importlib.util.module_from_spec(spec)
sys.modules["base_extract"] = base_extract
spec.loader.exec_module(base_extract)

from train.ablations import COMBINATION_PAIRS, COMBINATION_PARTS, COMBINATION_SEEDS, combination_run_name  # noqa: E402

# (key, sign): sign +1 means larger is better.
MEASURES = [
    ("return_last5", 1), ("return_mean_all", 1), ("loop_at_1000", -1),
    ("noise_std_after_takeoff", -1), ("max_drawdown5", -1), ("loops_with_drop_over_200", -1),
]


def read_full(name):
    """The summary on loops 0-99 (read_run) and on the full curve, and the full return curve."""
    _, result = base_extract.read_run(name)
    if result is None:
        return name, None
    acc = base_extract.EventAccumulator(str(base_extract.LOG_ROOT / name), size_guidance = {"scalars": 0}).Reload()
    returns = base_extract.values(acc, "rollout/average_return")
    return name, (result[2], base_extract.summary(returns, []), returns)


def main():
    with open(ROOT / "data" / "run_summary.csv") as file:
        single = list(csv.DictReader(file))
    groups = collections.defaultdict(list)
    for row in single:
        groups[row["arm"]].append({key: float(row[key]) if row[key] not in ("", "None") else np.nan for key, _ in MEASURES})

    names = {combination_run_name(pair, seed): ("+".join(pair), seed) for pair in COMBINATION_PAIRS for seed in COMBINATION_SEEDS}
    with ProcessPoolExecutor(9) as pool:
        results = dict(pool.map(read_full, names))
    missing = [name for name, result in results.items() if result is None]
    if missing:
        raise RuntimeError(f"missing runs: {missing}")

    full = collections.defaultdict(list)
    with open(ROOT / "data" / "stage1_summary.csv", "w", newline = "") as file, open(ROOT / "data" / "stage1_returns.csv", "w", newline = "") as returns_file:
        writer = None
        returns_writer = csv.writer(returns_file)
        returns_writer.writerow(["run", "loop", "return"])
        for name, (pair, seed) in names.items():
            first100, whole, returns = results[name]
            groups[pair].append({key: float(first100[key]) if first100[key] is not None else np.nan for key, _ in MEASURES})
            full[pair].append(whole)
            row = {"run": name, "pair": pair, "seed": seed, **{f"{k}_0_99": first100[k] for k, _ in MEASURES},
                   **{f"{k}_0_174": whole[k] for k, _ in MEASURES}, "return_mean_100_174": float(returns[100:175].mean())}
            if writer is None:
                writer = csv.DictWriter(file, fieldnames = list(row))
                writer.writeheader()
            writer.writerow(row)
            returns_writer.writerows((name, loop, float(value)) for loop, value in enumerate(returns))

    tests = []
    for key, sign in MEASURES:
        values = {group: np.array([row[key] for row in rows]) for group, rows in groups.items()}
        # loop_at_1000 is missing for runs that never pass 1,000; those runs get 105 (as in the 10-2-26 figures).
        values = {group: np.where(np.isnan(v), 105., v) for group, v in values.items()}
        dof = sum(len(v) - 1 for v in values.values())
        pooled_var = sum(((v - v.mean()) ** 2).sum() for v in values.values()) / dof

        def compare(a, b):
            diff = a.mean() - b.mean()
            t = diff / np.sqrt(pooled_var * (1 / len(a) + 1 / len(b))) if pooled_var > 0 else 0.
            p_pooled = 2 * stats.t.sf(abs(t), dof)
            p_welch = stats.ttest_ind(a, b, equal_var = False).pvalue if np.ptp(np.r_[a, b]) > 0 else 1.
            return diff, p_pooled, p_welch

        print(f"\n== {key} (loops 0-99; {'higher' if sign > 0 else 'lower'} is better); pooled sd {np.sqrt(pooled_var):.1f}, dof {dof}")
        for pair in COMBINATION_PAIRS:
            name = "+".join(pair)
            combo = values[name]
            line = f"{name:18s} {combo.mean():7.1f} ± {combo.std(ddof = 1):5.1f} |"
            for other in ("base", *(COMBINATION_PARTS[part] for part in pair)):
                diff, p_pooled, p_welch = compare(combo, values[other])
                better = sign * diff > 0
                mark = "*" if better and p_pooled < 0.04 else ("-" if not better and p_pooled < 0.04 else " ")
                line += f" vs {other[:14]:14s} {values[other].mean():7.1f} {diff:+7.1f} p {p_pooled:.3f}/{p_welch:.3f}{mark} |"
                tests.append({"measure": key, "pair": name, "against": other, "pair_mean": combo.mean(), "pair_sd": combo.std(ddof = 1),
                              "against_mean": values[other].mean(), "diff": diff, "better": better, "p_pooled": p_pooled, "p_welch": p_welch})
            print(line)

    print("\n== seed spread of the return at loops 95-99 (Brown-Forsythe against each arm)")
    for pair in COMBINATION_PAIRS:
        name = "+".join(pair)
        combo = np.array([row["return_last5"] for row in groups[name]])
        parts = []
        for part in pair:
            arm = np.array([row["return_last5"] for row in groups[COMBINATION_PARTS[part]]])
            parts.append(f"{part} sd {arm.std(ddof = 1):4.0f} p {stats.levene(combo, arm, center = 'median').pvalue:.2f}")
        print(f"{name:18s} sd {combo.std(ddof = 1):4.0f} | " + " | ".join(parts))

    print("\n== loops 0-174 of the pairs (no reference runs of 175 loops)")
    for pair in COMBINATION_PAIRS:
        name = "+".join(pair)
        rows = full[name]
        def m(key):
            v = np.array([row[key] if row[key] is not None else np.nan for row in rows], dtype = float)
            return f"{np.nanmean(v):7.0f} ± {np.nanstd(v, ddof = 1):4.0f}"
        print(f"{name:18s} last5(170-174) {m('return_last5')} best5 {m('return_best5')} mean {m('return_mean_all')} "
              f"noise {m('noise_std_after_takeoff')} drawdown {m('max_drawdown5')} drops>200 {m('loops_with_drop_over_200')} "
              f"below_best {m('last5_below_best5')}")

    with open(ROOT / "data" / "stage1_tests.csv", "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = list(tests[0]))
        writer.writeheader()
        writer.writerows(tests)


if __name__ == "__main__":
    main()
