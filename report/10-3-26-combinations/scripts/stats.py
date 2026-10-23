"""Tests of each one-change arm against the grouped-normalization base, from ../data/run_summary.csv.

For each measure and arm: the arm mean, the base mean, the difference, and two two-sided p values.
  welch   Welch t test (no equal-variance assumption)
  pooled  t test with the seed variance pooled over all arms (more degrees of freedom; the same method as the 10-2-26 report)
  holm    the pooled p value with the Holm correction over the arms of that measure
For seed variability: the Brown-Forsythe (median Levene) test of the arm seeds against the base seeds.
Writes ../data/tests.csv.
"""

import collections
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

# (key, label, sign): sign +1 means larger is better.
MEASURES = [
    ("return_last5", "return, loops 95-99", 1),
    ("return_mean_all", "mean return, loops 0-99", 1),
    ("noise_std_after_takeoff", "loop-to-loop noise after loop at 0", -1),
    ("max_drawdown5", "largest drop of the 5-loop mean", -1),
    ("loops_with_drop_over_200", "loops with a drop over 200", -1),
]


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def main():
    groups = collections.defaultdict(list)
    with open(ROOT / "data" / "run_summary.csv") as file:
        for row in csv.DictReader(file):
            groups[row["arm"]].append(row)
    arms = [arm for arm in groups if arm != "base"]

    out = []
    for key, label, sign in MEASURES:
        values = {arm: np.array([float(row[key]) for row in rows]) for arm, rows in groups.items()}
        base = values["base"]
        # The pooled variance of all arms (each arm about its own mean).
        dof = sum(len(v) - 1 for v in values.values())
        pooled_var = sum(((v - v.mean()) ** 2).sum() for v in values.values()) / dof
        pooled_p = []
        for arm in arms:
            arm_values = values[arm]
            diff = arm_values.mean() - base.mean()
            t = diff / np.sqrt(pooled_var * (1 / len(arm_values) + 1 / len(base)))
            pooled_p.append(2 * stats.t.sf(abs(t), dof))
        adjusted = holm(np.array(pooled_p))
        for arm, p, p_holm in zip(arms, pooled_p, adjusted):
            arm_values = values[arm]
            welch = stats.ttest_ind(arm_values, base, equal_var = False).pvalue if np.ptp(np.r_[arm_values, base]) > 0 else 1.
            out.append({
                "measure": key, "arm": arm, "n": len(arm_values), "arm_mean": arm_values.mean(), "arm_sd": arm_values.std(ddof = 1),
                "base_mean": base.mean(), "base_sd": base.std(ddof = 1), "diff": arm_values.mean() - base.mean(),
                "better": bool(sign * (arm_values.mean() - base.mean()) > 0),
                "p_welch": welch, "p_pooled": p, "p_holm": p_holm,
            })
        print(f"\n== {label} ({key}); base {base.mean():.0f} ± {base.std(ddof = 1):.0f} (n {len(base)}); pooled sd {np.sqrt(pooled_var):.0f}, dof {dof}")
        for row in sorted((r for r in out if r["measure"] == key), key = lambda r: r["p_pooled"]):
            mark = "*" if row["better"] and row["p_pooled"] < 0.04 else " "
            print(f"{mark} {row['arm']:28s} {row['arm_mean']:7.0f} ± {row['arm_sd']:5.0f}  diff {row['diff']:+7.0f}  "
                  f"welch {row['p_welch']:.4f}  pooled {row['p_pooled']:.4f}  holm {row['p_holm']:.4f}")

    # Seed variability: the spread of the final return over the seeds.
    print("\n== seed variability of return, loops 95-99 (Brown-Forsythe against the base)")
    base = np.array([float(row["return_last5"]) for row in groups["base"]])
    for arm in arms:
        arm_values = np.array([float(row["return_last5"]) for row in groups[arm]])
        p = stats.levene(arm_values, base, center = "median").pvalue
        lower = arm_values.std(ddof = 1) < base.std(ddof = 1)
        print(f"{'*' if lower and p < 0.04 else ' '} {arm:28s} sd {arm_values.std(ddof = 1):5.0f} (base {base.std(ddof = 1):.0f})  "
              f"cv {arm_values.std(ddof = 1) / arm_values.mean():.3f} (base {base.std(ddof = 1) / base.mean():.3f})  p {p:.3f}")
        out.append({"measure": "seed_sd_return_last5", "arm": arm, "n": len(arm_values), "arm_mean": arm_values.std(ddof = 1),
                    "base_mean": base.std(ddof = 1), "diff": arm_values.std(ddof = 1) - base.std(ddof = 1), "better": lower, "p_welch": p})

    keys = list(dict.fromkeys(key for row in out for key in row))
    with open(ROOT / "data" / "tests.csv", "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = keys)
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    main()
