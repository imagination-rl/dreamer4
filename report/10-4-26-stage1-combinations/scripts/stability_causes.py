"""Is the difference in stability between runs a property of the settings, of the seed, or of chance?

Runs: the base (4 seeds), the 13 arms (3 seeds; PMPO 2) and the 6 pairs (3 seeds), from ../data/loop_metrics.csv.
Window A: loops 50-99 (all runs; every run is past its start by loop 50 except some PMPO seeds).
Window B: loops 100-174 (the 18 pair runs only).

Stability measures of each run in a window:
  rel_noise    std of the return about a 9-loop centered mean / mean return of the window
  flip_mean    mean flipped fraction of the window
  flip_loops   loops with a flipped fraction above 0.2
  drop_loops   loops whose return is more than 15 % below the 5-loop mean that ends at the loop before
Covariates (how aggressive the policy is): mean velocity, mean saturated-action fraction, mean policy entropy.

Tests:
  1. Settings: one-way ANOVA over the groups (and Kruskal-Wallis), and the intraclass correlation ICC(1):
     the share of the variance between runs that the settings explain.
  2. Seed: two-way additive model group + seed (seeds 4, 42, 1 of the groups with all three); F test of the seed term.
  3. Speed: the variance that the covariates explain (R^2), and the F test of the group term after the covariates.
Writes ../data/stability_runs.csv.
"""

import collections
import csv
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TAGS = {
    "rollout/average_return": "ret", "audit/rollout/flipped_frac": "flip", "audit/rollout/x_velocity": "vel",
    "audit/rollout/action_saturation_frac": "sat", "cleanrl_updated_ppo/entropy": "ent",
}
SEEDS = (4, 42, 1)


def group_of(run):
    stem, seed = run.rsplit("_s", 1)
    name = stem[len("a2_grouped_"):]
    if name.startswith("combo_"):
        name = "+".join(name[len("combo_"):].split("_"))
    return ("base" if name == "loss_normalization" else name), int(seed)


def run_measures(series, lo, hi):
    ret = np.array([series["ret"][i] for i in range(lo, hi)])
    flip = np.array([series["flip"][i] for i in range(lo, hi)])
    trend = np.convolve(ret, np.ones(9) / 9, mode = "valid")
    residual = ret[4:-4] - trend
    full = np.array([series["ret"][i] for i in range(0, hi)])
    smooth = np.convolve(full, np.ones(5) / 5, mode = "valid")  # smooth[j] = mean of loops j..j+4
    drops = sum(1 for i in range(lo, hi) if i >= 5 and full[i] < 0.85 * smooth[i - 5])
    return {
        "mean_return": float(ret.mean()), "rel_noise": float(residual.std() / max(ret.mean(), 1.)),
        "flip_mean": float(flip.mean()), "flip_loops": int((flip > 0.2).sum()), "drop_loops": drops,
        "vel": float(np.mean([series["vel"][i] for i in range(lo, hi)])),
        "sat": float(np.mean([series["sat"][i] for i in range(lo, hi)])),
        "ent": float(np.mean([series["ent"][i] for i in range(lo, hi)])),
    }


def ols_rss(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond = None)
    residual = y - X @ beta
    return float(residual @ residual), np.linalg.matrix_rank(X)


def f_test(X_small, X_big, y):
    rss_small, rank_small = ols_rss(X_small, y)
    rss_big, rank_big = ols_rss(X_big, y)
    df1, df2 = rank_big - rank_small, len(y) - rank_big
    if df1 <= 0 or df2 <= 0 or rss_big <= 0:
        return float("nan"), float("nan")
    F = ((rss_small - rss_big) / df1) / (rss_big / df2)
    return F, float(stats.f.sf(F, df1, df2))


def dummies(labels):
    levels = sorted(set(labels))
    return np.array([[1. if label == level else 0. for level in levels[1:]] for label in labels]).reshape(len(labels), -1)


def icc1(values_by_group):
    groups = [np.asarray(v) for v in values_by_group if len(v) >= 2]
    k = len(groups)
    n = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def analyse(rows, title, measures):
    print(f"\n######## {title}: {len(rows)} runs, {len({r['group'] for r in rows})} groups")
    groups = sorted({r["group"] for r in rows})
    ones = np.ones((len(rows), 1))
    G = dummies([r["group"] for r in rows])
    cov = np.array([[r["vel"], r["sat"], r["ent"]] for r in rows])
    cov = (cov - cov.mean(0)) / cov.std(0)
    seeded = [r for r in rows if r["seed"] in SEEDS and sum(1 for q in rows if q["group"] == r["group"] and q["seed"] in SEEDS) == 3]
    for key in measures:
        y = np.array([r[key] for r in rows], dtype = float)
        by_group = [[r[key] for r in rows if r["group"] == g] for g in groups]
        F_group, p_group = f_test(ones, np.hstack([ones, G]), y)
        p_kw = stats.kruskal(*[g for g in by_group if len(g) > 1]).pvalue if np.ptp(y) > 0 else 1.
        icc = icc1(by_group)
        # speed covariates
        rss0, _ = ols_rss(ones, y)
        rss_cov, _ = ols_rss(np.hstack([ones, cov]), y)
        r2_cov = 1 - rss_cov / rss0 if rss0 > 0 else float("nan")
        F_after, p_after = f_test(np.hstack([ones, cov]), np.hstack([ones, cov, G]), y)
        # seed term
        ys = np.array([r[key] for r in seeded], dtype = float)
        Gs, Ss = dummies([r["group"] for r in seeded]), dummies([str(r["seed"]) for r in seeded])
        os_ = np.ones((len(seeded), 1))
        F_seed, p_seed = f_test(np.hstack([os_, Gs]), np.hstack([os_, Gs, Ss]), ys)
        seed_means = {s: np.mean([r[key] for r in seeded if r["seed"] == s]) for s in SEEDS}
        print(f"{key:11s} settings: F {F_group:5.2f} p {p_group:.4f} (Kruskal-Wallis p {p_kw:.4f}), ICC {icc:+.2f} | "
              f"speed covariates R^2 {r2_cov:.2f}, settings after speed p {p_after:.3f} | "
              f"seed: p {p_seed:.3f}, means s4 {seed_means[4]:.3f} s42 {seed_means[42]:.3f} s1 {seed_means[1]:.3f}")


def main():
    series = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(ROOT / "data" / "loop_metrics.csv") as file:
        for row in csv.DictReader(file):
            if row["tag"] in TAGS:
                series[row["run"]][TAGS[row["tag"]]][int(row["loop"])] = float(row["value"])

    rows_a, rows_b = [], []
    for run, s in series.items():
        if not run.startswith("a2_grouped_") or any(len(s[t]) < 100 for t in TAGS.values()):
            continue
        group, seed = group_of(run)
        kind = "pair" if "+" in group else ("base" if group == "base" else "arm")
        rows_a.append({"run": run, "group": group, "seed": seed, "kind": kind, "window": "50-99", **run_measures(s, 50, 100)})
        if kind == "pair":
            rows_b.append({"run": run, "group": group, "seed": seed, "kind": kind, "window": "100-174", **run_measures(s, 100, 175)})

    with open(ROOT / "data" / "stability_runs.csv", "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = list(rows_a[0]))
        writer.writeheader()
        writer.writerows(rows_a + rows_b)

    measures = ["rel_noise", "flip_mean", "flip_loops", "drop_loops", "mean_return"]
    analyse([r for r in rows_a if r["group"] != "pmpo"], "loops 50-99, base + arms + pairs (no PMPO)", measures)
    analyse(rows_b, "loops 100-174, pairs", measures)

    # The link between speed and flips, run by run.
    for title, rows in (("loops 50-99, all runs except PMPO", [r for r in rows_a if r["group"] != "pmpo"]), ("loops 100-174, pairs", rows_b)):
        print(f"\n== {title}: Spearman correlation with the flipped fraction")
        flip = [r["flip_mean"] for r in rows]
        for key in ("vel", "sat", "ent", "mean_return"):
            rho, p = stats.spearmanr([r[key] for r in rows], flip)
            print(f"  {key:12s} rho {rho:+.2f} p {p:.4f}")

    print("\n== loops 100-174, pairs: seeds of each pair")
    for group in sorted({r["group"] for r in rows_b}):
        line = " | ".join(f"s{r['seed']}: noise {r['rel_noise']:.3f} flip {r['flip_mean']:.3f} flip_loops {r['flip_loops']:2d} vel {r['vel']:.2f}"
                          for r in sorted((r for r in rows_b if r["group"] == group), key = lambda r: SEEDS.index(r["seed"])))
        print(f"{group:18s} {line}")


if __name__ == "__main__":
    main()
