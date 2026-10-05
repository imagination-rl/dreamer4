"""Flip onsets in the real episodes of the replay: what happens just before the cheetah turns over.

A flip onset is the first step t0 of an episode at which the root angle is past pi/2 and stays past it for 50 steps.
For each onset with t0 >= 300, the script compares the 30 steps before t0 with a baseline window (t0 - 250 .. t0 - 150) of the same
episode, and with the same steps of upright episodes of the same loop (no flipped step):
  surprise   -log pi(a|s) of the sampled action (its mean over states is the entropy). A high value means that the sampled
             action was improbable under the policy: the flip came from exploration noise. A normal value means that
             the flip came from the policy's usual actions.
  saturation fraction of action dims within 5 % of the bounds; x velocity; |pitch rate| (obs index 10).
It also counts the onsets with a recovery (upright again for 50 steps after the onset).

Runs: the 18 stage-1 pair runs, loops 100-174, and the base and arm runs, loops 50-99.
Writes ../data/flip_onsets.csv (one row for each onset) and prints a summary for each group.
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT.parents[1] / "runs" / "gym_imagination"
STEPS, HOLD, PRE, MIN_T0 = 1000, 50, 30, 300
ROOT_ANGLE, X_VELOCITY, PITCH_RATE = 1, 8, 10


def flipped_of(states):
    return np.abs(np.angle(np.exp(1j * states[..., ROOT_ANGLE]))) > np.pi / 2


def onset(flip):
    """First t with HOLD flipped steps from t, or -1."""
    run_length = np.convolve(flip.astype(np.int32), np.ones(HOLD, dtype = np.int32), "valid")
    hits = np.nonzero(run_length == HOLD)[0]
    return int(hits[0]) if len(hits) else -1


def run_onsets(job):
    name, lo, hi = job
    replay = Path(os.path.realpath(RUNS / name)) / "replay_buffer"
    if not (replay / "log_probs_continuous.data.npy").exists():
        return []
    load = lambda f: np.load(replay / f"{f}.data.npy", mmap_mode = "r")
    log_probs, actions, states = load("log_probs_continuous"), load("actions_continuous"), load("critic_state")
    rows = []
    for loop in range(lo, hi):
        episodes = slice(loop * 64, loop * 64 + 64)
        s = np.asarray(states[episodes, :STEPS])
        surprise = -np.asarray(log_probs[episodes, :STEPS]).sum(-1) + 6 * np.log(2.)
        saturation = (np.abs(np.asarray(actions[episodes, :STEPS]) - 0.5) > 0.45).mean(-1)
        flip = flipped_of(s)
        upright = ~flip.any(1)
        for e in range(64):
            t0 = onset(flip[e])
            if t0 < MIN_T0:
                continue
            pre, base = slice(t0 - PRE, t0), slice(t0 - 250, t0 - 150)
            after = flip[e, t0 + HOLD:]
            recovered = len(after) >= HOLD and onset(~after) >= 0
            ref = lambda x: float(x[upright, pre].mean()) if upright.any() else float("nan")
            rows.append({
                "run": name, "loop": loop, "episode": e, "t0": t0, "recovered": int(recovered),
                "surprise_pre": float(surprise[e, pre].mean()), "surprise_base": float(surprise[e, base].mean()), "surprise_upright_ref": ref(surprise),
                "surprise_pre_max": float(surprise[e, pre].max()), "surprise_base_p95": float(np.percentile(surprise[e, base], 95)),
                "saturation_pre": float(saturation[e, pre].mean()), "saturation_base": float(saturation[e, base].mean()), "saturation_upright_ref": ref(saturation),
                "velocity_pre": float(s[e, pre, X_VELOCITY].mean()), "velocity_base": float(s[e, base, X_VELOCITY].mean()), "velocity_upright_ref": ref(s[..., X_VELOCITY]),
                "pitch_rate_pre": float(np.abs(s[e, pre, PITCH_RATE]).mean()), "pitch_rate_base": float(np.abs(s[e, base, PITCH_RATE]).mean()),
                "pitch_rate_upright_ref": ref(np.abs(s[..., PITCH_RATE])),
            })
    return rows


def group_of(run):
    return run[len("a2_grouped_"):].rsplit("_s", 1)[0]


def main():
    names = sorted({p.name for p in RUNS.iterdir() if p.name.startswith("a2_grouped_")})
    jobs = []
    for name in names:
        if name.startswith("a2_grouped_combo_"):
            jobs.append((name, 100, 175))
        elif not name.startswith(("a2_grouped_pmpo", "a2_grouped_flat")):
            jobs.append((name, 50, 100))
    with ProcessPoolExecutor(max_workers = 8) as pool:
        rows = [row for result in pool.map(run_onsets, jobs) for row in result]
    with open(ROOT / "data" / "flip_onsets.csv", "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} onsets with t0 >= {MIN_T0}")
    keys = ["surprise", "saturation", "velocity", "pitch_rate"]
    groups = sorted({group_of(r["run"]) for r in rows})
    print(f"{'group':34s} {'n':>4s} {'recov':>6s} " + " ".join(f"{k + ' pre/base/ref':>27s}" for k in keys))
    for group in groups + ["ALL"]:
        sub = [r for r in rows if group == "ALL" or group_of(r["run"]) == group]
        cells = " ".join(f"{np.mean([r[k + '_pre'] for r in sub]):8.2f} {np.mean([r[k + '_base'] for r in sub]):8.2f} {np.nanmean([r[k + '_upright_ref'] for r in sub]):8.2f}" for k in keys)
        print(f"{group:34s} {len(sub):4d} {np.mean([r['recovered'] for r in sub]):6.2f} {cells}")
    pre = np.array([r["surprise_pre"] - r["surprise_base"] for r in rows])
    print(f"\nsurprise pre - base: mean {pre.mean():+.3f}, median {np.median(pre):+.3f}, fraction > 0 {np.mean(pre > 0):.2f}")
    spike = np.array([r["surprise_pre_max"] > r["surprise_base_p95"] for r in rows])
    print(f"onsets whose 30 pre-onset steps hold a surprise above the baseline 95th percentile: {spike.mean():.2f} (chance about {1 - 0.95 ** 30:.2f})")


if __name__ == "__main__":
    main()
