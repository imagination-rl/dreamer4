# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Plot the SLURM queue wait history of the cluster.

The script reads job records with `sacct` only. It does not submit, change, or
cancel jobs. It fetches the history in date chunks, sleeps between queries, and
caches each closed chunk on disk, so a second run sends no queries for old chunks.

Wait = Start - Eligible. Eligible excludes time that a job spends in a hold or on a
dependency. Jobs that never started are not in the averages.

Usage:
    uv run scripts/slurm_wait_history.py
    uv run scripts/slurm_wait_history.py --bin day --since 2026-08-01
    uv run scripts/slurm_wait_history.py --offline   # plot from the cache only
"""

import argparse
import datetime as dt
import pathlib
import re
import statistics
import subprocess
import sys
import time

FIELDS = "JobIDRaw,Partition,Submit,Eligible,Start,State,ReqTRES,NNodes"
FIRST_DAY = dt.date(2024, 8, 14)  # first job record in the DeltaAI accounting database
OPEN_DAYS = 3  # re-fetch chunks that end less than this many days ago


def parse_time(s):
    if s in ("", "Unknown", "None"):
        return None
    return dt.datetime.fromisoformat(s)


def run_sacct(start, end, partitions):
    cmd = ["sacct", "-a", "-X", "-n", "-P", "-S", start.isoformat(), "-E", end.isoformat(), "-o", FIELDS]
    if partitions:
        cmd += ["-r", partitions]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0 or "Too wide" in out.stderr:
        raise RuntimeError(out.stderr.strip() or f"sacct exit {out.returncode}")
    return out.stdout


def fetch(args):
    """Fill the cache chunk by chunk. Returns the list of cache files."""
    cache = pathlib.Path(args.cache) / (args.partitions or "all")
    cache.mkdir(parents=True, exist_ok=True)
    today = dt.date.today()
    files, queries = [], 0
    start = max(args.since, FIRST_DAY)
    while start <= today:
        end = min(start + dt.timedelta(days=args.chunk_days), today + dt.timedelta(days=1))
        f = cache / f"{start}_{end}.txt"
        closed = end < today - dt.timedelta(days=OPEN_DAYS)
        if not args.offline and not (f.exists() and closed):
            if queries >= args.max_queries:
                print(f"stop: --max-queries={args.max_queries} reached; run again to continue", file=sys.stderr)
                break
            if queries:
                time.sleep(args.sleep)
            print(f"sacct {start} .. {end}", file=sys.stderr)
            text = run_sacct(start, end, args.partitions)  # on "Too wide", rerun with a smaller --chunk-days
            queries += 1
            tmp = f.with_suffix(".part")
            tmp.write_text(text)
            tmp.replace(f)
        if f.exists():
            files.append(f)
        start = end
    return files


def load(files):
    """Read cached rows and keep the last record of each job ID."""
    jobs = {}
    for f in files:
        for line in f.read_text().splitlines():
            p = line.split("|")
            if len(p) != 8:
                continue
            jid, part, sub, elig, start, state, tres, nn = p
            elig, start = parse_time(elig) or parse_time(sub), parse_time(start)
            if not elig or not start or start < elig:
                continue
            m = re.search(r"gres/gpu=(\d+)", tres)
            jobs[jid] = dict(
                part=part,
                elig=elig,
                wait=(start - elig).total_seconds() / 3600,
                gpus=int(m.group(1)) if m else 0,
                nodes=int(nn or 1),
            )
    return list(jobs.values())


def bin_key(t, how):
    if how == "day":
        return t.date()
    if how == "week":
        return (t - dt.timedelta(days=t.weekday())).date()
    return t.date().replace(day=1)


def aggregate(jobs, how, keep):
    groups = {}
    for j in jobs:
        if keep(j):
            groups.setdefault(bin_key(j["elig"], how), []).append(j["wait"])
    rows = []
    for k in sorted(groups):
        w = sorted(groups[k])
        q = lambda p: w[min(len(w) - 1, int(p * len(w)))]
        rows.append(dict(t=k, n=len(w), mean=statistics.fmean(w), med=q(0.5), p90=q(0.9), le15=sum(x <= 0.25 for x in w) / len(w)))
    return rows


def plot(series, how, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3, 1.3]))
    colors = ["tab:blue", "tab:red", "tab:green"]
    for (name, rows), c in zip(series.items(), colors):
        t = [r["t"] for r in rows]
        ax[0].plot(t, [r["mean"] * 60 for r in rows], "-", color=c, lw=1.8, label=f"{name}: mean")
        ax[0].plot(t, [r["med"] * 60 for r in rows], "--", color=c, lw=1.2, label=f"{name}: median")
        ax[1].plot(t, [100 * r["le15"] for r in rows], "-", color=c, lw=1.5, label=name)
        ax[2].plot(t, [r["n"] for r in rows], "-", color=c, lw=1.5, label=name)
    ax[0].set_yscale("symlog", linthresh=1)
    ax[0].set_yticks([0, 1, 2, 5, 15, 60, 180, 600, 1440, 2880])
    ax[0].set_yticklabels(["0", "1m", "2m", "5m", "15m", "1h", "3h", "10h", "1d", "2d"])
    ax[0].set_ylabel("wait (Start - Eligible)")
    ax[0].set_title(f"SLURM queue wait per {how} (started jobs, grouped by eligible time)")
    ax[1].set_ylabel("% started\nwithin 15 min")
    ax[1].set_ylim(0, 100)
    ax[2].set_ylabel(f"jobs per {how}")
    ax[2].set_yscale("log")
    for a in ax:
        a.grid(True, which="major", alpha=0.3)
        a.legend(loc="upper left", fontsize=8, ncol=2)
    ax[2].xaxis.set_major_locator(mdates.MonthLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partitions", default="ghx4", help="sacct -r value; empty string for all partitions")
    ap.add_argument("--since", type=dt.date.fromisoformat, default=FIRST_DAY)
    ap.add_argument("--bin", choices=["day", "week", "month"], default="week")
    ap.add_argument("--chunk-days", type=int, default=14, help="days per sacct query")
    ap.add_argument("--sleep", type=float, default=3.0, help="seconds between sacct queries")
    ap.add_argument("--max-queries", type=int, default=80, help="upper limit of sacct queries per run")
    ap.add_argument("--cache", default=str(pathlib.Path.home() / ".cache" / "slurm_wait_history"))
    ap.add_argument("--offline", action="store_true", help="use the cache only; send no sacct queries")
    ap.add_argument("--out", default="slurm_wait_history.png")
    args = ap.parse_args()

    jobs = load(fetch(args))
    jobs = [j for j in jobs if j["elig"].date() >= args.since]
    print(f"{len(jobs)} started jobs", file=sys.stderr)
    series = {
        "all jobs": aggregate(jobs, args.bin, lambda j: True),
        "1 GPU, 1 node": aggregate(jobs, args.bin, lambda j: j["gpus"] == 1 and j["nodes"] == 1),
        "multi-node": aggregate(jobs, args.bin, lambda j: j["nodes"] > 1),
    }
    print(f"{'bin':10} {'jobs':>7} {'mean_min':>9} {'med_min':>8} {'p90_min':>8} {'<=15m%':>7}")
    for r in series["all jobs"]:
        print(f"{r['t']!s:10} {r['n']:7} {60 * r['mean']:9.1f} {60 * r['med']:8.1f} {60 * r['p90']:8.1f} {100 * r['le15']:7.1f}")
    plot(series, args.bin, args.out)


if __name__ == "__main__":
    main()
