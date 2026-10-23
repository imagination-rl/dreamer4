"""Run probe_context_drift.py on the selected checkpoints, one job at a time with 14 threads (one job peaks at about 5 GB RAM).
Each job uses the next CPU set of CPU_SETS, so that the load moves between the cores.

Checkpoints: the 18 stage-1 pair runs at loop_175.pt (episodes of loop 174), and the base and the arms cevalue,
warmup15 and symexp at loop_50.pt (episodes of loop 50, collected by that checkpoint) and loop_100.pt (episodes of loop 99).
Each job writes ../data/probe/<run>__<checkpoint>.csv and ../data/probe/<run>__<checkpoint>.log. A finished job is skipped.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "probe"
SCRATCH = sys.argv[1]
sys.path.insert(0, str(HERE.parents[2]))
from train.ablations import COMBINATION_PAIRS, COMBINATION_SEEDS, combination_run_name  # noqa: E402

JOBS = [(combination_run_name(pair, seed), "loop_175.pt", 174) for pair in COMBINATION_PAIRS for seed in COMBINATION_SEEDS]
ARMS = ["loss_normalization_s4", "loss_normalization_s42", "loss_normalization_s1", "loss_normalization_s2"] + [
    f"{arm}_s{seed}" for arm in ("hl_gauss_value_loss", "norm_warmup15", "symexp_hl_gauss_actor_critic") for seed in COMBINATION_SEEDS]
JOBS += [(f"a2_grouped_{arm}", name, loop) for name, loop in (("loop_50.pt", 50), ("loop_100.pt", 99)) for arm in ARMS]


def run(job, cpus):
    run_name, checkpoint, loop = job
    stem = OUT / f"{run_name}__{checkpoint[:-3]}"
    if Path(f"{stem}.csv").exists():
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "14", "MKL_NUM_THREADS": "14", "MALLOC_ARENA_MAX": "2"}
    with open(f"{stem}.log", "w") as log:
        subprocess.run(["taskset", "-c", cpus, sys.executable, str(HERE / "probe_context_drift.py"), run_name, checkpoint, str(loop), f"{stem}.csv", SCRATCH],
                       stdout = log, stderr = subprocess.STDOUT, env = env, cwd = HERE)


CPU_SETS = ("0-13", "10-23", "20-23,0-9")  # 14 threads: the total stays at 16 with 2 for the other analyses


def main():
    OUT.mkdir(parents = True, exist_ok = True)
    for index, job in enumerate(JOBS):
        run(job, CPU_SETS[index % len(CPU_SETS)])


if __name__ == "__main__":
    main()
