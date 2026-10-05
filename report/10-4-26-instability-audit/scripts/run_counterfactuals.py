"""Run probe_flip_counterfactual.py on the selected checkpoints with WORKERS single-thread processes (about 1 GB RAM each).

Checkpoints: the 18 pair runs at loop_175.pt (onsets of loops 170-174) and the base and the arms cevalue, warmup15 and symexp at
loop_50.pt (onsets of loops 50-54; loop 50 is the checkpoint's own). A job with a .log file is started or done and is skipped,
so a second runner can start later on the same list. Job i runs on CPU (5 i) mod 24, so the load moves between the cores.
Usage: python run_counterfactuals.py WORKERS
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "counterfactual"
sys.path.insert(0, str(HERE.parents[2]))
from train.ablations import COMBINATION_PAIRS, COMBINATION_SEEDS, combination_run_name  # noqa: E402

JOBS = [(combination_run_name(pair, seed), "loop_175.pt", "170-174") for seed in COMBINATION_SEEDS for pair in COMBINATION_PAIRS]
ARMS = ["loss_normalization_s4", "loss_normalization_s42", "loss_normalization_s1", "loss_normalization_s2"] + [
    f"{arm}_s{seed}" for arm in ("hl_gauss_value_loss", "norm_warmup15", "symexp_hl_gauss_actor_critic") for seed in COMBINATION_SEEDS]
JOBS += [(f"a2_grouped_{arm}", "loop_50.pt", "50-54") for arm in ARMS]


def run(index):
    run_name, checkpoint, loops = JOBS[index]
    stem = OUT / f"{run_name}__{checkpoint[:-3]}"
    try:
        log = open(f"{stem}.log", "x")  # an existing log means another runner has the job
    except FileExistsError:
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "2"}
    with log:
        subprocess.run(["taskset", "-c", str((5 * index) % 24), sys.executable, str(HERE / "probe_flip_counterfactual.py"), run_name, checkpoint, loops,
                        f"{stem}.csv", "6"], stdout = log, stderr = subprocess.STDOUT, env = env, cwd = HERE)


def main():
    OUT.mkdir(parents = True, exist_ok = True)
    with ThreadPoolExecutor(int(sys.argv[1])) as pool:
        list(pool.map(run, range(len(JOBS))))


if __name__ == "__main__":
    main()
