"""Short RL run of an ablation entry for a throughput test (loop time, GPU memory).

It runs train_gym.main with the train kwargs of train/ablations.py entry --index, with a new
world-model micro-batch, --loops loops, no checkpoints, and a log dir under --log_root.
The loop time comes from audit/progress/elapsed_hours (one value per loop).

Usage:  python bench_rl_throughput.py --index 17 --micro 64 --loops 12 --log_root <dir> --name m64_s42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from train import ablations as A  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type = int, required = True)
    parser.add_argument("--micro", type = int, required = True)
    parser.add_argument("--loops", type = int, default = 12)
    parser.add_argument("--log_root", required = True)
    parser.add_argument("--name", required = True)
    args = parser.parse_args()

    spec = A.build_ablation_specs(ablation_indices = [args.index])[0]
    kwargs = dict(spec.train_kwargs)
    kwargs.update(
        world_model_micro_batch_size = args.micro,
        stop_after_loops = args.loops,
        checkpoint_every = 0,
        log_dir = args.log_root,
        run_name = args.name,
        unique_log_dir = False,
    )
    from train.train_gym import main as train_main
    train_main(**kwargs)


if __name__ == "__main__":
    main()
