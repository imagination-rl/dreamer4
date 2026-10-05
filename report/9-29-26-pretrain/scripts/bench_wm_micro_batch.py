"""World-model step time and peak VRAM against world_model_micro_batch_size, for the offline pretrain stage.

The step is train_gym.train_world_model with the batch shape of the pretrain stage (128 windows x 200 steps).
The model is the A2 world model of a2_reduced_reward_transformer_gradient2 (loop 150): the same architecture
as the A2 pretrain run. The window content does not change the step time.
Each case compiles the loss, runs 6 warm-up steps (both shortcut branches), then times --calls calls of 13 steps.

Output: report/9-29-26-pretrain/data/bench_wm_micro_batch.csv
Usage:  python bench_wm_micro_batch.py [--calls 2]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "reports" / "9-28-26" / "scripts"))

import numpy as np
import pandas as pd
import torch

from common import RUNS
from probe_common import build, checkpoints, seed_all
from bench_optimized import PowerSampler

from memmap_replay_buffer import ReplayBuffer

from train.compile_runtime import WorldModelTrainingLoss, configure_sac
from train import train_gym

GIB = 1024 ** 3
BATCH, SEQ = 128, 200
OUT = HERE.parent / "data" / "bench_wm_micro_batch.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default = "a2_reduced_reward_transformer_gradient2")
    parser.add_argument("--loop", type = int, default = 150)
    parser.add_argument("--calls", type = int, default = 2)
    args = parser.parse_args()

    train_gym.configure_matmul_precision(True, torch.device("cuda"))
    configure_sac(0.825)
    seed_all(0)
    ckpt = [c for c in checkpoints(args.run) if c.endswith(f"loop_{args.loop}.pt")][0]
    h, _, wm, _, _ = build(args.run, ckpt)
    replay = ReplayBuffer.from_folder(RUNS / args.run / "checkpoints" / f"replay_loop_{args.loop}", read_only = True)
    optimizer = train_gym.make_optimizer(
        wm.world_model_parameters(), lr = h["world_model_learning_rate"], weight_decay = h["optimizer_weight_decay"],
        use_muon = h["use_muon_optimizer"], muon_params = wm.muon_parameters(),
    )
    pools = [p for p in [*wm.transformer.attn_pools, wm.transformer.final_attn_pool] if p is not None and hasattr(p, "fn")]
    wm.train()
    rows = []

    def case(micro, bf16, einsum):
        for pool in pools:
            pool.fn.einsum_attend = einsum
        label = f"{'bf16' if bf16 else 'fp32'} {'einsum' if einsum else 'sdpa'} pools, micro {micro}"
        torch._dynamo.reset()
        optimizer.zero_grad(set_to_none = True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        compiled = torch.compile(WorldModelTrainingLoss(wm), dynamic = False)
        times, status, t_start = [], "ok", time.time()

        def run(steps):
            torch.cuda.synchronize(); t0 = time.time()
            train_gym.train_world_model(
                wm, optimizer, replay, world_model_loss_fn = compiled, world_model_loss_timing = None, world_model_step_timing = None,
                steps = steps, batch_size = BATCH, sequence_length = SEQ, static_batch_shape = True, max_grad_norm = h["max_grad_norm"],
                global_step = 0, writer = None, training_profiler = None, shortcut_train_mode = "combined", desc = label,
                micro_batch_size = micro, autocast_bf16 = bf16, update_loss_ema = True,
            )
            torch.cuda.synchronize(); times.append((time.time() - t0) / steps)

        power = None
        try:
            run(6)
            with PowerSampler() as power:
                for _ in range(args.calls):
                    run(13)
        except torch.cuda.OutOfMemoryError:
            status = "OOM"
        except Exception as error:   # the SDPA pools fail in bf16 at large micro-batches
            status = f"error: {type(error).__name__}: {str(error).splitlines()[0][:120]}"
        steady = times[1:]
        step = float(np.median(steady)) if steady else float("nan")
        row = dict(case = label, micro_batch = micro, bf16 = bf16, einsum_pools = einsum, status = status,
                   step_sec = step, hours_per_20k_steps = step * 20000 / 3600,
                   frames_per_sec = BATCH * SEQ / step if steady else float("nan"),
                   peak_alloc_gib = torch.cuda.max_memory_allocated() / GIB, peak_reserved_gib = torch.cuda.max_memory_reserved() / GIB,
                   mean_power_w = power.power_w if steady and power else float("nan"),
                   mean_util_pct = power.util_pct if steady and power else float("nan"),
                   case_sec = time.time() - t_start)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush = True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index = False)

    for bf16, einsum in ((False, False), (False, True), (True, True), (True, False)):
        for micro in (16, 32, 64, 128):
            case(micro, bf16, einsum)


if __name__ == "__main__":
    main()
