"""Raw flow loss of the offline-pretrained world model against the online A2 world model, on the same held-out episodes.

Each model encodes the 50 held-out offline episodes (15 expert, 20 expert + noise, 15 random) with its own tokenizer.
The loss is WorldModelTrainingLoss ("combined", no EMA update, no gradient) on 200-step windows with the same seeds for
each model. The script also gives latent statistics per tokenizer: the variance of the latents and the mean squared
change of a latent between two steps. The flow loss is an MSE on the latents, so its scale follows these statistics.

Output: report/9-29-26-pretrain/data/flow_compare.csv
Usage:  python flow_compare.py [--step 10000] [--batches 8]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "reports" / "9-28-26" / "scripts"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from memmap_replay_buffer.replay_buffer import ReplayDatasetTimeWindow

from common import RUNS
from probe_common import build
from train import offline_pretraining as off
from train import train_gym
from train.compile_runtime import WorldModelTrainingLoss

BASE = "audit_a2_next_state_reward_target"
PRETRAIN = REPO / "runs" / "offline_pretrain" / "a2_next_state_reward"
DATASET = REPO / "runs" / "offline_data" / "HalfCheetah-v4" / "halfcheetah_v4_mix_1m"
TMP = Path(sys.argv[0]).resolve().parent / "_tmp_replays"
OUT = HERE.parent / "data" / "flow_compare.csv"
BATCH, SEQ = 16, 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type = int, default = 10000)
    parser.add_argument("--batches", type = int, default = 8)
    parser.add_argument("--tmp", default = str(TMP))
    args = parser.parse_args()
    tmp = Path(args.tmp)
    device = torch.device("cuda")

    dataset = off.load_offline_dataset(DATASET)
    split = off.offline_split(dataset, 0.05, 10042)
    models = {
        f"offline pretrain step {args.step}": dict(pkg = PRETRAIN / f"pretrained_step_{args.step}.pt"),
        "online A2 loop 25": dict(pkg = RUNS / BASE / "checkpoints" / "loop_25.pt"),
        "online A2 loop 125": dict(pkg = RUNS / BASE / "checkpoints" / "loop_125.pt"),
    }
    rows = []
    for name, spec in models.items():
        h, tokenizer, wm, _, action_range = build(BASE, None)
        pkg = torch.load(spec["pkg"], map_location = device, weights_only = True)
        tokenizer.load_state_dict(pkg["tokenizer"])
        res = wm.load_state_dict(pkg["world_model"], strict = False)
        assert not res.unexpected_keys and all(k in off.agent_state_keys(wm) for k in res.missing_keys), res
        wm.train()
        loss_fn = WorldModelTrainingLoss(wm)

        for mode, mode_id in off.OFFLINE_MODE_IDS.items():
            indices = split.validation[dataset.mode[split.validation] == mode_id]
            latents = off.encode_observations(tokenizer.eval(), dataset.obs[indices], device)   # (E, T+1, n, d)
            folder = tmp / f"{name.replace(' ', '_')}_{mode}"
            shutil.rmtree(folder, ignore_errors = True)
            replay = off.build_offline_replay(tokenizer, dataset, indices, folder, step_size = off.interact_step_size(wm),
                                              action_range = action_range, device = device, marker = {})
            window = ReplayDatasetTimeWindow(replay, window_length = SEQ, include_metadata = True)
            # One dataset item per episode: sample episodes with replacement. The seeds fix the window starts.
            torch.manual_seed(0); np.random.seed(0)
            sampler = RandomSampler(window, replacement = True, num_samples = BATCH * args.batches, generator = torch.Generator().manual_seed(0))
            loader = iter(DataLoader(window, batch_size = BATCH, sampler = sampler))
            flow, shortcut, reward = [], [], []
            for b in range(args.batches):
                exp = train_gym.replay_batch_to_experience(next(loader), device)
                torch.manual_seed(100 + b); torch.cuda.manual_seed_all(100 + b)
                with torch.no_grad():
                    _, losses = loss_fn(exp.latents, exp.rewards, exp.terminals, exp.actions.continuous, exp.lens, "combined", False)
                flow.append(losses.diagnostics.raw_flow.item())
                shortcut.append(losses.diagnostics.raw_shortcut.item())
                reward.append(losses.diagnostics.raw_reward.item())
            rows.append(dict(
                model = name, mode = mode, episodes = len(indices),
                flow_loss_raw = float(np.mean(flow)), shortcut_loss_raw = float(np.mean(shortcut)), reward_loss_raw = float(np.mean(reward)),
                latent_var = float(latents.float().var(dim = (0, 1)).mean()),
                latent_step_msd = float((latents[:, 1:] - latents[:, :-1]).float().pow(2).mean()),
                latent_abs_max = float(latents.abs().max()),
            ))
            print(rows[-1], flush = True)
            del replay
            shutil.rmtree(folder, ignore_errors = True)
        del wm, tokenizer
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df["flow_over_latent_var"] = df.flow_loss_raw / df.latent_var
    df.to_csv(OUT, index = False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(df.to_string(index = False))


if __name__ == "__main__":
    main()
