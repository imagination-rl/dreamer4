"""Effect of the world-model trunk on a fresh policy head: pretrained trunk against online A2 trunks.

Each trunk gets the same fresh policy and value heads (same seed), as the RL stage does after it loads a pretrained
checkpoint. On the same held-out offline windows (random mode: states near the first online states; expert + noise
mode), the script measures:
  - the scale of the policy-head input (L2 norm of a vector, std of one feature across states),
  - the entropy of the fresh policy,
  - the change of the policy after one Adam step (lr 3e-4, as agent_learning_rate) on a fixed PPO-like surrogate
    with random +-1 advantages: the mean KL(old || new) and the entropy change.
Adam normalizes the gradient scale, so the change of the output for one step grows with the input scale.

Output: report/9-29-26-pretrain/data/probe_policy_input.csv
"""

from __future__ import annotations

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
from probe_common import build, seed_all
from probe_drift import policy_on_contexts
from train import offline_pretraining as off

BASE = "audit_a2_next_state_reward_target"
PRETRAIN = REPO / "runs" / "offline_pretrain" / "a2_next_state_reward"
DATASET = REPO / "runs" / "offline_data" / "HalfCheetah-v4" / "halfcheetah_v4_mix_1m"
TMP = HERE / "_tmp_probe_replays"
OUT = HERE.parent / "data" / "probe_policy_input.csv"
BATCH, SEQ, EPISODES = 16, 32, 8

TRUNKS = {
    "pretrained step 12000 (RL start)": PRETRAIN / "pretrained_step_12000.pt",
    "H100 A2 s42 loop 25": RUNS / BASE / "checkpoints" / "loop_25.pt",
    "H100 A2 s42 loop 75": RUNS / BASE / "checkpoints" / "loop_75.pt",
    "H100 A2 s4 loop 50": RUNS / (BASE + "2") / "checkpoints" / "loop_50.pt",
    "pretrained + rehearsal s42 loop 50": RUNS / "offline_pretrained_a2_rehearsal" / "checkpoints" / "loop_50.pt",
}


def main():
    device = torch.device("cuda")
    dataset = off.load_offline_dataset(DATASET)
    split = off.offline_split(dataset, 0.05, 10042)
    rows = []

    for name, path in TRUNKS.items():
        seed_all(0)
        h, tokenizer, wm, _, action_range = build(BASE, None)
        pkg = torch.load(path, map_location = device, weights_only = True)
        tokenizer.load_state_dict(pkg["tokenizer"])
        # All trunks keep the same fresh agent heads, as the RL stage after a pretrained load.
        off.load_world_model_without_agent_keys(wm, pkg["world_model"])
        head_params = list(wm.policy_head_parameters())

        for mode in ("random", "expert_noise"):
            indices = split.validation[dataset.mode[split.validation] == off.OFFLINE_MODE_IDS[mode]][:EPISODES]
            folder = TMP / f"{name.split(' (')[0].replace(' ', '_')}_{mode}"
            shutil.rmtree(folder, ignore_errors = True)
            replay = off.build_offline_replay(tokenizer, dataset, indices, folder, step_size = off.interact_step_size(wm),
                                              action_range = action_range, device = device, marker = {})
            window = ReplayDatasetTimeWindow(replay, window_length = SEQ, include_metadata = True)
            sampler = RandomSampler(window, replacement = True, num_samples = BATCH, generator = torch.Generator().manual_seed(0))
            batch = next(iter(DataLoader(window, batch_size = BATCH, sampler = sampler)))
            latents = batch["latents"].to(device)
            actions = batch["actions_continuous"].to(device)

            dist, value, policy_input = policy_on_contexts(wm, latents, actions)
            x = policy_input.float()

            # One Adam step on a fixed surrogate: -mean(A * log pi(a)) with random +-1 advantages and sampled actions.
            state = [p.detach().clone() for p in head_params]
            gen = torch.Generator(device = device).manual_seed(1)
            sampled = dist.sample()
            advantages = torch.randint(0, 2, sampled.shape[:-1], generator = gen, device = device).float() * 2 - 1
            optimizer = torch.optim.Adam(head_params, lr = 3e-4, eps = 1e-5)
            wm.eval()
            with torch.enable_grad():
                policy_embed = wm.policy_head(x)
                _, params = wm.action_embedder.unembed(policy_embed, pred_head_index = 0)
                new_dist = wm.action_embedder.continuous_readout.continuous_dist.dist(params)
                loss = -(advantages * new_dist.log_prob(sampled).sum(-1)).mean()
                optimizer.zero_grad()
                loss.backward()
            grad_norm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in head_params if p.grad is not None)))
            optimizer.step()
            with torch.no_grad():
                _, params = wm.action_embedder.unembed(wm.policy_head(x), pred_head_index = 0)
                stepped = wm.action_embedder.continuous_readout.continuous_dist.dist(params)
                kl = torch.distributions.kl_divergence(dist, stepped).sum(-1).mean()
                entropy_change = (stepped.entropy().sum(-1) - dist.entropy().sum(-1)).mean()
            with torch.no_grad():
                for p, s in zip(head_params, state):
                    p.copy_(s)

            rows.append(dict(
                trunk = name, mode = mode,
                input_norm = float(x.norm(dim = -1).mean()),
                input_feature_std = float(x.flatten(0, -2).std(0).mean()),
                input_feature_mean_abs = float(x.flatten(0, -2).mean(0).abs().mean()),
                fresh_entropy = float(dist.entropy().sum(-1).mean()),
                surrogate_grad_norm = grad_norm,
                kl_after_one_adam_step = float(kl),
                entropy_change_after_one_adam_step = float(entropy_change),
            ))
            print(rows[-1], flush = True)
            shutil.rmtree(folder, ignore_errors = True)

    shutil.rmtree(TMP, ignore_errors = True)
    OUT.parent.mkdir(parents = True, exist_ok = True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index = False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index = False))


if __name__ == "__main__":
    main()
