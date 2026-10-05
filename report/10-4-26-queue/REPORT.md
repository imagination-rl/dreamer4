# Queue 10-4-26: Runs, Flags, and Reasons

## 1. Summary

- Queue script: `scripts/queue_10-4-26.sh`. Record: `runs/queues/10-4-26_2026-10-04_210731`. Launch log: `runs/queues/queue_10-4-26_launch.log`.
- Earlier launches (records `10-4-26_2026-10-04_121901_superseded` and `10-4-26_2026-10-04_204439_superseded`) were stopped before any run started: at 20:44 to remove entries 142-144, and at 21:07 to add entries 163-165 (section 3).
- Start: 2026-10-04 23:00. The queue runs two runs at a time on the GPU.
- GPU break: no run starts if it can overlap 06:00-08:00. The queue expects 70 minutes for each run. The stage-1 runs took 50-63 minutes.
- Size: 27 runs, entries 136-141 and 145-165 of `train/ablations.py`, 9 settings × 3 seeds (4, 42 and 1).
- Length: each run stops after 175 loops (`stop_after_loops: 175`, `num_loops: 500`). It saves one checkpoint and one replay snapshot, at loop 175 (`checkpoint_every: 175`).
- Disk: `DELETE_LATENTS=1`. After a run ends with exit code 0, the queue deletes the two `latents.data.npy` files of that run (the live replay and the loop-175 snapshot). `deleted_latents.txt` in the record lists each deleted path. A failed run keeps its files.
- Order: each lane starts with one pretraining run (163, 164), so the 3-hour runs end before the GPU break. Then seed 4 of the other settings, then seed 42, then seed 1. Entry 165 is the last run of lane a. The GPU-break check uses 180 minutes for the pretraining runs. If the queue stops early, each setting still has its first seeds.

## 2. Check of the Settings

I compared the `train_kwargs` of each entry in the record `run_settings.json` with the stage-1 run of the same seed: `a2_grouped_combo_symexp_warmup15_s4/_s42/_s1` (entries 127, 128 and 129).

- All 27 entries have `stop_after_loops = 175`, `checkpoint_every = 175`, and seeds 4, 42 and 1.
- Each entry differs from its stage-1 run only in the flags of the next table, and in `run_name` and `run_details`.

## 3. Runs

The common base of all runs is stage-1 symexp + warmup15:
- The grouped dynamics-loss normalization, with the plain loss sum before loop 15 (`dynamics_loss_normalization_start_loop: 15`).
- The closer CleanRL PPO (`cleanrl_updated_ppo`, `cleanrl_ppo_closer`, Beta policy, `agent_learning_rate: 3e-4`, entropy weight 0).
- The symexp two-hot value head beside a raw HL-Gauss critic for GAE (`use_symexp_value_head`, `use_hl_gauss_actor_critic`, `cleanrl_ppo_closer_hl_gauss_value_loss`).

| Entries | Run names (`a2_grouped_combo_symexp_warmup15_*_s{4,42,1}`) | Flags that change | Reason |
| --- | --- | --- | --- |
| 136-138 | `block2` | `attn_res: block`, `use_attn_pool: false`, `dynamics_ff_expansion_factor: 7` | Stage 2 by the stage-1 rule. Only symexp + warmup15 was better than both of its arms (+439, p 0.022, and +595, p 0.002, at loops 95-99). The rule then adds the better arm of the other pair. Of the attention arms, block2 had the highest return (1,915 at loops 95-99). |
| 139-141 | `depth5` | `use_attn_pool: false`, `depth: 5` | The second attention arm. block2 is not significantly better than depth5 (1,915 against 1,740, p about 0.26). rw4 + depth5 had the best world model of stage 1 (action-ranking ρ 0.54, p 0.017). |
| 142-144 | `wmpre640` | `pretrain_world_model_combined_steps: 640` (base 256) | **Removed from the queue.** The pretrain steps train the world model at loop 0 on the replay, which then holds only the 64 episodes of the untrained policy (`train/train_gym.py`, the `did_world_model_pretrain` block). The pretraining must use the precollected dataset (`offline_pretraining`). |
| 163-165 | `a2_grouped_combo_symexp_scaled40pre_s{4,42,1}` | `offline_pretraining: true`, `offline_stages: [pretrain, rl]`, `offline_dataset_name: halfcheetah_v4_mix_scaled40_5p5m`, `offline_world_model_steps: 12000`, `offline_rehearsal_fraction: 0`, `dynamics_loss_normalization_start_loop: 0` (no warmup15) | Pretraining on the precollected dataset, in place of `wmpre640`. One run pretrains the tokenizer and the world model on the existing scaled40 data (the data and length of the best local pretrained run), then runs 175 online loops without rehearsal. No new collection and no old checkpoint (the old pretrains have `use_dynamics_loss_normalization: false`). warmup15 is out: the code rejects it on the offline path, and pretraining gives the same head start. CPU smoke test passed. About 3 hours each. |
| 145-147 | `lr150` | `agent_learning_rate: 1.5e-4` | PPO learning-rate sweep (you asked for it). It tests whether a smaller policy step reduces the late drops. |
| 148-150 | `lr600` | `agent_learning_rate: 6e-4` | The other side of the sweep: does a faster policy learn faster, or become less stable? |
| 151-153 | `ent001` | `cleanrl_ppo_entropy_weight: 0.001` | The policy entropy falls to about −5 nats by loop 175. Saturated actions predict flipped episodes (ρ +0.52). A small entropy bonus can slow the collapse. |
| 154-156 | `ent01` | `cleanrl_ppo_entropy_weight: 0.01` | A strong entropy bonus (the value of the old PMPO setup). |
| 157-159 | `wmlr300` | `world_model_learning_rate: 3e-4` (base 6e-4) | World-model learning-rate sweep. A slower world model moves the shared representation less in each loop. |
| 160-162 | `wmlr1200` | `world_model_learning_rate: 1.2e-3` | A faster world model can follow the faster policy (1-step velocity error 0.45 at high speed, against 0.16 early). |

Not in this queue:
- The discount (0.99). It is fixed in code (`gae_discount_factor = 0.99`), not a flag.
- GAE λ, horizon, PPO clip and epochs. The logged PPO step has the same size in all stage-1 pairs (approx-KL 0.03-0.05).
- The warmup start loop.

## 4. Link to the Instability Audit

`report/10-4-26-instability-audit/REPORT.md` measures the causes of the instability. Its results give each sweep a clear prediction:
- `lr150` and `lr600` change the size of the PPO step. Each world-model PPO update is about 5 times larger in KL than a model-free update with the same settings.
- `wmlr300` and `wmlr1200` change the world-model drift of the policy. One loop of world-model training moves the acting policy by about as much as one PPO update, or more.
- `ent001` and `ent01` change the entropy collapse. At the same real return, the policy entropy is about 2-3 nats below that of model-free PPO.
