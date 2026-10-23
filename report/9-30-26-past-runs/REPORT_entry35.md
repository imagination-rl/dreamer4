# Entry 35: pretrain on the past-runs + scaled-expert data (2026-09-30)

Run: `offline_pretrain_a2_past_runs_scaled_expert_s4` (entry 35, seed 4).
Data: `halfcheetah_v4_mix_past_runs_scaled_expert_5p5m`: expert 750, expert_noise 1500, random 1000, past_runs 1750, expert_scaled 500 episodes. The data is now in `runs/offline_data/archive/HalfCheetah-v4/`.
The run started at 10:33 and was stopped by request at 11:35, after the step 12,000 checkpoint (the checkpoint that the RL runs load).
Checkpoints: `runs/offline_pretrain/a2_past_runs_scaled_expert_s4/pretrained_step_{2000..12000}.pt`.

## 1. Validation world-model loss

| Step | Entry 35 (5.5M data) | Entry 24, old pretrain s4 (1M data) | Entry 23, old pretrain s42 (1M data) |
|---|---|---|---|
| 2,000 | 21.96 | 22.44 | 22.43 |
| 4,000 | 21.69 | 22.37 | 22.36 |
| 6,000 | 21.61 | 22.61 | 22.49 |
| 8,000 | 21.37 | 22.84 | 22.83 |
| 10,000 | 21.44 | 22.88 | 23.03 |
| 12,000 | 21.51 | 23.04 | 23.28 |

The validation sets are different (each run holds out 5 % of its own data), so compare the trends, not the values.
The old pretrains overfit after step 4,000: their loss increases by 0.7 to 0.9 up to step 12,000.
Entry 35 does not overfit: its loss decreases to step 8,000 and is then flat (±0.1).

## 2. Probes on the same online states (repro s4 replay, not in any training data)

Scripts: `report/9-30-26-pretrain-rootcause/scripts/probe_latent_resolution.py` and `probe_action_sources.py`.
Data: `report/9-30-26-past-runs/data/probe_*_entry35.csv`.

One-step forward-velocity prediction (online loops 45-54; copying the last velocity gives an error of 0.40 m/s):

| Model | Error (m/s) | Bias (m/s) | Slope (1 is ideal) | Latent change per 0.1 m/s |
|---|---|---|---|---|
| Old pretrain, entry 24, step 12,000 | 1.42 | −1.36 | 0.21 | 0.028 |
| **Entry 35, step 12,000** | **0.58** | **−0.51** | **0.52** | **0.032** |
| No pretraining (repro s4, loop 50) | 0.24 | +0.18 | 0.87 | 0.136 |

Dream reward per step against MuJoCo, from the same upright online starts (32 steps):

| Model | Online gait: dream / true | Bang-bang: dream / true | Expert: dream / true | Corr. (online) |
|---|---|---|---|---|
| Old pretrain, entry 24 | −0.18 / 1.11 | 0.54 / 0.17 | 1.61 / 3.14 | 0.29 |
| **Entry 35** | **0.49 / 1.11** | **0.77 / 0.17** | **3.04 / 3.14** | **0.85** |
| No pretraining | 1.15 / 1.11 | 0.92 / 0.17 | 1.02 / 3.14 | 0.89 |

## 3. Result

- Entry 35 is much better than the old pretrains:
  - The velocity error is 2.4 times smaller.
  - The velocity bias is 2.7 times smaller.
  - The dream reward of the online gait correlates with the true reward (0.85 against 0.29).
  - Expert actions are rated almost correctly.
- Entry 35 still fails both gates of the root-cause report:
  - The velocity slope is 0.52; the gate is above 0.7.
  - The model still rates bang-bang actions (0.77) above the online gait (0.49). In MuJoCo, the online gait gives 1.11 and bang-bang gives 0.17. PPO in imagination can therefore still push the policy toward saturated actions.
- The tokenizer still has 4 to 5 times less latent resolution for velocity than a model without pretraining. The normalization is mean 4.0 and std 4.6 m/s, but online speeds are 0 to 3 m/s.
- An RL run from this checkpoint is expected to be better than entries 25-28, but it is not expected to match the runs without pretraining. There is no RL result for this data: entries 36-39 did not run.

## 4. Next data: `halfcheetah_v4_mix_scaled40_5p5m` (entry 44)

- 40 % expert_scaled episodes. The scale is uniform in [0.3, 0.8] and changes every 100 steps, with noise std 0.1.
- The other modes keep their ratios: expert 500, expert_noise 1000, random 650, past_runs 1150.
- Effect on the data:
  - The correlation of action size and speed goes from 0.712 to 0.661.
  - At 2-4 m/s, 7.6 % of the action values are saturated (was 22.5 %). At 4-6 m/s, 8.4 % (was 25.6 %).
  - Above 8 m/s, the data is still expert only (45-60 % saturated).
- Entries 45-54 (queue 9-30-26c) pretrain on it and run the same RL arms.
- Check the pretrains of entries 45-46 with the two probes before you read the RL results.

## 5. Step 12,000 of the scaled40 pretrains (entries 45 and 46), 2026-09-30 13:28

Data: `report/9-30-26-past-runs/data/probe_*_step12000.csv` (same online states as section 2).

| Model | Velocity slope (loops 45-54; gate > 0.7) | Bias (m/s) | Online gait: dream (true 1.11) | Bang-bang: dream (true 0.17) | Ranking correct? |
|---|---|---|---|---|---|
| Old pretrain, entry 24 | 0.30 | −1.31 | −0.19 | 0.53 | No |
| Entry 35 (5.5M data) | 0.55 | −0.46 | 0.48 | 0.76 | No |
| Entry 45 (scaled40, s42) | **0.77** | −0.50 | 0.65 | 0.74 | No (by 0.09) |
| Entry 46 (scaled40, s4) | 0.58 | −0.83 | 0.23 | −0.10 | **Yes** |
| No pretraining | 0.86 | +0.18 | 1.16 | 0.91 | Yes |

- Each scaled40 pretrain passes one gate and fails the other. The two seeds fail in different ways.
- Both are better than the old pretrain.
- Both still under-predict the online gait: speed −0.5 to −0.8 m/s, reward 0.23 to 0.65 against 1.11.
- The RL runs 47-50 test the result directly.
