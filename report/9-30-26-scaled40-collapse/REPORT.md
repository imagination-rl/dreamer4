# Collapse of the scaled40 RL runs (entries 47, 48), 2026-09-30

Runs: `offline_pretrained_a2_scaled40_rehearsal_s42` (47) and `_s4` (48). They start from pretrains 45 and 46 (data `halfcheetah_v4_mix_scaled40_5p5m`) and use 0.5 rehearsal.
Both runs were stopped by request at 13:45, at loops 60 and 55.
Scripts and data: `report/9-30-26-scaled40-collapse/scripts`, `.../data`.

## 1. What collapsed: the flipped gait

| Loops | 47 (s42): flipped / return | 48 (s4): flipped / return | Old pretrain, rehearsal 0.5 (25 / 26): flipped | No pretraining (repro s42 / s4): flipped |
|---|---|---|---|---|
| 10-19 | 22 % / −62 | 17 % / **+526** | 0 % / 0 % | 0 % / 1 % |
| 25-34 | 72 % / −140 | 57 % / +187 | 0 % / 0 % | 15 % / 4 % |
| 40-49 | 90 % / −195 | 80 % / −59 | 0 % / 1 % | 46 % / 2 % |

- Seed 4 learned faster than all reference runs at first (526 at loops 10-19). Then it flipped.
- In the dreams, the return stayed high: 7-21 per 32 steps. The closed-loop dream reward was 0.17-0.41 per step above the true reward. For the reference runs, the gap is −0.23 to +0.2.
- Once flipped, the cheetah does not recover in the episode. The world model predicts flipped states correctly (dream −0.16 to −0.25, true −0.23 to −0.25 per step).

## 2. Mechanism: exploitation of an optimistic model that does not see flips

Flip probe (`probe_flip_dreams.py`, loop-50 checkpoints, 1024 windows of 32 steps from each run's own replay, policy actions):

| Model, loop 50 | Upright starts: dream / true reward per step | Upright starts: true flip / dream flip | Segments that really flip: dream / true |
|---|---|---|---|
| 47 (scaled40, s42) | 0.80 / 0.06 (dream above true in 99 %) | 9 % / 1.4 % | 0.80 / 0.40 |
| 48 (scaled40, s4) | 1.44 / 0.68 | **57 % / 3.4 %** | **1.72 / 0.80** |
| Old pretrain, rehearsal s42 | −0.06 / −0.08 | 0.1 % / 0 % | — |
| Old pretrain, rehearsal s4 | 0.30 / 0.42 | 2.7 % / 0 % | 0.05 / 0.27 |
| No pretraining, repro s4 | 1.10 / 0.90 | 0.6 % / 0.4 % | — |

- The new world models are optimistic about the upright behavior of their own policy: 0.74 to 0.76 per step above the truth. The old pretrains were a little pessimistic.
- The policy's actions flip the cheetah in reality, and the model does not predict these flips (the known flip blind spot). The segments that cause a flip get the highest dream reward.
- PPO in imagination therefore moves the policy toward flip-causing actions. After the flip, the policy stays in the flipped gait.

## 3. Checked and not the cause

- **The data does not reward flips.** In every mode of every dataset, flipped states have a reward of about −0.2 per step. The new data has fewer flipped steps (4.5 %) than the old 1M data (6.0 %). (`flipped_data.csv`)
- **The policies do not copy the expert.** The cosine similarity of the policy action to the expert action at the same upright state is 0.05-0.06. It is 0.02-0.20 for the old pretrains and the runs without pretraining. The size ratio is about 0.6 for all runs. (`expert_likeness.csv`)
- **The reward head does not overfit** on the offline data: the train and validation losses are equal.

## 4. Why the new pretrain makes this worse (inference)

- **The data now links moderate actions to speed.** 40 % of the data is expert_scaled: moderate actions at 1.3-6 m/s, and none of those 2200 episodes flips.
  - The model learned that moderate actions give forward speed and do not flip the cheetah.
  - This is true only on the expert's closed-loop states. The data has no examples of the same actions from other states.
  - This is the fix of the old confound (large actions ⇒ fast), taken too far.
- **The optimism speeds up early learning.** Seed 4 was the fastest of all runs at loops 10-19. It then reaches aggressive gaits where flips happen, earlier than the other runs.
- **Rehearsal 0.5 keeps the effect.** Half of each world-model batch is offline data without flips, so the flips of the policy get half the weight in the world-model updates.
- **Confidence:** medium. There are two seeds and one arm; the no-rehearsal arm (49/50) did not run.

## 5. Suggestions

1. **Counterfactual data.** From online and past-run states, roll out changed actions: scaled, noisy, the expert's actions, or bang-bang. The model then sees what these actions do from states that are not expert states, including the flips.
2. **Flip transitions.** Add upright→flipped transitions from running gaits. Past runs contain 493 of them in the scaled40 data, mostly at low speed.
3. **Run the no-rehearsal arm (49/50)** before a new dataset. The old no-rehearsal arm did better than the old rehearsal arm.
4. **Stop rule for runs.** Stop a run when the closed-loop reward gap stays above 0.3 per step, or the flipped fraction stays above 0.3, for 5 loops. Both signals were clear by loops 25-34.

## 6. The no-rehearsal arm (entries 49, 50), 2026-09-30 14:40

Same pretrains as entries 47 and 48 (scaled40, step 12,000), with no rehearsal. Entry 50 ran alone after an OOM at startup.
Data: `data/loops.csv` (the script `collapse_logs.py` now includes these runs and the H100 runs).

| Run | Return, loops 95-99 | Best 5-loop mean | First loop with 5-loop mean ≥ 1,000 / ≥ 1,500 | Flipped, last 10 loops |
|---|---|---|---|---|
| **50: scaled40, no rehearsal, s4** | **1,925** | **2,168** | **38 / 52** | 18 % |
| 49: scaled40, no rehearsal, s42 | 851 | 1,089 | 49 / — | 38 % |
| 28: old pretrain, no rehearsal, s4 | 1,049 | 1,057 | 87 / — | 1 % |
| 27: old pretrain, no rehearsal, s42 | 317 | 1,456 | 42 / — | 54 % |
| 22: no pretraining (local repro), s4 | 1,194 | 1,411 | 47 / — | 0 % |
| 21: no pretraining (local repro), s42 | 112 | 912 | — / — | 93 % |
| H100 A2, s4 (100 of 500 loops) | 1,755 | 1,755 | 58 / 75 | 4 % |
| H100 A2, s42 | 2,153 | 2,153 | 49 / 65 | 17 % |

- Entry 50 is the best local run so far.
  - It reached 1,000 at loop 38 and 1,500 at loop 52; the H100 s4 run needed loops 58 and 75.
  - At loops 95-99 it is 1.6 times the matched run without pretraining (1,925 against 1,194).
  - It also flipped at loops 25-34 (39 %), then recovered.
- Entry 49 (s42) is better than the other local s42 runs at the end (851 against 112 and 317). It peaked at 1,089 near loop 45, and its flipped fraction then rose to 30-44 %. Seed 42 flips in every local arm.
- Averaged over both seeds, the result at loops 95-99 is 1,388 against 653 for the local runs without pretraining and 683 for the old no-rehearsal arm. This is one run per seed.
- Rehearsal is the difference between collapse and the best result: entries 47 and 48 used the same pretrains with 0.5 rehearsal.
- Warning signs in entry 50 at loops 90-99:
  - The action saturation rises to 21 %; runs without pretraining stay at 1-5 %.
  - The policy entropy is the lowest of all runs (−6.5).
  - The dream is above the true reward by 0.45 per step.
  - The flipped fraction rises from 8 % to 18 %.
  - The world model's one-step velocity error is 0.49 m/s; the runs without pretraining have 0.14-0.21. A longer run can still collapse.
