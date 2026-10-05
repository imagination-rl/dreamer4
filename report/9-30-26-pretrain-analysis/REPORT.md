# Pretrained RL runs against online-only runs (2026-09-30)

Scope: all 16 local RL runs that start from an offline pretrain, and 7 online-only A2 runs (same model, same RL settings). All numbers are for loops 0-99, because the local runs stop at loop 100. The H100 runs are cut at loop 99.
Scripts: `report/9-30-26-pretrain-analysis/scripts/`. Data: `.../data/` (`runs.csv`, `loops.csv`, `pretrain_curves.csv`, `data_mix.csv`, `candidate_replays.csv`).

## 0. Short answers

1. **Signs of a good run.**
   - At loops 20-39, a good run has a world-model velocity error near 0.3 m/s and a dream-to-true return correlation of 0.65 or more.
   - Its policy entropy goes down slowly (about −1.8), and its actions are rarely saturated (about 3 %).
   - Early flips do not predict failure: the good runs flip and then recover.
   - In the pretrained runs, the two strongest factors are "no rehearsal" and a pretrain that rates bang-bang actions below the online gait. Only one pretrain (entry 46) passed that gate, and it gave the best run.
   - To replicate: run a crossover of pretrains and RL seeds (section 6). One run for each arm cannot separate a real effect from run-to-run noise here.
2. **Data mix.**
   - Yes, remove the expert mode. Also remove the expert + noise episodes with std 0.1 and 0.2, because they are almost expert data.
   - More past runs helps only if you change *which* runs you use. The current pool is slow (mean 0.65 m/s). The local online-only runs and entry 50 have thousands of episodes at 1.7-2.5 m/s with moderate actions, plus thousands of real flips from a running gait. These are the two things that the current data does not have.
3. **Larger model with more data.** This is possible, but it is not the first thing to change.
   - The transformer is too small for the offline data: it does not overfit after about 56 passes over the 5.5M data, and its loss still goes down.
   - But the failures in RL come from the frozen tokenizer's velocity scale and from gaps in data coverage. A larger model does not fix those.
   - If you scale, grow the transformer (model_dim 256) and remove the terminal head, which holds 61 % of the parameters and does nothing on HalfCheetah.
4. **Learning rate.**
   - Do not increase the RL learning rates. The pretrained policies already change too fast: their entropy and saturation move faster than in the online-only runs.
   - For pretraining, the peak lr of 6e-4 is not shown to be the limit. Add a decay schedule and train longer on the 5.5M data. A short lr sweep would answer the rest.

## 1. The runs

Return = mean episode return. "Best 5" = best mean over 5 consecutive loops. "Loop 500/1000/1500" = first loop at which the 5-loop mean reaches that value.

| Run | Pretrain (data) | Rehearsal | Loops 95-99 | Best 5 | Mean 0-99 | Loop 500 / 1000 / 1500 | Flipped, loops 80-99 |
|---|---|---|---|---|---|---|---|
| H100 s42 | — | — | 2,153 | 2,153 | 969 | 32 / 49 / 65 | 21 % |
| H100 s4 | — | — | 1,755 | 1,755 | 840 | 34 / 58 / 75 | 5 % |
| local 9-28 s42 | — | — | 1,616 | 1,618 | 833 | 35 / 52 / 80 | 5 % |
| local 9-28 s4 | — | — | **2,656** | **2,666** | 1,136 | 33 / 53 / 61 | 3 % |
| local 9-28 s1 | — | — | 1,519 | 1,519 | 824 | 32 / 49 / 98 | 0 % |
| repro s42 (21) | — | — | 112 | 912 | 296 | 33 / — / — | 93 % |
| repro s4 (22) | — | — | 1,194 | 1,411 | 826 | 31 / 47 / — | 1 % |
| 17 | 1m_a (1M, s42) | 0.5 | 1,050 | 1,050 | 401 | 52 / 96 / — | 1 % |
| 18 | 1m_a | 0.5 | 1,082 | 1,136 | 517 | 44 / 74 / — | 1 % |
| 19 (log ends at loop 81) | 1m_a | 0 | — | 1,305 | 652 | 23 / 32 / — | 48 % |
| 20 (log ends at loop 77) | 1m_a | 0 | — | 1,469 | 522 | 26 / 32 / — | 85 % at loops 70-77 |
| 25 | 1m_b (1M, per seed) | 0.5 | 498 | 498 | 87 | — | 0 % |
| 26 | 1m_b | 0.5 | 18 | 513 | 195 | 48 / — / — | 56 % |
| 27 | 1m_b | 0 | 317 | 1,456 | 722 | 33 / 42 / — | 37 % |
| 28 | 1m_b | 0 | 1,049 | 1,057 | 396 | 62 / 87 / — | 2 % |
| 29 (entropy 0.01) | 1m_b | 0.25 | 783 | 787 | 244 | 64 / — / — | 1 % |
| 30 (entropy 0.01) | 1m_b | 0.25 | −109 | 640 | 85 | 30 / — / — | 88 % |
| 31 (BC) | 1m_b | 0 | −12 | −12 | −115 | — | 0 % |
| 32 (BC) | 1m_b | 0 | 187 | 187 | −32 | — | 1 % |
| 47 (stopped at 60) | scaled40 | 0.5 | — | −46 | −148 | — | 90 % at loops 50-59 |
| 48 (stopped at 55) | scaled40 | 0.5 | — | 589 | 151 | 15 / — / — | 77 % at loops 50-55 |
| 49 | scaled40 | 0 | 851 | 1,089 | 592 | 21 / 49 / — | 36 % |
| **50** | scaled40 | 0 | 1,925 | 2,168 | **1,348** | **14 / 38 / 52** | 13 % |

### 1.1 Run-to-run noise is very large

- The repro runs and the local 9-28 runs use the same config and the same seeds. They differ only in the world-model micro-batch (32 against 64) and the env mode. The env mode gives bitwise-equal data.
- Results with the same seed: s42 gives 112 against 1,616, and s4 gives 1,194 against 2,656.
- **One run cannot resolve a difference of less than about 1,000 at loop 100.** The mean over loops 0-99 is less noisy, but it still goes from 296 to 1,136 across the 7 online-only runs.

### 1.2 What pretraining gives and what it does not give

- **A head start (consistent).**
  - 5 of the 6 no-rehearsal pretrained runs reach 500 at loops 14-33. All 7 online-only runs reach 500 at loops 31-35.
  - 5 of the 6 reach 1,000 at loops 32-49. The online-only runs reach 1,000 at loops 47-58.
  - This is a gain of 10-20 loops.
- **No better final result (so far).**
  - The best pretrained arm (scaled40, no rehearsal) gets 1,388 on average at loops 95-99. The 7 online-only runs get 1,572.
  - For the mean over loops 0-99, the arm gets 970 against 818: 19 % better. Entry 50 alone accounts for the gain.
- **A plateau or collapse after loop 50.** 4 of the 6 no-rehearsal pretrained runs peak between loops 30 and 80, then flip or stop improving (entries 19, 20, 27, 49). Only entry 50 keeps improving.

## 2. Question 1: signs of the better runs

### 2.1 Early metrics by outcome class

- Class A: 1,500 or more at loop 100 (the 5 good online-only runs and entry 50).
- Class B: peaked at 1,000 or more, but did not reach 1,500 (8 runs).
- Class C: never reached 1,000 (9 runs).

Median over the runs of each class, loops 20-39 (min and max of each class in `data/`):

| Metric | A | B | C | What it means |
|---|---|---|---|---|
| World-model one-step velocity error (m/s) | **0.31** | 0.63 | 0.57 | The model sees speed changes. |
| Dream-to-true return correlation (closed loop) | **0.69** (min 0.65) | 0.57 | 0.51 | The dreams rank the policy's segments correctly. |
| Policy entropy | **−1.8** | −3.0 | −2.0 | A slow decrease keeps exploration. B runs become deterministic early. |
| Action saturation | **3 %** | 7 % | 4 % | The policy stays in the moderate-action gait. |
| Dream gap (dream − true reward per step) | +0.11 | +0.05 | +0.03 | Small optimism is normal in good runs. |
| Flipped fraction | 13 % | 3 % | 2 % | Not a warning sign by itself. Good runs flip, then recover. |

Later, at loops 40-59, the same metrics separate the classes, and class A also has the highest return correlation (0.83).

### 2.2 In the pretrained runs only (16 runs, Spearman correlation)

| Factor | Correlation with mean return 0-99 | Correlation with return at loops 10-19 |
|---|---|---|
| Action-ranking agreement with MuJoCo, loops 5-24 (`ranking_spearman`) | **+0.65** | +0.45 |
| Rehearsal fraction | **−0.40** | −0.17 |
| Dream-to-true return correlation, loops 5-24 | +0.24 | **+0.66** |
| Dream gap, loops 5-24 | −0.02 | **+0.73** |

- A world model that is a little optimistic and correct in its ranking gives a fast start (entries 19, 20, 48, 50). The final result depends on what happens after loop 40.
- Rehearsal hurts in every data set. The best result with 0.5 rehearsal is 1,136, and 0.5 rehearsal on the scaled40 pretrains collapsed.
- BC in the world-model updates (entries 31/32) freezes the policy: entropy stays near −0.7 to −1.4, and the return stays near 0.

### 2.3 The pretrain gates against the RL result

| Pretrain | Velocity slope (gate > 0.7) | Bang-bang below online gait? | Best no-rehearsal RL result |
|---|---|---|---|
| 1m_b s42 (entry 23) | 0.08 | No (−0.09 > −0.56) | 317 (peak 1,456, then flipped) |
| 1m_b s4 (entry 24) | 0.30 | No | 1,049 (slow) |
| scaled40 s42 (entry 45) | **0.77** | No (0.74 > 0.65) | 851 (peak 1,089) |
| scaled40 s4 (entry 46) | 0.58 | **Yes** (−0.10 < 0.23) | **1,925** |

- The ranking gate agrees with the result. The slope gate alone does not.
- This comes from 4 pretrains, and each pretrain has one RL seed. The pretrain seed and the RL seed are the same, so the effect of the pretrain and the effect of the RL seed cannot be separated yet.

### 2.4 Why the pretrained runs plateau: the frozen tokenizer

- In every pretrained run, the world-model velocity error stays at 0.33-1.05 m/s for all 100 loops. In the online-only runs, it falls to 0.14-0.21 m/s by loops 90-99. The exception is repro s42 (0.37), which flipped (`loops.csv`).
- Both paths freeze the tokenizer after its first training (`train/train_gym.py:3653-3655`).
  - The pretrained tokenizer normalizes the forward velocity with std 3.9-5.1 m/s. The online tokenizer uses std 0.74 m/s.
  - So 0.1 m/s moves the latent 0.028-0.036 in the pretrained tokenizer and 0.136-0.164 in the online tokenizer (4-5 times more).
  - For the same velocity precision, the pretrained world model must predict latents 4-5 times more precisely. The measured velocity error is 2-4 times worse.
- Then PPO exploits the error:
  - At loops 90-99, the entropy of most pretrained runs is −5.7 to −8.4 (entries 30 and 49: −3.5 and −4.6). For the online-only runs, it is −4.7 to −5.3.
  - The saturation of the pretrained runs is 5-29 %. For the online-only runs, it is 1-5 %; repro s42, which flipped, reaches 13 %.
  - The dream gap is 0.35-0.48 per step, and the flip rate goes up.
- Entry 50 has the same problem (velocity error 0.49, saturation 21 %, entropy −6.5). A longer run can still collapse.

### 2.5 How to replicate entry 50

1. Pretrain on the scaled40 data (or better data, section 3). Use step 12,000.
2. Gate the pretrain with `probe_action_sources.py`: bang-bang must rank below the online gait.
3. Run RL with no rehearsal.
4. Stop a run when the flipped fraction stays above 0.3 **and** the dream gap stays above 0.3 per step for 10 loops. Entry 50 had 37 % flips at loops 30-39 with a small gap (0.07), and it recovered.
5. First, confirm that the result repeats (section 6, step 1).

## 3. Question 2: the data distribution

### 3.1 Remove modes from the scaled40 data (`data_mix.py`, no new collection)

| Variant | Episodes | Corr(|a|, speed) | Velocity std (tokenizer scale) | Velocity resolution vs now | Steps at 0-3 m/s | Steps above 8 m/s |
|---|---|---|---|---|---|---|
| Current scaled40 | 5,500 | 0.661 | 3.91 m/s | 1.00 | 42 % | 18 % |
| No expert | 5,000 | 0.560 | 3.22 | 1.21 | 46 % | 10 % |
| **No expert, no expert + noise std ≤ 0.2** | 4,495 | **0.327** | **2.30** | **1.70** | 51 % | 1 % |
| No expert, no expert_scaled | 2,800 | 0.680 | 3.95 | 0.99 | 49 % | 17 % |

By mode:

| Mode | Mean speed | Corr(|a|, speed) | Upright → flipped |
|---|---|---|---|
| expert | 11.2 m/s (94 % above 8 m/s) | −0.02 | 0 |
| expert + noise | 7.2 | 0.30 | 129 |
| random | −0.1 | 0.00 | 772 |
| past_runs | **0.65** | 0.18 | 493 |
| expert_scaled | 3.5 | **0.66** | **0** |

- **The expert mode.**
  - It covers only 10-12 m/s. The online policy runs at 1.5-2.7 m/s at loop 100.
  - It still sets a large part of the tokenizer scale and of the action-speed confound.
  - Remove it, and remove expert + noise with std 0.1 and 0.2: those episodes are at 10.6 m/s.
  - Keep std 0.4 and 0.8: they give moderate speeds and some flips. Set `offline_expert_noise_stds: [0.4, 0.8]`.
- **The expert_scaled mode.**
  - It is needed for moderate actions at speed. Without it, the confound goes back up (0.680).
  - But it has the confound built in (a larger scale gives a higher speed), and it has **no flips** in 2,200 episodes.
  - This is the likely source of the optimism of entries 45-48 (`report/9-30-26-scaled40-collapse/REPORT.md`, section 4). Reduce it to about 20 %.

### 3.2 Which past runs to use (`candidate_replays.py`)

The current pool (12 older pretrained runs) has only 1,643 episodes with a return of 1,500 or more, from 3 runs. Only 11 of those are above 2,500. The other replays on disk have much more:

| Replay | Episodes ≥ 1,500 | Speed of those episodes | Mean |a| (saturated) | Upright → flipped |
|---|---|---|---|
| local 9-28 attnpool_on s4 | 2,549 (1,121 ≥ 2,500) | 2.48 m/s | 0.45 (1 %) | 2,775 |
| local 9-28 attnpool_on s42 | 1,510 | 1.75 | 0.36 (0.1 %) | 3,740 |
| local 9-28 attnres_full s42 / s4 | 1,807 / 1,137 | 1.8-2.0 | 0.36-0.41 | 6,138 / 6,496 |
| entry 50 (scaled40, no rehearsal, s4) | 3,222 | 2.41 | 0.60 (10 %) | 4,826 |
| entry 49 | 114 | 1.86 | 0.56 (5 %) | 7,796 |
| 8 other local 9-28 runs | 0-953 each | 1.6-2.0 | 0.36-0.45 | 663-14,158 |
| Entries 47/48 (collapsed) | 0 | — | — | 7,587 / 6,727 |

- These episodes are exactly what the data lacks:
  - the moderate Beta-policy actions at 1.7-2.5 m/s;
  - upright → flipped transitions from a running gait (the flip blind spot).
- **Use them, with three conditions.**
  - Keep the repro replays (entries 21/22) out, because the probes use their states.
  - Add a third return bucket [1500, 3000). Collapsed runs (47/48) give flips from the flipped basin; the good runs give flips from running gaits.
  - These replays cost 64 × 100 episodes of env steps each. Report this cost in the comparison with the online-only runs, because the data comes from RL on the same task.

### 3.3 Proposed mix (5,500 episodes)

- past_runs 2,750 (50 %): three equal buckets [−150, 500), [500, 1,500), [1,500, 3,000). Use the 12 current replays, plus the local 9-28 replays, plus entries 47-50.
- expert + noise 1,000, std 0.4 and 0.8 only.
- random 650.
- expert_scaled 1,100 (20 %).
- expert 0.

Run `data_mix.py` on the new dataset before you pretrain. The targets are a correlation below 0.4 and a velocity std near 2 m/s.

**Confidence: medium.** The change removes the two measured causes: the action-speed confound and the missing flip transitions. But the past-run data also shows the world model more of the "moderate action ⇒ speed" pattern, and that pattern gave optimism in entries 47/48. The real flips in the same data are the counterweight. Only an RL run can show which effect is stronger.

## 4. Question 3: a larger model with more data

**Evidence that capacity limits the fit to the offline data:**

- On the 5.5M data, the train and validation losses are equal at every checkpoint (`pretrain_curves.csv`).
  - 12,000 steps × 128 windows × 200 steps is about 56 passes over each transition. A model that does not overfit after 56 passes cannot memorize the data.
  - The flow loss still goes down by about 2 % for each 1,000 steps at step 12,000 (scaled40 s42: 0.0106 → 0.0101 from step 8k to 11k).
- On the 1M data, the reward loss overfit after step 4,000. So a larger model needs the 5.5M data or more, not the 1M data.
- The transformer holds only 1.01M of the 8.80M parameters. The terminal head holds 5.34M (61 %), and HalfCheetah never terminates.

**Evidence that capacity is not the main limit of the RL results:**

- The online-only runs use the same model. Their velocity error falls to 0.14-0.21 m/s, and they reach 1,500-2,650 by loop 100. The model is large enough for the online distribution at this stage.
- The pretrained runs fail for two reasons. Both are independent of size:
  - The frozen tokenizer scale gives 4-5 times less velocity resolution (section 2.4).
  - The data coverage is incomplete: no flips from running gaits, and no moderate actions from off-expert states. The online-only models and the H100 models also have the flip blind spot.

**About "larger models learn faster":**

- In LLMs, and in DreamerV3's scaling results, a larger model reaches a given loss with fewer samples. That applies to the loss on the training distribution.
- Here, the problem is off-distribution: the policy's own new actions, for which the data has no examples. Scale can help extrapolation, but it does not replace coverage.
- Cost: model_dim 256 makes the transformer about 4 times larger (about 4M). The world-model step takes about 2-3 times longer, so a local RL loop goes from about 13 s to about 25-35 s (estimate, not measured).

**Verdict:** a larger transformer is a second step, not the first. Probability that it helps the RL result by itself: about 30 %. First fix the data (section 3) and the velocity resolution. Then test capacity with a pretrain-only check before any RL: model_dim 256 with `old_terminal_head`, 12,000 steps on the same data. Compare the validation loss at equal steps and the two probes.

## 5. Question 4: the hyperparameters

| Setting | Value | Evidence | Recommendation |
|---|---|---|---|
| Pretrain world-model lr | 6e-4, constant, AdamW, wd 0.01 | Loss still goes down at step 12,000; no overfit | Keep the peak. Add a decay (for example cosine to 10 %) and train 20,000-30,000 steps on the 5.5M data. The 12,000-step stop was chosen for the 1M data (overfit after step 4,000). |
| Grad clip | 1.0; grad norm is 4-11 | Clipped on every step | With Adam, the clip changes the step size little (Adam divides by the gradient scale). No change needed. Log the clip fraction if you do an lr sweep. |
| Tokenizer | 2,000 steps × 1,024; recon loss 2-3e-4 | Reconstruction is fine | The problem is the normalization scale, not the training (section 2.4). |
| Agent lr (RL) | 3e-4, linear decay over 500 loops (80 % at loop 100) | Pretrained policies lose entropy faster (−5.7 to −8.4 against −4.7 to −5.3 at loops 90-99) and saturate more (5-29 % against 1-5 %) | **Do not increase.** For pretrained runs, test an entropy bonus with no rehearsal. Entries 29/30 tested it only with 0.25 rehearsal, so that result is confounded. |
| World-model lr (RL) | 6e-4 constant | The pretrained velocity error does not improve in 100 loops | A higher lr does not fix a frozen tokenizer scale. No change. |
| Rehearsal | 0.5 / 0.25 / 0 | 0 is best in every data set | Use 0. |
| Pretrain checkpoint | step 12,000 | Chosen for the 1M data | Retest on the new data, with the decay schedule. |

- There is no lr sweep for the pretrain yet. A sweep of 3e-4, 6e-4, and 1.2e-3 for 4,000 steps each takes about 20-30 min for each run when the run is alone on the GPU. It gives the answer directly.
- The decay schedule needs a new flag, because the default path must not change.

## 6. Recommended next runs (in order)

1. **Replicate entry 50 with a crossover** (4 RL runs, about 1.5 h with two runs at a time and a 200 s start gap):
   - pretrain 46 with RL seeds 42 and 1;
   - pretrain 45 with RL seeds 4 and 1.
   - This separates "good pretrain" from "good RL seed".
2. **New dataset** (section 3.3): collect it, pretrain two seeds (about 1 h alone each), gate the pretrains with the probes, then run RL with no rehearsal.
3. **Pretrain schedule:** add an lr-decay flag, run a short lr sweep, and train 20,000-30,000 steps on the new data.
4. **Capacity check:** model_dim 256 + `old_terminal_head`, pretrain only, then the probes. Run RL only if the gates improve.
5. **Velocity resolution:** try a tokenizer normalization fit on the speeds that the online policy uses. This needs a new flag.

## 7. Limits

- There is one run for each arm and seed, and run-to-run noise is about ±1,000 at loop 100 (section 1.1). All arm comparisons are indicative.
- Entries 19 and 20 end at loops 81 and 77. Entries 47 and 48 were stopped at loops 60 and 55.
- The H100 runs have 500 loops. This report uses only loops 0-99.
- The capacity estimates in section 4 are not measured.
