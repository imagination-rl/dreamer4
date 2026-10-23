# Data Collection + Pretraining Runs and Flip-Onset Ablations (2026-10-04)

## 1. Summary

- **Data source:** the replays of the 18 stage-1 runs (entries 118-135, 175 loops, 11,200 episodes each). They are the first local runs with fast data (2.4-4.3 m/s at loops 170-174) from the policy itself.
- **New dataset:** `halfcheetah_v4_own_runs_6m`, 6,000 episodes, all past runs, no expert data. It is collected (2026-10-04 21:33, CPU, from the replays only).
- **New runs (queue 10-5-26, 12 runs, 175 loops, seeds 4, 42 and 1):**

| Entries | Run names | Change | Compare with |
| --- | --- | --- | --- |
| 166-168 | `a2_grouped_combo_symexp_ownrunspre_s*` | collect + pretrain (12,000 steps) on the new data, then RL | 163-165 (scaled40 data, the same settings), 127-129 |
| 169-171 | `a2_grouped_combo_symexp_warmup15_flipwm25_s*` | world model: 25 % of the windows hold a real flip onset | 127-129 |
| 172-174 | `a2_grouped_combo_symexp_warmup15_flipprompt25_s*` | actor and critic: 25 % of the dreams start 1-32 steps before a real flip | 127-129 |
| 175-177 | `a2_grouped_combo_symexp_ownrunspre_flipwm25_s*` | 166-168 + flipwm25 in the pretrain and in RL | 166-168 |

- The queue `scripts/queue_10-5-26.sh` waits for the queue 10-4-26 (PID 995617) to end. Record: `runs/queues/10-5-26_2026-10-04_214125`. Expected run time: about 12-13 hours, plus the GPU break.

## 2. Lessons of the Earlier Datasets

| Dataset | Result | Lesson |
| --- | --- | --- |
| `mix_1m` (expert, expert + noise, random) | slow learning or the flipped gait | Saturated actions occur only at high speed. The world model learned "large action means fast" and rated bang-bang actions above the online gait. |
| `past_runs_*` (old runs + expert) | the confound stays (corr 0.73) | The old past runs were slow (at most 1.6 m/s). Fast data came only from the expert. |
| `scaled40_5p5m` | best pretrained run (1,925, s4) and 851 (s42); rehearsal collapsed | Moderate actions at speed help. Rehearsal hurts. The frozen tokenizer compresses the velocity (vx std 3.9). |
| `realistic_7p5m` | −11 and 1,490 | Without scaled expert data, the confound returns above 3 m/s. A flip share of 0.45 added the flipped steady state, not flip onsets. |
| `bangbang_scaled_7p5m` | 115 and 810 | Bang-bang episodes are 86 % flipped. A low confound alone did not help. |

Other lessons: a pretrain gives a head start of 10-20 loops; the run-to-run noise is about ±1,000; one seed cannot decide an arm. No pretrained world model dreamed the flips of the real policy.

## 3. Choice of the Source Runs (`scripts/episode_stats.py`, `data/episodes.csv`)

Measures of all 201,600 episodes of the 18 replays. "corr" is the pooled correlation of the mean |a| of a step and the x velocity.

| Run | Return, loops 170-174 | vx | Saturated | Flip episodes, loops 140-174 | corr |
| --- | --- | --- | --- | --- | --- |
| symexp_warmup15 s4 / s42 / s1 | 3,829 / 3,290 / 3,921 | 4.1 / 3.6 / 4.3 | 16-21 % | 29 / 4 / 3 % | 0.51-0.59 |
| cevalue_warmup15 s4 / s42 / s1 | 3,355 / 3,737 / 2,878 | 3.6 / 4.0 / 3.1 | 8-14 % | 29 / 20 / 36 % | 0.37-0.43 |
| symlog_warmup15 s4 / s42 / s1 | 3,568 / 672 / 3,183 | 3.9 / 0.9 / 3.4 | 9-15 % | 24 / 96 / 0 % | 0.27-0.34 |
| rw4_block2 s4 / s42 / s1 | 2,917 / 1,416 / 1,504 | 3.1 / 1.6 / 1.7 | 4-5 % | 32 / 2 / 7 % | −0.20 to −0.03 |
| rw4_depth5 s4 / s42 / s1 | 2,454 / 2,190 / 2,664 | 2.4-2.8 | 3-4 % | 0-12 % | −0.04 to 0.13 |
| rw4_fullres s4 / s42 / s1 | 2,495 / 1,443 / 2,265 | 1.6-2.7 | 4-6 % | 7-41 % | −0.04 to 0.05 |

- The fast value runs give the high speeds. Their policies use larger actions at speed (corr 0.3-0.6).
- The rw4 runs give moderate actions at all speeds (corr near 0) and other gaits.
- Thus all 18 runs are in the data: speed range from the fast runs, action variety from the rw4 runs, six settings and three seeds.
- About half of the upright-to-flipped transitions recover within 10-18 steps. The flips of the fast episodes come late (median step 680-740), so they add onsets with few flipped steps.
- The replays of queue 10-4-26 are not in the data. The probes of the new pretrains can use their states as held-out data.

**Selection:** 5 equal return buckets ([−1000, 0), [0, 1000), [1000, 2000), [2000, 3000), [3000, 6000)), 1,200 episodes each, equal shares of the runs. In each bucket 30 % of the episodes have a flip onset. Data seed 20261005.

## 4. The New Dataset against the Earlier Datasets (`scripts/dataset_compare.py`, `data/dataset_compare.csv`)

| Dataset | corr | vx mean / std | vx p95 | Saturated | Flipped steps | Onsets |
| --- | --- | --- | --- | --- | --- | --- |
| mix_1m | 0.71 | 6.1 / 5.1 | 12.8 | 40 % | 6 % | 431 |
| scaled40_5p5m | 0.66 | 3.8 / 3.9 | 11.7 | 16 % | 4 % | 1,394 |
| realistic_7p5m | 0.55 | 1.7 / 2.7 | 8.7 | 12 % | 16 % | 7,614 |
| bangbang_scaled_7p5m | 0.12 | 1.8 / 2.7 | 8.0 | 24 % | 24 % | 6,396 |
| **own_runs_6m** | **0.34** | **1.7 / 1.7** | 4.5 | **4 %** | 11 % | 4,011 |

Saturated fraction for each speed bin (m/s):

| Dataset | 2-3 | 3-4 | 4-5 | 5-6 |
| --- | --- | --- | --- | --- |
| scaled40 | 7 % | 8 % | 8 % | 9 % |
| realistic | 11 % | 25 % | 42 % | 44 % |
| own_runs | 4 % | 7 % | 10 % | 13 % |

- The new data has the lowest saturation at all speeds. Its confound (0.34) is the confound of the online policies themselves, not that of an expert.
- Its velocity spread (std 1.7) is the smallest, so the frozen pretrained tokenizer keeps more velocity resolution.
- 24 % of its steps are at 3-5 m/s (realistic 6 %, scaled40 16 %). That is the speed range in which the stage-1 dreams became optimistic (+0.7-0.9 reward per step).
- Limit: no data above 6 m/s. A pretrained world model knows only the speeds of the stage-1 runs.

## 5. The Flip-Onset Flags (new, default off)

The instability audit (`report/10-4-26-instability-audit`) found:
- The late return drops are flips.
- 3-8 % of the dreams end flipped, against 9-41 % of the real rollouts.
- The policy is near uniform on the states where the body tips.
- About 5 % of the 200-step world-model windows of the late loops hold an onset.

| Flag | What it does | What it tests |
| --- | --- | --- |
| `world_model_flip_onset_window_fraction` | That share of the world-model windows holds a real onset at a random place. It applies in the online loop, in the loop-0 steps, and in the offline pretrain. | Does more onset data make the world model dream the flips? |
| `imagination_flip_onset_prompt_fraction` | That share of the 8-step prompts ends upright, 1-32 rows before a real onset. The real flip then falls in the dream steps. | Does training on tipping states teach the policy to avoid or recover from a flip? |

- Code: `train/train_gym.py` (`FlipOnsets`, `FlipOnsetTimeWindow`, `replay_time_windows`). The onset index reads only the new episodes of each loop.
- With the value 0.0, the code builds the same `ReplayDatasetTimeWindow` objects as before. The resolved settings of the 27 entries of queue 10-4-26 are unchanged (two new keys with the value 0.0).
- New test: `tests/test_flip_onset_windows.py`.
- Log: `extra/flip_onsets_in_replay` (and `offline_pretrain/train/extra/...`).
- CPU smoke test of entry 175 with both flags: collect, pretrain (34 onsets in the small offline replay), 2 RL loops: exit 0.

Other changes:
- `train/offline_pretraining.py`: the past_runs collector opens the replays read-only. `ReplayBuffer.from_folder` would create a zero `latents.data.npy` in each source run, because the queues deleted the latents.
- `tests/test_offline_pretraining.py` checks this.

## 6. How to Read the Results

- **166-168 against 163-165:** the same settings, only the data changes. Compare the return over loops 0-174, the loop at which the return passes 500, and the flipped fraction.
- **166-168 against 127-129:** does pretraining on the own runs give more than warmup15? Note: the data holds 3.4 M real steps of other runs, so a gain is not free.
- **169-171 and 172-174 against 127-129:** read the dream flip rate (dreams that end flipped) and the real flipped fraction, as well as the return. If flipwm25 works, the dream flip rate rises toward the real rate. If flipprompt25 works, the real flip rate falls.
- **175-177 against 166-168:** does the onset sampling make the pretrained world model learn the flips that the flip share alone did not teach?
- Before the RL results, gate each new pretrain with `report/9-30-26-pretrain-rootcause/scripts/probe_action_sources.py` (bang-bang must rank below the online gait) and `probe_latent_resolution.py`. Use states from the 10-4-26 replays.

## 7. Not in This Queue

- `preserve_policy_during_world_model_updates` (drift limit). Its consistency forward pass runs on the full batch (128 windows × 200 steps) with gradients, outside the micro-batches. It is likely out of memory at 48 % of the GPU. A batch-size flag for the consistency term is necessary first.
- A target-KL stop for the closer PPO. The lr150 sweep of queue 10-4-26 gives a first answer.
- Data above 6 m/s (scaled expert episodes), or the replays of queue 10-4-26 as a second data source.

## 8. Files

| File | Content |
| --- | --- |
| `scripts/episode_stats.py` | `data/episodes.csv`: statistics of each episode of the 18 replays |
| `scripts/collect_own_runs.py` | collects the data of entry 166 on the CPU |
| `scripts/dataset_compare.py` | `data/dataset_compare.csv`, `data/dataset_speed_bins.csv` |
| `logs/` | collection and test logs |
