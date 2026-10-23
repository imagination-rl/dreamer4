# Realistic-data runs (queue 9-30-26e, entries 55-61): root cause and next ablations

Date: 2026-10-01. Queue record: `runs/queues/9-30-26e_2026-09-30_205208` (all 7 entries exit 0, no OOM).

## 0. Summary

- The realistic-data arm is worse than the online-only runs. All 4 RL runs are below the online median at loops 95-99 (Mann-Whitney one-sided p = 0.036). For the mean of loops 0-99, p = 0.012.
- The pretrain decides the outcome, not the RL arm. Both RL runs from the s4 pretrain lock into the flipped gait at loops 10-29. Both RL runs from the s42 pretrain lose 20-40 loops in a flip phase, then recover in part.
- Root cause, part 1 (data): the mix has no fast data with moderate actions. Above 3 m/s, only expert + noise data remains, with 25-45 % saturated action dimensions. The scaled40 mix had 8 %. Thus the action-size/speed confound of `report/9-30-26-pretrain-rootcause` is back, in the speed band that the RL policy enters next.
- Root cause, part 2 (model): the s4 pretrain overrates aggressive actions at every saved checkpoint (bang-bang dream reward 0.9-1.9 per step above MuJoCo). At its final checkpoint it ranks bang-bang above the online gait. The s42 pretrain passes this gate at steps 2000, 4000, and 12000, but fails it at step 8000. Thus the gate result changes from checkpoint to checkpoint.
- Root cause, part 3 (flips): the flip share of 0.45 did not teach the world model when a flip starts. It added mostly the flipped steady state: the past runs hold about 1.0 M flipped steps but only 5,861 flip onsets. The s4 pretrain dreams 0 % flips on real flip segments. During RL, the dreams show only 20-50 % of the real flips, as in the online runs. The 32 steps before a real flip earn more true reward (0.54 vs 0.33 per step), so imagination gets no signal against the flip.
- What improved: velocity resolution. The tokenizer vx std falls from 3.9 to 2.75. The world-model vx error during RL is 0.31 m/s on average, against 0.48 for scaled40.
- Head re-anchor did nothing. The policy KL before the fit is 0.001-0.066 nats, so the heads do not drift in a world-model phase. The anchor and no-anchor results differ in opposite directions for the two seeds. That is run noise.
- Next ablations (section 7): first, a seed cross and a checkpoint-gate test that use the existing pretrains (6 RL runs, about 2 h). Then, a data fix that brings back fast data with moderate actions.

## 1. RL results

| run | entry | pretrain | ret 95-99 | mean 0-99 | loop to 500 | episodes with a flip, loops 10-29 | flipped steps, loops 80-99 |
|---|---|---|---|---|---|---|---|
| real_noreh_s42 | 58 | realistic s42 | 1,490 | 707 | 46 | 39 % | 2 % |
| real_noreh_s4 | 59 | realistic s4 | −11 | 14 | – | 90 % | 87 % |
| real_anchor_s42 | 60 | realistic s42 | 630 | 335 | 20 | 32 % | 4 % |
| real_anchor_s4 | 61 | realistic s4 | 474 | 118 | – | 83 % | 36 % |
| s40_noreh_s42 | 49 | scaled40 s42 | 851 | 592 | 21 | 6 % | 36 % |
| s40_noreh_s4 | 50 | scaled40 s4 | 1,925 | 1,348 | 14 | 58 % | 13 % |
| online (7 runs) | – | none | median 1,616 (112-2,656) | mean 820 | 31-35 | 15-51 % (3 local runs) | 0-93 % |

Arm means at loops 95-99: realistic 740, realistic + anchor 552, scaled40 1,388, online 1,572.

The pretrained head start is still present. At loops 10-19 the realistic runs return 205-351 per episode; the online runs return −30 to −119. The return then stalls. In three realistic runs the dream return stays at 14-25 for loops 20-99; the no-anchor s42 run reaches 30-33 at loops 80-99. In scaled40 s4 and online s4 it climbs to 52-55. The stall occurs together with the flip phase.

Data: `data/runs.csv`, `data/loops.csv` (`scripts/run_table.py`).

## 2. The flip timeline in the replays

From the replays, 64 episodes per loop (`scripts/flip_events.py`, `data/flip_blocks.csv`):

| run | loops | episodes with a flip | median step of the first flip | flipped steps | saturated before the flip |
|---|---|---|---|---|---|
| real_noreh_s4 | 10-19 | 80 % | 253 | 30 % | 3.3 % |
| real_noreh_s4 | 20-29 | 100 % | 126 | 58 % | 3.8 % |
| real_noreh_s4 | 30-69 | 100 % | 83-121 | 71-81 % | 4.2-4.7 % |
| real_anchor_s4 | 20-59 | 100 % | 79-148 | 51-76 % | 4.7-6.5 % |
| real_noreh_s42 | 30-39 | 98 % | 179 | 33 % | 3.9 % |
| real_noreh_s42 | 60-69 | 35 % | 536 | 4 % | 4.9 % |
| s40_noreh_s4 | 30-39 | 81 % | 304 | 37 % | 4.9 % |
| local928_s42 (online) | 20-29 | 71 % | 353 | 15 % | 1.2 % |
| local928_s4 (online) | 30-39 | 57 % | 362 | 20 % | 1.3 % |

- All runs go through a flip phase. The online runs reach it at loops 20-40. Their first flips stay late in the episode (step 350 or later), and the runs recover.
- The s4 realistic runs reach the flip phase at loops 10-19. Then the first flip moves to about step 100, so the replay fills with flipped states. The no-anchor run never recovers. The anchor run recovers in part after loop 70 (37 % flipped steps at loops 90-99).
- Before a flip, the realistic runs saturate 3-4× more action dimensions than the online runs (3.3-6.5 % vs 0.7-2.6 %). This agrees with an optimistic world model for aggressive actions (section 4).

## 3. The dataset

`scripts/dataset_compare.py`, `data/dataset_compare.csv`, `data/dataset_speed_bins.csv`.

| speed bin (m/s) | scaled40: share of steps | scaled40: saturated | realistic: share of steps | realistic: saturated | realistic: flipped |
|---|---|---|---|---|---|
| < 0 | 11 % | 5 % | 24 % | 4 % | 34 % |
| 0-1 | 17 % | 5 % | 30 % | 4 % | 25 % |
| 1-2 | 15 % | 5 % | 18 % | 6 % | 5 % |
| 2-3 | 10 % | 7 % | 11 % | 11 % | 0 % |
| 3-4 | 8 % | 8 % | 4 % | 25 % | 0 % |
| 4-6 | 14 % | 8 % | 3 % | 43 % | 0 % |
| ≥ 6 | 25 % | 46 % | 10 % | 44 % | 0 % |

- Speed: 53 % of the realistic steps are below 1 m/s. The band of 3-6 m/s holds only 7 % of the steps, against 22 % in scaled40.
- Confound: in scaled40, the expert_scaled mode filled 2-6 m/s with moderate actions (0.4 % saturated). In the realistic mix, the only fast data is expert + noise (45 % saturated). The past runs have moderate actions but are slow (mean 0.85 m/s, p90 2.4 m/s). Thus "saturated means fast" is again true in the data at 3-6 m/s.
- Flips: the realistic mix has 7,607 upright-to-flipped transitions (scaled40: 1,394) and 16 % flipped steps (scaled40: 4 %). But a transition is only 0.1 % of all transitions. Most flipped steps come from the flipped steady state of the past runs: each selected flip episode stays flipped for the rest of its 1,000 steps. Flipped steps get −0.21 reward per step, the same as in scaled40.

## 4. The pretrains

### 4.1 Losses

`scripts/pretrain_curves.py`, `data/pretrain_curves.csv`. At step 10,000 the validation flow loss is 0.0104 (s42) and 0.0085 (s4). For scaled40 it is 0.0103-0.0111. Train and validation are equal, so the models do not overfit. The reward MAE is 0.40, higher than for scaled40 (0.36), because the realistic data has more slow and random steps. The s4 pretrain has the lowest flow loss of all four, but it gives the worst RL. The losses do not predict the RL result.

### 4.2 Velocity resolution (improved)

`report/9-30-26-pretrain-rootcause/scripts/probe_latent_resolution.py` on online states (`data/probe_latent_resolution.csv`):

| model | tokenizer vx std | latent change per 0.1 m/s | one-step vx error, loops 45-54 | vx slope |
|---|---|---|---|---|
| realistic s42 / s4 | 2.75 | 0.047 / 0.048 | 0.34 / 0.25 m/s | 0.76 / 0.79 |
| scaled40 s42 / s4 | 3.91 | 0.035 / 0.034 | 0.56 / 0.88 m/s | 0.76 / 0.62 |
| online-only (repro s4 loop 50) | 0.74 | 0.136 | 0.24 m/s | 0.87 |

The realistic pretrains predict online velocity almost as well as an online-only world model. The RL logs agree: the vx error is 0.20-0.44 m/s over the 100 loops (mean 0.31; scaled40 0.48; online 0.16-0.29). Velocity compression is not the cause of these failures.

### 4.3 Action ranking (the gate)

`report/9-30-26-pretrain-rootcause/scripts/probe_action_sources.py` (`data/probe_action_sources.csv`, `data/probe_action_sources_steps.csv`). Same online start states, 32 steps open loop. "bb − online" is the dream reward of bang-bang actions minus that of the online gait (the gate fails when it is above 0). "bb gap" is the dream minus the MuJoCo reward of bang-bang (true: 0.17 per step).

| pretrain checkpoint | bb − online (dream) | bb gap | RL ret 95-99 (no rehearsal) |
|---|---|---|---|
| realistic s4, step 2,000 | +0.75 | +1.88 | – |
| realistic s4, step 4,000 | +0.18 | +0.91 | – |
| realistic s4, step 8,000 | −0.06 | +0.90 | – |
| realistic s4, step 12,000 (used) | **+0.06** | **+1.09** | −11; 474 (anchor) |
| realistic s42, step 2,000 | −0.21 | +0.05 | – |
| realistic s42, step 4,000 | −0.46 | +0.20 | – |
| realistic s42, step 8,000 | **+0.22** | **+1.26** | – |
| realistic s42, step 12,000 (used) | −0.36 | +0.33 | 1,490; 630 (anchor) |
| scaled40 s4, step 12,000 | −0.33 | −0.27 | 1,925 |
| scaled40 s42, step 12,000 | +0.10 | +0.58 | 851 |
| 1M s4, step 12,000 | +0.72 | +0.37 | 1,049 |
| 1M s42, step 12,000 | +0.47 | −0.26 | 317 |

- Pretrains that fail the gate gave 5 runs: 317, 1,049, 851, −11, and 474 (mean 536). Pretrains that pass gave 3 runs: 1,925, 1,490, and 630 (mean 1,348). No run from a failed pretrain reached the online median.
- The s4 realistic pretrain overrates bang-bang by 0.9-1.9 per step at every checkpoint. This is the largest overrate of all probed models.
- The s42 result changes between checkpoints (bb gap 0.05 → 0.20 → 1.26 → 0.33). With a constant learning rate, the last checkpoint is a sample from a wide range. The gate result is thus partly chance.

### 4.4 Flip segments (`scripts/probe_flip_segments.py`, `data/probe_flip_segments.csv`)

Real 32-step segments from the realistic RL replays, open loop into each pretrain. "Real flip" segments flip in MuJoCo; "no real flip" segments stay upright (loops 10-29). The flipped gait comes from the s4 replay at loops 50-99; the upright gait comes from the s42 replay at loops 80-99.

| model | real flip: dream / true | dreamed flip | no real flip: dream / true | dreamed flip | flipped gait: dream / true | upright gait: dream / true |
|---|---|---|---|---|---|---|
| realistic s42 | 0.15 / 0.54 | 39 % | 0.12 / 0.33 | 27 % | −0.13 / −0.22 | 1.35 / 1.60 |
| realistic s4 | 0.51 / 0.54 | **0 %** | 0.48 / 0.33 | 0 % | −0.09 / −0.22 | 1.99 / 1.60 |
| scaled40 s42 | 0.16 / 0.54 | 0 % | 0.23 / 0.33 | 0 % | −0.23 / −0.22 | 1.13 / 1.60 |
| scaled40 s4 | −0.11 / 0.54 | 0 % | −0.07 / 0.33 | 0 % | −0.21 / −0.22 | 0.73 / 1.60 |
| online-only, repro s4 loop 50 | 0.58 / 0.54 | 42 % | 0.44 / 0.33 | 41 % | −0.10 / −0.22 | 1.22 / 1.60 |

- All models rate the flipped gait correctly as poor. The data does not teach a reward for the flipped gait.
- No model separates the segments that flip from the segments that do not. The s4 pretrain dreams no flips at all. The s42 pretrain dreams flips in both groups at almost the same rate. The flip share did not fix the flip blind spot of `report/9-30-26-repro`.
- In MuJoCo, the segments that lead into a flip earn more reward than the others (0.54 vs 0.33 per step), because the cheetah moves faster before it flips. The cost of the flip comes after the 32-step dream, as −0.2 per step for the rest of the episode. Only the value head can carry that cost, and the value head learns it from the replay only after the flips occur.

## 5. Head re-anchor (entries 60/61)

| run | policy KL before fit (loops 0-99) | KL after | value CE before → after |
|---|---|---|---|
| real_anchor_s42 | 0.001-0.066 | ≤ 0.002 | 1.10 → 1.10 … 1.80 → 1.79 |
| real_anchor_s4 | 0.000-0.032 | ≤ 0.001 | 1.22 → 1.22 … 1.78 → 1.77 |

The world-model phase moves the policy output by at most 0.07 nats. The fit removes that small change, so the anchor and no-anchor arms are the same algorithm with different random paths. The results differ in opposite directions (s42: 630 vs 1,490; s4: 474 vs −11). Head drift is not a cause of these failures. Do not use the re-anchor in the next queues.

## 6. Root cause

1. The realistic mix removed the only fast data with moderate actions (expert_scaled). Above 3 m/s, the data is again "saturated actions and high speed" (section 3).
2. A pretrain on this data can overrate aggressive actions. The s4 pretrain does so at every checkpoint. The s42 pretrain does so at some checkpoints (section 4.3).
3. With such a pretrain, the policy reaches the flip regime at loops 10-19, with 3-4× more saturated actions than the online runs (section 2).
4. The world model does not predict which segments flip (section 4.4). In imagination, the flip-causing gait looks as good as or better than a safe gait. The flips fill the replay. The policy then learns from flipped prompts, and the s4 runs stay in the flipped gait.
5. The s42 runs recover because their pretrain passes the gate (bang-bang below the online gait). The cost is still a slow start: 500 at loop 46 for no-anchor, against 31-35 for online runs.

Confidence: the dataset facts, the probe results, and the replay statistics are measured. The causal link from the gate to the RL outcome rests on 8 runs from 6 pretrains. The pretrain seed equals the RL seed in all of them, and the RL runs kept no world-model checkpoints (`checkpoint_every: 0`). Thus the world model during RL cannot be probed. Ablation A tests this link.

## 7. Recommended ablations

Order: A first (no new data or pretrain), then B. Use 2 seeds for each arm. Compare arms by the mean of loops 0-99 and by the flipped-step fraction, because one run cannot resolve differences below about 1,000.

**A. Seed cross and checkpoint gate (6 RL runs, about 2 h with 2 lanes, no code change).** The RL stage already accepts a step checkpoint (`offline_pretrained_checkpoint: .../pretrained_step_N.pt`).

| arm | pretrain checkpoint | RL seeds | gate | prediction if the pretrain decides |
|---|---|---|---|---|
| A1 cross | realistic s4, step 12,000 | 42 | fail (+0.06, gap +1.09) | flip lock-in |
| A2 cross | realistic s42, step 12,000 | 4 | pass (−0.36) | recovers |
| A3 gate fail | realistic s42, step 8,000 | 42, 4 | fail (+0.22, gap +1.26) | flip lock-in |
| A4 gate pass | realistic s42, step 4,000 | 42, 4 | pass (−0.46, gap +0.20) | recovers, as s42 step 12,000 |

Set `checkpoint_every` to 10 loops in these runs (about 1 GB per run), so that the flip probes can run on the world model at loops 10-30. If A3 locks in and A4 does not, the gate is causal. Then select each pretrain checkpoint with the gate before RL.

**B. Bring back fast data with moderate actions (1 collect, 2 pretrains, 2 RL runs).** Keep the realistic past runs and replace part of the expert + noise data with expert_scaled. Example: past runs 50 %, expert_scaled 25 %, expert + noise 10 % (std 0.2/0.4/0.8), random 15 %, flip share 0.45 (unchanged, one variable at a time). Gate each pretrain at steps 4,000, 8,000, and 12,000 with both probes before RL. The goal: bang-bang below the online gait at every checkpoint, and the tokenizer vx std below 3.

**C. Slow saturated data (needs a new data mode behind a flag).** Add a mode with random actions at the bounds, held for a few steps, from upright starts (about 10 %). This mode gives saturated actions at low speed, which breaks "saturated means fast" directly. It also gives many flip onsets from upright states, which section 4.4 shows the model does not learn from the flipped steady state. Run C after B, with the B mix as the base.

**D. Flip share 0 vs 0.45 (low priority).** The flip share did not teach flip onsets and uses 21 % of the past-run steps for flipped states. Test it only on the best mix from B or C.

**E. A stable end of pretraining (optional).** The gate result changes by up to 1.2 per step between checkpoints 4,000 apart. A learning-rate decay or a weight average (EMA) over the last 2,000 steps can make the final checkpoint less random. Test it only if A shows that the gate is causal.

Do not repeat: head re-anchor (section 5), rehearsal (`report/9-30-26-pretrain-analysis`), and more pretrain steps at a constant learning rate.

## 8. Files

- `scripts/run_table.py`: RL run table (`data/runs.csv`, `data/loops.csv`).
- `scripts/flip_events.py`: flips per episode from the replays (`data/flip_events.csv`, `data/flip_blocks.csv`).
- `scripts/dataset_compare.py`: realistic vs scaled40 dataset (`data/dataset_compare.csv`, `data/dataset_speed_bins.csv`).
- `scripts/pretrain_curves.py`: pretrain losses (`data/pretrain_curves.csv`).
- `scripts/probe_pretrains.sh`: velocity-resolution and action-ranking probes (`data/probe_latent_resolution.csv`, `data/probe_action_sources.csv`).
- `scripts/probe_flip_segments.py`: flip and gait segments (`data/probe_flip_segments.csv`).
- `scripts/probe_with_runs.py`: runs a probe of `report/9-30-26-*` with the hyperparameters of runs that `reports/9-28-26/data/run_text.json` does not hold.
- `data/probe_action_sources_steps.csv`: action ranking at pretrain steps 2,000, 4,000, and 8,000.
