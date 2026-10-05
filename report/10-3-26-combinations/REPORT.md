# Combinations of the grouped-normalization arms

Date: 2026-10-03. Queue: `scripts/queue_10-3-26c.sh` (stage 1), start 23:00. Entries 118-135 of `train/ablations.py`.

## 1. Selection of the arms

All one-change arms of the grouped-normalization base (entries 73, 77-117) were tested against the base. The base has 4 seeds (4, 42, 1, 2), each arm has 3 seeds (PMPO: 2). All runs are 100 loops. `scripts/stats.py` gives the tests; `data/tests.csv` holds the values.

An arm is selected when it improves on the base with p < 0.04 in the pooled t test (the method of the 10-2-26 report). The Welch t test is also given; it does not assume an equal seed spread.

- **Stability.** No arm has less loop-to-loop noise or smaller drops than the base with p < 0.04. The base is the smoothest run (noise 29, largest drop 32). `wm_pretrain640` and PMPO have significantly larger drops.
- **Seed variability.** No arm has a smaller seed spread than the base (Brown-Forsythe, all p > 0.26 except PMPO, which is larger).
- Thus only the return selects the arms.

| Arm | Return, loops 95-99 | Diff | p Welch | p pooled | Mean return, loops 0-99 | Diff | p Welch | p pooled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1,305 ± 98 | | | | 486 ± 117 | | | |
| `hl_gauss_value_loss` (cevalue) | 2,304 ± 106 | +999 | 0.0002 | <0.0001 | 850 ± 37 | +364 | 0.005 | 0.002 |
| `symlog_hl_gauss_value_head` (symlog) | 2,251 ± 112 | +946 | 0.0003 | <0.0001 | 796 ± 73 | +310 | 0.008 | 0.007 |
| `symexp_hl_gauss_actor_critic` (symexp) | 2,179 ± 257 | +874 | 0.019 | <0.0001 | 714 ± 89 | +228 | 0.033 | 0.041 |
| `norm_warmup15` (warmup15) | 2,023 ± 122 | +718 | 0.001 | <0.0001 | 1,027 ± 84 | +541 | 0.001 | <0.0001 |
| `wm_pretrain640` | 1,932 ± 157 | +627 | 0.008 | 0.0002 | 1,325 ± 213 | +839 | 0.010 | <0.0001 |
| `block2_attn_res` (block2) | 1,915 ± 155 | +610 | 0.008 | 0.0002 | 778 ± 183 | +292 | 0.089 | 0.010 |
| `full_attn_res` (fullres) | 1,830 ± 390 | +525 | 0.14 | 0.001 | 824 ± 254 | +338 | 0.13 | 0.004 |
| `reward_weight4` (rw4) | 1,754 ± 117 | +449 | 0.006 | 0.004 | 711 ± 102 | +225 | 0.044 | 0.043 |
| `no_attn_pool_depth5` (depth5) | 1,740 ± 148 | +435 | 0.018 | 0.005 | 738 ± 99 | +252 | 0.028 | 0.025 |
| `flat_lr340` | 1,550 ± 66 | +245 | 0.011 | 0.099 | 619 ± 36 | +133 | 0.10 | 0.22 |
| `no_attn_pool` | 1,517 ± 288 | +212 | 0.33 | 0.15 | 584 ± 145 | +98 | 0.39 | 0.36 |
| `flat_lr` | 1,299 ± 54 | −6 | 0.92 | 0.97 | 540 ± 41 | +54 | 0.44 | 0.61 |
| `pmpo` | 927 ± 318 | −378 | 0.33 | 0.028 | 133 ± 286 | −353 | 0.31 | 0.007 |

Selected: the 8 arms with a short name in the table.
- **Value arms (alternatives):** cevalue, symlog and symexp. These are three ways to apply HL-Gauss to the critic. Each one gets its own pair.
- **Attention arms (alternatives):** block2, fullres and depth5. fullres passes the pooled test only: one seed (42) ends at 1,384, so its seed spread is the largest of all arms.
- **Two other arms:** warmup15 (start speed) and rw4 (reward-head weight).
- **`wm_pretrain640` is not in this queue.** It passes both tests, but it has the largest drops (6 loops with a drop over 200 on average). Test it in a later combination.
- **Not selected:** `flat_lr340` (pooled p 0.099), `no_attn_pool`, `flat_lr` and PMPO.

## 2. Stage 1 queue

All runs: the base with two arms, 175 loops, seeds 4, 42 and 1.
- **Loops:** `stop_after_loops` stops each run at loop 175. `num_loops` stays 500, so the schedules of loops 0-99 are the same as in the 100-loop runs.
- **Checkpoint:** one checkpoint and replay snapshot, at loop 175.
- **Order:** seed 4 of all pairs runs first, then seed 42, then seed 1.
- **Pairs:** each value arm goes with warmup15, because the value arms change the slope after the start and warmup15 changes the start. Each attention arm goes with rw4, because both change the world model.

| Entries | Run | Arms |
| --- | --- | --- |
| 118-120 | `a2_grouped_combo_cevalue_warmup15_s*` | cevalue + warmup15 |
| 121-123 | `a2_grouped_combo_rw4_block2_s*` | rw4 + block2 |
| 124-126 | `a2_grouped_combo_symlog_warmup15_s*` | symlog + warmup15 |
| 127-129 | `a2_grouped_combo_symexp_warmup15_s*` | symexp + warmup15 |
| 130-132 | `a2_grouped_combo_rw4_depth5_s*` | rw4 + depth5 |
| 133-135 | `a2_grouped_combo_rw4_fullres_s*` | rw4 + fullres |

Stage 2 is not in the queue. After stage 1, the logs are reviewed and each pair is tested against its two arms and against the base. Then the stage-2 combinations are made by the rule of the request:
- If two pairs are better than their arms, run the two pairs together.
- If only one pair is better, run that pair + the better arm of the other pair.

## 3. Stage 1 results

All 18 runs ended with exit 0 (record `runs/queues/10-3-26c_2026-10-03_194348`). `scripts/stage1.py` gives the numbers and writes `data/stage1_*.csv`.

The comparison with the arms uses loops 0-99, because the arms ran 100 loops. The p values are from the pooled t test (seed variance pooled over the base, all arms and all pairs, 40 degrees of freedom), as for the selection of the arms. A star marks p < 0.04.

| Pair | Return, loops 95-99 | Against arm 1 | Against arm 2 | Mean return, loops 0-99 | Against arm 1 | Against arm 2 |
| --- | --- | --- | --- | --- | --- | --- |
| cevalue + warmup15 | 2,555 ± 277 | +251 (p 0.18) | +533 (p 0.006)* | 1,206 ± 116 | +356 (p 0.003)* | +179 (p 0.12) |
| symlog + warmup15 | 2,282 ± 426 | +31 (p 0.87) | +259 (p 0.17) | 1,112 ± 156 | +317 (p 0.008)* | +86 (p 0.45) |
| **symexp + warmup15** | **2,618 ± 372** | **+439 (p 0.022)*** | **+595 (p 0.002)*** | 1,133 ± 143 | +420 (p 0.001)* | +107 (p 0.35) |
| rw4 + block2 | 1,815 ± 249 | +60 (p 0.74) | −101 (p 0.59) | 934 ± 175 | +223 (p 0.055) | +157 (p 0.17) |
| rw4 + depth5 | 1,851 ± 201 | +97 (p 0.60) | +112 (p 0.55) | 720 ± 43 | +8 (p 0.94) | −18 (p 0.87) |
| rw4 + fullres | 1,677 ± 117 | −77 (p 0.68) | −153 (p 0.41) | 818 ± 135 | +106 (p 0.35) | −7 (p 0.95) |

All six pairs are above the base on both measures with p < 0.04 (base: 1,305 and 486).

Stability (loops 0-99):
- **symlog + warmup15 is less stable** than the base and than both of its arms: loop-to-loop noise 130 (base 29), largest drop 382, 7 loops with a drop over 200 (p < 0.04 for all three). After loop 100, seed 42 falls to 672 at loops 170-174 (largest drop 1,881).
- cevalue + warmup15 has more noise than the base (80, p 0.031), but not more than its arms.
- No pair is more stable than the base or than its arms.

Seed spread: no pair differs from its arms (Brown-Forsythe, all p > 0.25).

Loops 100-174 (no reference runs of 175 loops):

| Pair | Return, loops 170-174 | Best 5-loop mean | Last 5 below best | Largest drop | Loops with a drop over 200 |
| --- | --- | --- | --- | --- | --- |
| symexp + warmup15 | 3,680 ± 341 | 3,722 | 42 | 326 | 12 |
| cevalue + warmup15 | 3,323 ± 431 | 3,692 | 369 | 633 | 20 |
| symlog + warmup15 | 2,474 ± 1,573 | 3,133 | 658 | 1,114 | 21 |
| rw4 + depth5 | 2,436 ± 237 | 2,501 | 64 | 161 | 3 |
| rw4 + fullres | 2,068 ± 553 | 2,155 | 88 | 362 | 6 |
| rw4 + block2 | 1,945 ± 842 | 2,439 | 494 | 571 | 9 |

## 4. Stage 2 choice

The criteria are the same as for stage 1: a pair is better when it is above each of its two arms with p < 0.04 (pooled t test).
- **symexp + warmup15 is the only pair that is better** than both of its arms on one measure: +439 and +595 at loops 95-99.
- cevalue + warmup15 is above both arms in the mean values on both measures, but on each measure only one of the two differences has p < 0.04.
- symlog + warmup15 is not better than its arms, and it is less stable.
- No attention pair (rw4 + block2, depth5 or fullres) is better than its arms.

The rule of the request: when only one pair is better, run that pair + the better arm of the other pair. The better arm of the attention pairs is block2: 1,915 at loops 95-99 alone (rw4 1,754, fullres 1,830, depth5 1,740), and block2 passes both tests against the base.

Stage 2: **symexp + warmup15 + block2**, entries 136-138, seeds 4, 42 and 1, 175 loops. Queue `scripts/queue_10-4-26.sh`: seeds 4 and 42 run together at 13:00 (one set), and seed 1 runs at 23:00.

Not in stage 2:
- rw4: its pair with block2 is not better than block2 alone.
- wm_pretrain640: it is still to be tested.

## 5. Limits

- The arms do not have a reference run of 175 loops. The comparisons with the single arms use loops 0-99.
- Three seeds detect a difference of about 350 or more in the return at loops 95-99.

## 6. Files

- `scripts/extract.py`: reads the TensorBoard logs of all arms and seeds and writes `data/loop_metrics.csv`, `data/episodes.csv` and `data/run_summary.csv`.
- `scripts/stats.py`: does the tests of section 1 and writes `data/tests.csv`.
- `scripts/stage1.py`: does the stage-1 tests of section 3 and writes `data/stage1_summary.csv`, `data/stage1_tests.csv` and `data/stage1_returns.csv`.
