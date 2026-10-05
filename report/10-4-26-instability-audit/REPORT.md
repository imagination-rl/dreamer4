# Instability Audit of the Stage-1 Runs (2026-10-04)

The full report with figures and tables is `page.html`. `scripts/build_page.py` makes it from `data/` and `figures/`. It is also published at https://claude.ai/artifact/2GBxJsQbtQ8WNLWyUtqijA (private).

## Findings

1. **The return drops are flips.** In loops 100-174 of the pair runs, flipped episodes explain about 40 % of the loop-to-loop return variance. None of 12,431 flip onsets recovered.
2. **The policy is not trained on flipped states.** On flipped states its entropy is 2.5-4.2 nats, near the uniform 4.16. From the same start states, 3-8 % of dreams end flipped, against 9-41 % of real rollouts.
3. **A flip starts as bad luck.** Simulator branches started 30 steps before 113 recorded flips: new policy samples flip 5 % of the time, and the mean action 1 %. Of the 113 flips, 106 came from a rare draw of the action noise. As the body tips, the policy entropy rises by 2.1 nats.
4. **World-model drift is larger than the PPO step.** One loop of world-model training (with the policy heads fixed) moves the acting policy by a median of 3.3 times that loop's logged PPO step. The drift is larger in 93 % of 44 checkpoints. Nothing limits this change.
5. **The PPO step is large.** With the same PPO settings, the world-model runs have approx-KL 0.03-0.05 and clip fraction about 0.31. Model-free PPO has 0.006-0.008 and 0.06.
6. **The entropy collapse is normal per env step, but not per return.** At the same real return, the policy is 2-3.5 nats narrower than model-free PPO.
7. **No precursor.** The loops before a flip burst look like any other loop. The flip counts vary more than binomial chance, with no lag-1 persistence. Each loop re-draws the flip risk.
8. **Context length.** With the default attention, the 1,000-frame acting context against the 40-frame training context has a small effect. In the rw4 attention pairs it is larger (mean KL 0.03-0.29), but it stays below the drift.

## Hypotheses of the Conversation

| Hypothesis | Result |
| --- | --- |
| The policy learns too fast | Partly. The PPO steps are 4-7 times the model-free size, but no single step comes before a drop. |
| The entropy collapse is normal for a Beta policy | Per env step yes. Per return no. |
| Confident in imagination, normal in reality | Not supported on upright states (difference 0.15 nats or less in the value pairs). Only flipped states are uncertain. |
| Moving representations | Supported (finding 4). |

## Next Tests

- `preserve_policy_during_world_model_updates` and `reanchor_heads_after_world_model_updates` on the current base. Neither flag was tested on it.
- A target-KL stop for PPO.
- Oversampled flip-onset windows in world-model training.
- An evaluation with the mean action.
- In the 10-4-26 queue: `wmlr300`, `lr150`, `ent001` and `ent01` test findings 4-6. Read the flipped fraction of the entropy runs as well as their return.

## Scripts (in `scripts/`)

| Script | Output |
| --- | --- |
| `replay_stats.py` | `data/replay_loops.csv`: real-state entropy, flips and returns for each loop, from the replay log-probabilities |
| `extract_model_free.py` | `data/model_free.csv` |
| `compare_model_free.py` | `data/compare_*.csv` |
| `flip_onsets.py` | `data/flip_onsets.csv` |
| `flip_bursts.py`, `dispersion.py` | `data/flip_bursts*.csv`, `data/dispersion.csv` |
| `probe_common.py`, `probe_context_drift.py`, `run_probes.py` | `data/probe/` (44 checkpoints) |
| `probe_flip_counterfactual.py`, `run_counterfactuals.py` | `data/counterfactual/` (31 checkpoints) |
| `plot.py`, `build_page.py` | `figures/`, `page.html` |

## Limits

- The probes run on the CPU, because the GPU was in use.
  - The world-model step uses micro-batches of 8 (the run used 32). The gradient is the same; the loss-normalization statistic sees 8 rows instead of 32.
  - Only the first three drift jobs measured the 8- and 200-frame contexts.
- The loop-175 checkpoints did not collect episodes. Their probes use the episodes of loops 170-174.
- The model-free reference has 2 seeds and no flip measure.
