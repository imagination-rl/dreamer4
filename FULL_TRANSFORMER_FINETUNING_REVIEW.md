# Full Transformer Finetuning Review

## Scope

This report reviews the optional full-transformer finetuning path.

It compares the implementation against the Dreamer 4 paper and this repository's HalfCheetah results.

The successful runs validate the existing frozen-transformer path. They do not validate the new full-transformer path.

## Evidence From Existing Runs

The strongest runs show clear HalfCheetah progress across two seeds.

| Run | Initial return | Final return | Peak return |
|---|---:|---:|---:|
| `old_reward_shifting`, seed 42 | -232 | 1,059 | 1,110 |
| `old_reward_shifting2`, seed 4 | -245 | 880 | 968 |
| `modded_better_baseline`, seed 42 | -240 | 833 | 915 |
| `modded_better_baseline2`, seed 4 | -251 | 869 | 972 |

The strongest pair reaches about 880 to 1,060 from initial returns near -240.

No event file records `full_transformer_finetuning`. Every successful run predates that option.

Those runs validate these properties:

- The frozen transformer supports useful policy updates.
- The refreshed policy prior supports online HalfCheetah adaptation.
- The existing world-model update schedule supports useful imagined trajectories.
- The existing default replay path works.

They do not validate these new properties:

- RL gradients through the transformer.
- Joint auxiliary and RL loss composition.
- Shared parameters under two optimizers.
- The new 32-step preservation data path.

## Issue Summary

| Rank | Issue | Assessment |
|---|---|---|
| 1 | Auxiliary loss dominates the RL objective | High |
| 2 | Two optimizers own the same transformer parameters | Medium-high |
| 3 | Preservation context uses 32 steps instead of 200 | Medium |
| 4 | The policy prior refreshes every outer loop | Medium, intentional adaptation |
| 5 | Partial batches conflict with static compilation | Low, inactive by default |
| 6 | The terminal function lacks preservation | Low for HalfCheetah |
| 7 | `cycle()` repeats identical windows | Incorrect finding |

## 1. Auxiliary Loss Scale Can Suppress the RL Gradient

This is the most important issue.

The full path adds three objectives before one backward pass:

- The policy loss.
- The value loss.
- The world-model preservation loss.

The relevant composition starts in `train/compile_runtime.py`.

The paper states that dynamics, reward, and prior losses preserve existing functions during imagination.

The paper also normalizes every loss with running RMS statistics. This repository disables that normalization in `train/train_gym.py`.

The successful runs expose the scale difference:

| Metric | Typical late-run value |
|---|---:|
| World-model loss | 19 to 21 |
| Logged reward loss | 2.4 to 2.6 |
| Agent policy plus value loss | 1.1 to 1.4 |

The reward loss uses eight prediction heads. The total objective sums those head losses.

The log reports their mean. A logged reward loss near 2.4 contributes about 19 to the world-model objective.

The full path therefore combines an objective near 20 with an agent objective near 1.2.

Raw scalar size does not equal gradient size. However, this difference creates a strong warning.

The preservation objective can dominate the shared transformer gradient. That result opposes the feature's main purpose.

The feature must let policy and value losses improve the agent-token representation.

A dominant reward objective instead anchors that representation to reward prediction.

Gradient clipping does not fix this conflict. Clipping limits the final combined gradient norm.

It does not restore the policy gradient's original direction.

This issue does not affect the old successful runs. Their transformer remains frozen during agent updates.

Their world-model and agent objectives never compete inside one transformer gradient.

A sound full path needs one of these controls:

- Restore paper-style RMS loss normalization under the flag.
- Add explicit coefficients for dynamics and reward preservation.
- Measure transformer gradient norms from each objective.
- Select coefficients from those measured ratios.

The default disabled path must retain its current loss behavior.

## 2. Two Optimizers Own Every World-Model Parameter

The world optimizer already owns the transformer and world-model heads.

When the flag is active, `agent_optimizer_parameters()` returns the entire world model.

That creates complete overlap for the world-model parameter set:

- World optimizer parameters: about 6.98 million.
- Agent optimizer parameters under the flag: about 9.00 million.
- Shared parameters: about 6.98 million across 147 tensors.
- Shared transformer parameters: about 1.21 million.

PyTorch permits this arrangement. It does not cause an immediate correctness error.

The conceptual problem comes from optimizer state.

Each optimizer keeps separate momentum or moment estimates for the same parameter.

One optimizer sees world-model updates. The other sees joint agent and preservation updates.

The transformer therefore follows two unrelated optimizer histories:

1. Thirteen world-model steps use the world optimizer state.
2. Three imagination steps use the agent optimizer state.
3. Both schedules apply to the same parameter values.
4. Each schedule also applies its own weight decay.

This creates a hidden objective-dependent update rule. The paper does not describe this mechanism.

It can also create unstable transitions between phases.

The second optimizer has no knowledge of the momentum that the first optimizer accumulates.

No successful run validates this behavior. All successful runs keep optimizer ownership disjoint.

A cleaner structure keeps ownership disjoint:

- The world optimizer owns the transformer and prediction heads.
- The agent optimizer owns the policy and value heads.
- The full-step backward computes all gradients once.
- The code steps both optimizers after that backward.
- RL gradients into the transformer use the world optimizer state.

This structure also preserves separate world and agent learning rates.

## 3. Preservation Uses a 32-Step Context Instead of 200 Steps

The normal world-model phase uses a sequence length of 200.

The new preservation loader uses the imagination horizon as its window length. The current horizon is 32.

This difference matters because the transformer contains temporal attention.

The auxiliary objective only anchors behavior within a short context during full updates.

A full update can therefore alter behavior that appears only after 32 steps.

The simultaneous preservation loss cannot detect that drift.

The normal world-model phase reduces this risk:

- Each outer loop performs 13 world-model steps.
- Each world-model batch has 128 sequences of 200 steps.
- Each outer loop then performs three full updates.
- Each preservation batch has 512 sequences of 32 steps.

That gives approximately:

- 332,800 world-model positions per outer loop.
- 49,152 preservation positions per outer loop.

The long-context phase processes about 6.8 times more positions. This provides frequent recovery and reduces the immediate risk.

However, token counts do not determine update strength directly.

Each optimizer step uses a mean loss and a full learning-rate update.

The short auxiliary context can still bias the transformer toward short-horizon behavior.

HalfCheetah can hide this problem because its dynamics are regular and its episode length is fixed.

A more faithful implementation uses the world-model sequence length for preservation.

Memory use then increases by 6.25 times at the current preservation batch size.

A practical design uses these values:

- Preservation sequence length: 200.
- Preservation batch size: 128.
- Separate configuration fields for both values.

This concern is valid, but it ranks below the objective-scale issue.

## 4. The Behavioral Prior Refreshes Every Outer Loop

The paper creates a frozen copy of the behavior policy before the imagination phase.

It then uses that copy as a stable PMPO prior.

This implementation copies the policy head and refreshes it before each group of agent updates.

That is a paper mismatch, but the repository context changes the correct response.

The paper's prior comes from behavior cloning. This repository has no behavior-cloning phase.

A permanently frozen initial policy here represents a random policy.

A fixed random prior can obstruct online improvement. It can force every future policy toward actions with no demonstrated competence.

The refreshed prior instead acts as a local trust region:

1. Copy the current policy.
2. Perform three agent updates.
3. Penalize large changes from that copy.
4. Refresh the copy during the next outer loop.

The successful runs all use this design. Their PMPO KL values remain active:

- About 0.045 and 0.081 in the strongest two runs.
- About 0.174 and 0.010 in the modified baseline pair.

This evidence supports the refreshed prior for the frozen-transformer pipeline.

Full transformer updates add one subtlety.

The prior head stays frozen, but its input representation changes as the transformer changes.

Therefore, the prior distribution is not globally fixed.

A frozen head can produce different actions from a modified hidden state.

The current mechanism preserves local output behavior during each three-step update group.

It does not preserve one immutable behavior policy.

This is a deliberate online adaptation, not a clear bug.

Useful experiments include these variants:

- Refresh every outer loop as the baseline.
- Freeze the prior after a return threshold.
- Refresh every fixed number of loops.
- Use an exponential-average prior.
- Compare returns, policy KL, and action entropy.

Do not freeze the initial random policy without evidence.

## 5. Static Compilation and Partial Preservation Batches

The compatibility issue requires both conditions:

- `old_replay_sampling=True`.
- Static compilation remains active.

The preservation loader keeps partial batches under the old replay mode.

The padding path excludes that mode in `train/train_gym.py`.

Replay size grows in increments of 64 episodes. The batch size is 512.

Remainder batches can therefore contain 64 through 448 episodes.

Static compilation can create one compiled graph for each shape. This increases compile cost and memory use.

The partial batch also receives one complete optimizer step.

A 64-item batch then has the same step count as a 512-item batch.

This can overweight the final replay subset.

The issue does not affect the strongest recorded runs. They use `old_replay_sampling=False`.

One older successful baseline uses the old replay option.

However, that run did not use full transformer updates or this preservation loader.

The safest fix pads preservation batches for both replay modes.

Another option uses replacement selection for this loader only.

## 6. Terminal Preservation Has Little Value for Current HalfCheetah Runs

The full path preserves dynamics and reward functions, but it omits terminal prediction.

That omission initially looks serious because the paper refers to world-model function preservation.

In this repository, the practical effect is close to zero.

HalfCheetah-v4 always reports `terminated=False`. Episodes end through the time limit.

The replay evidence matches that behavior:

- `modded_better_baseline`: 7,872 episodes and zero terminal events.
- `old_reward_shifting`: 8,768 episodes and zero terminal events.
- `old_reward_shifting2`: 8,704 episodes and zero terminal events.
- Every recorded episode ends through truncation.

The terminal loss settles near 0.056.

That matches the trivial nonterminal solution under label smoothing.

The imagination path also disables terminal return generation.

Imagined GAE therefore does not use predicted termination.

Real environment interaction uses Gym's actual termination flags.

It does not depend on the learned terminal head.

For this exact task, terminal preservation cannot explain model quality.

It becomes important under either change:

- Use an environment with genuine failure or success terminals.
- Enable predicted terminals inside imagined trajectories.

This issue has low severity for the current repository.

## 7. The Claim About Repeated Identical Windows Is Incorrect

The adversarial review claimed that `cycle()` repeats identical preservation windows.

That claim does not match the code.

The custom cycle function starts a new DataLoader iteration after each pass.

The replay dataset selects a random start position during each `__getitem__` call.

Therefore, each pass can produce new windows.

One inefficiency exists at small replay sizes.

With 64 episodes and a batch size of 512, the code repeats one batch eight times inside a step.

That repetition preserves the same mean gradient. It wastes compute but does not multiply the batch's statistical weight within that step.

The next full step creates a new loader pass and selects new start positions.

The successful runs also use the same general prompt-loader structure.

Their results give no evidence of a window-repetition failure.

This finding is not a model-quality issue.

## Conclusion

The full-transformer idea fits the paper and the repository's missing agent-token adaptation.

However, the current implementation has two important design risks:

1. The unnormalized preservation objective can overpower RL gradients.
2. Two optimizers maintain unrelated state for the same transformer parameters.

The context-length mismatch is real but less urgent.

The refreshed prior is defensible for this online, no-BC setup.

The terminal concern does not apply to present HalfCheetah runs.

The repeated-window concern is incorrect.

The next experiment must compare the disabled and enabled paths from the same checkpoint and seed.

Log these measurements during that experiment:

- Transformer gradient norm from policy and value losses.
- Transformer gradient norm from preservation losses.
- Cosine similarity between those gradients.
- Policy KL to the refreshed prior.
- Reward prediction loss before and after each agent phase.
- Long-context world-model loss before and after each agent phase.
- Environment return and action entropy.

These measurements distinguish collapse from auxiliary-loss domination or optimizer-state conflict.
