# Differences between `sm_cleanrl_updated_PPO` and `cleanrl_updated_PPO`

## Scope

This report compares the seed 42 runs on `HalfCheetah-v4`.

The review used these sources:

- `train/ablations.py`, `train/config.yaml`, `train/train_model_free.py`, and `train/train_gym.py`.
- The PPO loss and value loss code in `dreamer4/dreamer4.py`.
- Both TensorBoard event files.
- `slurm-3189648.out` and `slurm-3190466.out`.
- The `loop_50.pt` checkpoint from `cleanrl_updated_PPO`.

The return comparison uses raw Gym episode returns. The small run records statistics before its reward normalization wrapper.

## Main result

The two runs start at nearly the same return. Their results separate within the first 640,000 environment steps.

| Environment steps | `sm_cleanrl_updated_PPO` | `cleanrl_updated_PPO` |
|---:|---:|---:|
| 64,000 | -237.0 | -234.3 |
| 640,000 | 119.2 | -174.9 |
| 1,600,000 | 2,023.9 | -198.3 |
| 3,200,000 | 4,130.9 | -247.5 |
| 4,608,000 | 5,091.9 | -133.5 |

At 4.608 million steps, the small run leads by 5,225.4 return points.

Across the first 72 rollouts, the small run averages 2,736.9. The other run averages -173.2.

The last ten matched rollouts average 4,845.3 and -170.5. The difference is 5,015.7 points.

The best matched rollout averages are 5,091.9 and -55.5. `cleanrl_updated_PPO` never records a positive rollout average.

The small run later reaches 9,333.4 at 27.712 million steps. Its last ten rollouts average 9,293.5.

The small run first crosses these rollout averages:

| Return | First environment step |
|---:|---:|
| 0 | 384,000 |
| 1,000 | 1,152,000 |
| 3,000 | 2,240,000 |
| 5,000 | 4,608,000 |
| 7,000 | 8,128,000 |
| 9,000 | 20,608,000 |

All episodes last 1,000 Gym steps. `cleanrl_updated_PPO` reports `rollout/average_length=1001` because replay includes one bootstrap item.

## Run completion and log limits

Neither run completes the planned 500 loops.

| Item | `sm_cleanrl_updated_PPO` | `cleanrl_updated_PPO` |
|---|---:|---:|
| Logged rollouts | 433 | 72 |
| Logged environment steps | 27,712,000 | 4,608,000 |
| Planned environment steps | 32,000,000 | 32,000,000 |
| Completion by rollouts | 86.6% | 14.4% |
| Stop reason | Signal `Terminated` | Slurm time limit |
| Checkpoints | None | Loops 25 and 50 |

The second run finishes 71 full loops. It then completes one PPO batch and 277 updates from the next batch.

The event file contains 215 generated batches and 68,757 PPO updates. A complete generated batch has 320 updates.

The launcher starts seeds 42 and 4 as two processes. It delays the second process by 60 seconds.

The Slurm script requests one GPU. Both processes therefore share one H100.

The `cleanrl_updated_PPO` log records several CUDA allocation failures. It also records compile fallbacks and graph breaks.

The process survives those warnings. The time limit stops it during an update.

## PPO configuration that matches

The core PPO constants match in both code paths.

| Parameter | Value |
|---|---:|
| Initial policy rate | 0.0003 |
| Discount factor | 0.99 |
| GAE lambda | 0.95 |
| PPO clip coefficient | 0.2 |
| Value loss coefficient | 0.5 |
| Entropy loss coefficient | 0.0 |
| Epochs per batch | 10 |
| Minibatches per epoch | 32 |
| Gradient norm limit | 0.5 |
| Adam epsilon | 0.00001 |
| Policy distribution | Bounded Beta |
| Policy rate schedule | Linear decay over 500 loops |

Both policies use `alpha,beta = 1 + softplus(head)`. Both store actions in the native Beta range.

Both policies rescale native actions to the Gym action range. Both compute a joint action probability across six action dimensions.

The PPO constant set does not explain the return gap. The data path, network, value loss, and execution path differ.

## Code differences

### Entry route and active configuration

`sm_cleanrl_updated_PPO` sets `small_model_only=true`. It selects `small_model.cleanrl_updated_ppo=true`.

`cleanrl_updated_PPO` sets the top-level `cleanrl_updated_ppo=true`. It also sets `align_generated_dream_transitions=true`.

The small route exits `train_gym.main()` before most baseline options take effect. Its recorded file still lists those inactive options.

The small route requires `checkpoint_every=0`. The other route keeps the baseline interval of 25 loops.

### Environment path

The small route always creates a `SyncVectorEnv`. It ignores the recorded `env_vectorization_mode=async` value.

The other route uses the configured asynchronous vector environment. Both routes use 64 environments.

The small route applies these wrappers:

1. Flatten the observation.
2. Record raw episode statistics.
3. Normalize observations and clip them to `[-10, 10]`.
4. Normalize rewards with discount `0.99` and clip them to `[-10, 10]`.

The other route records episode statistics but does not normalize Gym observations or rewards.

The small route resets once with seed 42. Later autoresets continue each environment's random number stream.

The other route resets each rollout with a derived seed. The seed changes with the loop index.

### Policy and critic networks

The small agent has separate actor and critic multilayer perceptrons. Each network has two 64-unit `Tanh` layers.

The actor consumes the normalized 17-value observation. The critic consumes the same observation and emits one scalar.

The complete small agent has 11,469 parameters. PPO updates all of them.

The other policy consumes seven 128-value tokens as one 896-value input. Its hidden width is 512.

Its value head consumes a 128-value critic representation. It emits 255 categorical value logits.

Its PPO optimizer updates 2,021,631 policy, action readout, and value head parameters. This is 176.3 times the small agent count.

The upstream representation stays fixed during each PPO update. The small route updates its input layers with PPO.

### Value targets and value loss

The small critic uses a scalar value. Its loss is clipped scalar mean squared error.

The other critic uses an HL-Gauss categorical value. Its support spans `[-1200, 1200]` with 255 bins.

That route encodes scalar returns as distributions. It then uses categorical cross-entropy for the value loss.

Both routes clip value changes by 0.2. The second route applies the clip before it encodes the categorical target.

The losses therefore share a coefficient and clip size, but they optimize different objectives.

### PPO batch construction

The small route collects 1,000 steps from 64 environments. Each PPO batch has 64,000 transitions.

It divides that batch into 32 minibatches of 2,000 transitions. Ten epochs produce 320 optimizer updates per rollout.

The other route creates three PPO batches per environment loop. Each batch starts with 512 trajectories of length 32.

The last item supplies bootstrap data. The code keeps 31 valid transitions per trajectory.

Each PPO batch therefore has 15,872 valid transitions. It uses 32 minibatches of 496 transitions and ten epochs.

One environment loop produces 960 PPO updates on this route. The small route produces 320 updates per loop.

The small route reuses 640,000 transition samples per loop. The other route reuses 476,160 samples per loop.

The second route performs three times as many optimizer steps. Each step uses about one quarter of the small minibatch size.

### Transition alignment and GAE

The small route computes GAE across each 1,000-step Gym rollout. It distinguishes termination from time-limit truncation.

For a truncation, it evaluates the final observation for bootstrap. It stops the GAE trace at either episode boundary.

The other route shifts each reward left by one position. This pairs each action with the next arrival reward.

It freezes GAE returns before the ten PPO epochs. It removes the final bootstrap item before minibatch selection.

Both routes normalize advantages inside each minibatch. Both use sample standard deviation with an epsilon of `1e-8`.

### Optimizer and numeric mode

Both routes use Adam with epsilon `1e-5`. Neither PPO optimizer uses weight decay.

The small route disables TF32. It requests deterministic cuDNN behavior and the highest matrix precision.

The other route enables TF32 through the baseline configuration. It requests high matrix precision and compiled execution.

The small route uses an explicit seeded generator for minibatch permutations. The other route uses the global Torch generator.

### Probability and entropy values

The small route subtracts the action-scale Jacobian from Beta log probabilities. It adds the same scale to entropy.

The other route logs probability and entropy in the native Beta range. The PPO ratio is unchanged by the constant Jacobian.

For six `[-1, 1]` actions, the entropy offset is `6 * log(2)`, or 4.159. Entropy values cannot be compared directly.

The entropy coefficient is zero in both runs. This metric offset does not change either update.

### Log frequency

The small route logs learner metrics once after all 320 updates. It has 433 learner records.

The other route logs learner metrics after every minibatch update. It has 68,757 learner records.

Raw record counts therefore do not measure equivalent update volume.

## Learner behavior in the logs

The policy update sizes differ despite the shared PPO constants.

The log aggregation differs. Treat the ratios below as diagnostics, not exact paired estimators.

| Metric over matched steps | `sm_cleanrl_updated_PPO` | `cleanrl_updated_PPO` |
|---|---:|---:|
| Mean approximate KL | 0.00784 | 0.07363 |
| Final approximate KL | 0.00671 | 0.08202 |
| Mean ratio clip fraction | 0.0769 | 0.3945 |
| Final ratio clip fraction | 0.0670 | 0.5121 |

The second route has 9.4 times the mean approximate KL. Its mean clip fraction is 5.1 times larger.

Its maximum logged approximate KL mean is 2.345. Its maximum per-sample approximate KL is 1,128.4.

The small critic reaches 0.790 explained variance at the matched endpoint. It reaches 0.936 at its final logged rollout.

The other route does not log explained variance. Its mean absolute value error is 2.027 in its own return scale.

The internal `imagination/dream_return` metric moves from -6.988 to 0.259. Its mean is 1.969 and its maximum is 15.541.

That internal increase does not transfer to the logged Gym rollout return across the available 72 rollouts.

## Runtime evidence

The event timestamps cover 27.648 million step increments in 1.045 hours for the small run. This is about 7,351 steps per second.

They cover 4.544 million step increments in 1.465 hours for the other run. This is about 862 steps per second.

The observed throughput ratio is about 8.5. The shared GPU and allocator warnings confound this runtime comparison.

The return comparison has a stronger basis than the runtime comparison. It uses the same seed and matched environment steps.

## Conclusions

The small run learns a strong HalfCheetah policy under the recorded conditions. The other run shows no direct return gain by 4.608 million steps.

The failure does not come from different headline PPO constants. Those constants match.

The largest implementation changes are the network size, value objective, batch layout, reward path, and numeric mode.

The large KL and clip fractions identify an unstable policy update regime in `cleanrl_updated_PPO`. They coincide with its flat Gym returns.

The result covers one seed and two partial runs. The Slurm launch also shares one GPU between two seed processes.
