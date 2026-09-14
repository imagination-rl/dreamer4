# REINFORCE implementation research

Date: 2026-09-04

Audience: Dreamer 4 maintainers

## Decision

Implement REINFORCE as `objective: reinforce` in the shared policy loss.

Use this policy objective:

```text
loss = -mean(log pi(action | state) * detached advantage)
```

Sum component log probabilities first because one continuous action vector is one policy event.

Use discounted reward-to-go returns with a detached state-value baseline.

Set GAE lambda to one only for REINFORCE.

Keep one full-batch update for each state-only rollout batch.

## Rationale

The original method uses sampled returns with the policy score.

The policy-gradient theorem supports an advantage estimate in the same score-function form.

A state-value baseline reduces variance without action-dependent bias.

Lambda one recovers the unbiased discounted advantage estimate.

PPO ratios and clipping do not belong in the REINFORCE policy objective.

PMPO sign partitions and policy-prior KL terms also do not belong in this objective.

The current value loss, masks, entropy term, and truncated-horizon bootstrap remain compatible.

## Operational constraint

REINFORCE is on-policy.

The standard path creates a new dream before every optimizer update.

The small path permits REINFORCE only with one full-batch update per rollout batch.

## Sources

- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf), Sutton et al., 1999.
- [Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html), OpenAI Spinning Up.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), Schulman et al., 2015.
- [PyTorch probability distributions](https://docs.pytorch.org/docs/stable/distributions.html), PyTorch documentation.

## Limitations

REINFORCE has higher variance than PPO or PMPO.

The fixed imagination horizon needs a critic bootstrap.

That bootstrap ties the last return estimate to critic accuracy.
