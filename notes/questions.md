## Questions for the Dreamer 4 Author

1. During the full-transformer imagination update, which parameters share one optimizer, and how many world-model and imagination steps use that state?
2. What exact coefficients, RMS statistics, MTP reductions, and sampling rules combine dynamics, reward, policy-prior, policy, and value losses?
3. Does the differentiable RL pass reuse rollout representations, or recompute them with which context noise, signal levels, step sizes, and cache rules?
4. Is the behavioral prior a frozen policy head or a frozen policy-plus-transformer model, and when does the prior refresh?
5. Has full-transformer imagination RL converged without behavior cloning, and what batch size, update ratio, and initialization did that run use?
