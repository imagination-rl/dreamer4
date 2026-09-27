# Prompts for the RL audits

Two `gpt-6-astra` sessions produced the reports below. This file excludes the fenced nine-idea reference block and its lead-in from each prompt.

## 2026-09-14: `runs/rl_audit/report.md`

Session ID: `01a0a1c7-cc8c-7290-8b33-61add2da7b4f`

### Prompt 1

````text
Currently, the RL dreamer4 agent learns very slowly. Perform a deep analysis of why it the model so slow to learn. The issues may include loss masking, RL algorithm implementation, or a variety of other subtle problems.

For context: 
- Standard SAC without a world model gets far better performance 
- REINFORCE does not seem to fix it 
- sm_* ablations were meant to test just the RL algorithm 
- Be vary of the differences between env steps and time, especially in the sm_* runs and the meaning of the differences there

The starting point is train/ablations.py. 

For everything, come up with theories, find evidence to support and refute them, and perform a detailed investigation. Recommend a small number of the most important ablations to run.
Take as long as you need.
````

## 2026-09-15: `runs/rl_audit_2026-09-15/report.md`

Session ID: `01a0a6cc-aa58-7863-a522-36320719debd`

### Prompt 1

````text
$unslop

Currently, the RL dreamer4 agent learns very slowly. Perform a deep analysis of why it the model so slow to learn. 
The issues may include loss masking, RL algorithm implementation, reward/action shifting, or a variety of other subtle problems. There is also a high chance of there being multiple problems.

For context: 
- Standard SAC without a world model gets far better performance (15k reward in a much shorter number of env steps) 
- REINFORCE and PPO for the RL algorithm does not seem to fix it
- sm_* ablations were meant to test just the RL algorithm
- Be vary of the differences between env steps and time, especially in the sm_* runs and the meaning of the differences there
- The goal is to make the dreamer4 agent effective, not just sm_*

The starting point is train/ablations.py.

For everything, come up with theories, find evidence to support and refute them, and perform a detailed investigation. Recommend a small number of the most important ablations to run.
Take as long as you need.
````

### Prompt 2

````text
You have 30% 5hr limit left, so keep that in
````
