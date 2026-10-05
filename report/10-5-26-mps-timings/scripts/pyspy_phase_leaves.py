# usage: pyspy_phase_leaves.py RAW : for each loop phase, classify leaf samples (kernel launch, GPU wait, python/other)
import sys, re, collections
phases = [('rollout', 'interact_with_env'), ('generate', '_generate_optimized'), ('ppo', 'train_cleanrl_updated_ppo_dream'),
          ('world_model', 'train_world_model')]
def classify(frames):
    leaf = frames[-1]
    joined = ';'.join(frames)
    if leaf.startswith('launcher') or 'triton' in leaf: return 'triton launch'
    if '<lambda> (dreamer4/dreamer4.py:7006)' in joined or ('.item' in leaf) or 'clip_grad_norm_by_group' in leaf or leaf.startswith('synchronize'): return 'wait for GPU (sync)'
    if 'step (gymnasium' in joined or '_recv' in leaf or 'connection.py' in leaf: return 'env step (pipes)'
    if '_engine_run_backward' in leaf or 'backward' in leaf: return 'autograd backward (C++)'
    if re.match(r'call \(', leaf) or 'compile_fx' in joined and leaf.startswith('call'): return 'inductor call'
    if 'guards.py' in leaf or 'eval_frame' in leaf: return 'dynamo guards'
    return 'other python/torch'
tot = collections.Counter(); per = collections.defaultdict(collections.Counter); total = 0
for line in open(sys.argv[1]):
    line = line.rstrip()
    if not line: continue
    stack, n = line.rsplit(' ', 1); n = int(n); frames = stack.split(';'); total += n
    ph = next((name for name, key in phases if any(f.startswith(key + ' ') for f in frames)), None)
    if ph is None: continue
    tot[ph] += n; per[ph][classify(frames)] += n
print('samples', total)
for ph, _ in phases:
    print(f'{ph:12s} {tot[ph]/total:6.1%} of all: ' + ', '.join(f'{k} {v/tot[ph]:.0%}' for k, v in per[ph].most_common()))
