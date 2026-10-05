import sys, re, collections
path = sys.argv[1]; focus = sys.argv[2] if len(sys.argv) > 2 else None
total = 0; incl = collections.Counter(); child = collections.Counter(); leaf = collections.Counter()
for line in open(path):
    line = line.rstrip()
    if not line: continue
    stack, n = line.rsplit(' ', 1); n = int(n); frames = stack.split(';')
    total += n
    names = [re.sub(r' \(.*?:\d+\)$', '', f) + ' ' + (re.search(r'\((.*?):\d+\)', f).group(1).split('/')[-1] if re.search(r'\((.*?):\d+\)', f) else '') for f in frames]
    seen = set()
    for nm in names:
        if nm not in seen: incl[nm] += n; seen.add(nm)
    leaf[names[-1]] += n
    if focus:
        for i, nm in enumerate(names):
            if nm.startswith(focus + ' ') and i + 1 < len(names):
                child[names[i+1]] += n; break
print('total samples', total)
for k in ['interact_with_env ', 'train_world_model ', 'train_agent_in_imagination ', 'generate ', 'train_cleanrl_updated_ppo_dream ', 'collect_and_store ', 'main ']:
    for nm, c in incl.items():
        if nm.startswith(k): print(f'{c/total:6.1%} {nm}')
if focus:
    print('--- children of', focus)
    for nm, c in child.most_common(25): print(f'{c/total:6.1%} {nm}')
print('--- top leaves')
for nm, c in leaf.most_common(25): print(f'{c/total:6.1%} {nm}')
