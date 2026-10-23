# usage: loop_breakdown.py LO HI [AFTER_EPOCH] RUN... : per-loop sums of each timing tag (cpu and cuda), median over loops LO..HI
import sys, glob, statistics, bisect, pickle, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
lo, hi = int(sys.argv[1]), int(sys.argv[2])
after = float(sys.argv[3]); runs = sys.argv[4:]
CACHE = os.path.join(os.path.dirname(__file__), 'tbcache')
os.makedirs(CACHE, exist_ok = True)
def load(run):
    f = sorted(glob.glob(f'/data/logs/dreamer4lucid/runs/gym_imagination/{run}/events*'))[-1]
    ea = EventAccumulator(f, size_guidance = {'scalars': 0}); ea.Reload()
    return {t: [(e.wall_time, e.value) for e in ea.Scalars(t)] for t in ea.Tags()['scalars'] if t.startswith('timings/')}
def table(run):
    d = load(run)
    roll = d['timings/rollout/model/interact_with_env/cpu_milliseconds']
    # loop i = from the start of rollout i to the start of rollout i+1
    starts = [w - v / 1000 for w, v in roll]
    out = {}
    for tag, ev in d.items():
        name, metric = tag[len('timings/'):].rsplit('/', 1)
        if metric == 'cuda_milliseconds':  # written at flush; take the wall time of the cpu event with the same index
            cpu = d.get(f'timings/{name}/cpu_milliseconds')
            if cpu is None: continue
            ev = [(cpu[i][0], v) for i, (_, v) in enumerate(ev) if i < len(cpu)]
        sums = {}
        for w, v in ev:
            i = bisect.bisect_right(starts, w) - 1
            if 0 <= i < len(starts) - 1: sums[i] = sums.get(i, 0.) + v
        out[(name, metric)] = sums
    loops = [i for i in range(len(starts) - 1) if lo <= i <= hi and starts[i] >= after]
    out[('LOOP', 'wall')] = {i: (starts[i + 1] - starts[i]) * 1000 for i in loops}
    return out, loops
res = {r: table(r) for r in runs}
keys = sorted(set(k for r in runs for k in res[r][0]), key = lambda k: (k[0] != 'LOOP', k))
print(f"{'tag':58s}" + ''.join(f'{r.split("warmup15_")[-1][:13]:>14s}' for r in runs))
print(f"{'loops used':58s}" + ''.join(f'{len(res[r][1]):14d}' for r in runs))
for k in keys:
    if k[0].startswith('tokenizer/') or k[0].startswith('audit'): continue
    vals = []
    for r in runs:
        sums, loops = res[r][0].get(k, {}), res[r][1]
        v = [sums.get(i, 0.) for i in loops]
        vals.append(statistics.median(v) if v else float('nan'))
    if max(vals) < 5: continue
    print(f"{(k[0] + ' ' + k[1].replace('_milliseconds', '')):58s}" + ''.join(f'{x:14.0f}' for x in vals))
