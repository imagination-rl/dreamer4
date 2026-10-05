# usage: loop_phases_range.py RUN LO HI [AFTER_EPOCH]: phase medians for loops LO..HI (rollout step index), optional wall-time floor
import sys, glob, statistics, datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
run, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
after = float(sys.argv[4]) if len(sys.argv) > 4 else 0.
f = sorted(glob.glob(f'/data/logs/dreamer4lucid/runs/gym_imagination/{run}/events*'))[-1]
ea = EventAccumulator(f, size_guidance={'scalars': 0}); ea.Reload()
def w(tag): return ea.Scalars(tag)
roll = w('timings/rollout/model/interact_with_env/cpu_milliseconds')
wm = [e.wall_time for e in w('timings/world_model/step/cpu_milliseconds')]
gen = [e.wall_time for e in w('timings/imagination/model/generate/cpu_milliseconds')]
lf = [e.wall_time for e in w('timings/imagination/model/learn_forward/cpu_milliseconds')]
rows = []
for i in range(len(roll) - 1):
    if not (lo <= i <= hi) or roll[i].wall_time < after: continue
    a, rdur = roll[i].wall_time, roll[i].value
    nxt, ndur = roll[i + 1].wall_time, roll[i + 1].value
    nstart = nxt - ndur / 1000
    W = [x for x in wm if a < x < nstart]; G = [x for x in gen if a < x < nstart]; L = [x for x in lf if a < x < nstart]
    if not (W and G and L): continue
    rows.append(dict(rollout=rdur / 1000, wm=W[-1] - a, gen=G[-1] - W[-1], ppo=L[-1] - G[-1], loop=nxt - a, t=a))
if not rows: print(run, 'no rows'); sys.exit()
med = {k: statistics.median(r[k] for r in rows) for k in ['rollout', 'wm', 'gen', 'ppo', 'loop']}
ts = lambda t: datetime.datetime.fromtimestamp(t).strftime('%H:%M')
print(f"{run[-34:]:34s} n={len(rows):3d} {ts(rows[0]['t'])}-{ts(rows[-1]['t'])}  " + '  '.join(f'{k} {v:5.2f}' for k, v in med.items()))
