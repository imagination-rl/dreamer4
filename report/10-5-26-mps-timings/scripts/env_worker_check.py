# usage: env_worker_check.py OUT.npz : step GroupedAsyncVectorEnv with fixed actions, save outputs, print worker memory
import sys, numpy as np
sys.path.insert(0, '/home/witherhoard/GithubProjects/dreamer4lucid')

def main():
    from train import train_gym
    env = train_gym.GroupedAsyncVectorEnv('HalfCheetah-v4', num_envs = 8, parallel_envs = 4)
    rng = np.random.default_rng(0)
    obs, info = env.reset(seed = 7)
    out = dict(obs0 = obs)
    o, r, te, tr = [], [], [], []
    for t in range(1100):  # past the 1000-step time limit, so autoreset runs too
        step = env.step(rng.uniform(-1, 1, size = (8, 6)).astype(np.float32))
        o.append(step[0]); r.append(step[1]); te.append(step[2]); tr.append(step[3])
    out.update(obs = np.stack(o), rew = np.stack(r), term = np.stack(te), trunc = np.stack(tr))
    def mem(pid):
        d = dict(l.split(':', 1) for l in open(f'/proc/{pid}/status'))
        return int(d['VmRSS'].split()[0]) // 1024
    libs = open(f'/proc/{env.processes[0].pid}/maps').read()
    print('worker RSS MB:', [mem(p.pid) for p in env.processes], 'torch loaded:', 'libtorch' in libs,
          'trunc steps:', np.argwhere(out['trunc'][:, 0]).ravel())
    env.close()
    np.savez(sys.argv[1], **out)

if __name__ == '__main__':
    main()
