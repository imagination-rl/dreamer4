"""Collect the data of entries 166-168 (OFFLINE_DATA_OWN_RUNS) on the CPU: the collect stage of entry 166 only.

The queue entries list the collect stage too; it then finds this dataset, checks its config, and skips the collection.
Usage: python report/10-4-26-pretrain-data/scripts/collect_own_runs.py
"""

from train import train_gym
from train.ablations import build_ablation_specs

kwargs = dict(build_ablation_specs(ablation_indices = [166])[0].train_kwargs)
kwargs.update(offline_stages = ("collect",), cpu = True, require_cuda = False)
train_gym.main(**kwargs)
