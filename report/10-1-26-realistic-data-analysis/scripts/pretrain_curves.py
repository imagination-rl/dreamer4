"""Pretrain losses of the realistic-data pretrains (entries 56/57) and the scaled40 pretrains (45/46).

It uses report/9-30-26-pretrain-analysis/scripts/pretrain_curves.py with a different run list.
Output: report/10-1-26-realistic-data-analysis/data/pretrain_curves.csv
Usage:  python report/10-1-26-realistic-data-analysis/scripts/pretrain_curves.py
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("prev", REPO / "report/9-30-26-pretrain-analysis/scripts/pretrain_curves.py")
prev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prev)
prev.RUNS = {"scaled40_s42": "offline_pretrain_a2_scaled40_s42", "scaled40_s4": "offline_pretrain_a2_scaled40_s4",
             "realistic_s42": "pretrain_realistic_data_s42", "realistic_s4": "pretrain_realistic_data_s4"}
prev.HERE = HERE

if __name__ == "__main__":
    prev.main()
