"""Train and validation losses of the offline pretrain runs, in blocks of 1,000 steps, and the tokenizer loss.

Output: report/9-30-26-pretrain-analysis/data/pretrain_curves.csv
Usage:  python report/9-30-26-pretrain-analysis/scripts/pretrain_curves.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
G = REPO / "runs" / "gym_imagination"
RUNS = {
    "1m_a (s42)": "offline_pretrain_a2_next_state_reward",
    "1m_b_s42": "offline_pretrain_a2_next_state_reward_s42",
    "1m_b_s4": "offline_pretrain_a2_next_state_reward_s4",
    "pastruns5p5m_s4": "offline_pretrain_a2_past_runs_scaled_expert_s4",
    "scaled40_s42": "offline_pretrain_a2_scaled40_s42",
    "scaled40_s4": "offline_pretrain_a2_scaled40_s4",
}
TRAIN = ("world_model/flow_loss", "world_model/shortcut_loss", "world_model/reward_loss", "world_model/reward_mae",
         "world_model/grad_norm", "world_model/trunk_grad_norm", "tokenizer/recon_loss")
VAL = ("flow_loss", "shortcut_loss", "reward_loss", "reward_mae")


def main():
    rows = []
    for name, folder in RUNS.items():
        series = {}
        for f in glob.glob(str(G / folder / "events.out.tfevents.*")):
            for ev in EventFileLoader(f).Load():
                if ev.HasField("summary"):
                    for v in ev.summary.value:
                        if v.tag in TRAIN or v.tag.startswith("offline_pretrain/val/world_model/"):
                            series.setdefault(v.tag, {})[ev.step] = v.simple_value if v.HasField("simple_value") else float(np.frombuffer(v.tensor.tensor_content, np.float32)[0]) if v.tensor.tensor_content else v.tensor.float_val[0]
        tok = pd.Series(series.get("tokenizer/recon_loss", {})).sort_index()
        steps = sorted(series.get("world_model/flow_loss", {}))
        last = max(steps) if steps else 0
        for hi in range(1000, last + 1, 1000):
            row = dict(pretrain = name, step = hi)
            for tag in TRAIN[:-1]:
                s = pd.Series(series.get(tag, {})).sort_index()
                row["train_" + tag.split("/")[-1]] = s[(s.index > hi - 1000) & (s.index <= hi)].mean()
            for tag in VAL:
                val = series.get("offline_pretrain/val/world_model/" + tag, {})
                row["val_" + tag] = next((x for st, x in val.items() if abs(st - hi) < 2), np.nan)
            row["tokenizer_recon_final"] = tok.iloc[-100:].mean() if len(tok) else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(HERE.parent / "data" / "pretrain_curves.csv", index = False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(out.round(4).to_string(index = False))


if __name__ == "__main__":
    main()
