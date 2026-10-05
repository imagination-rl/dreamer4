"""Run a probe script of report/9-30-26-* with the hyperparameters of runs that reports/9-28-26/data/run_text.json does
not hold (the realistic-data and scaled40 RL runs). It reads them from the TensorBoard text summary of each run.

Usage:  python report/10-1-26-realistic-data-analysis/scripts/probe_with_runs.py <probe script> [probe args ...]
"""

from __future__ import annotations

import glob
import json
import runpy
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "reports" / "9-28-26" / "scripts"))
import common  # noqa: E402

EXTRA = ["pretrain_rl_realistic_data_s42", "pretrain_rl_realistic_data_s4", "offline_pretrained_a2_scaled40_no_rehearsal_s42",
         "offline_pretrained_a2_scaled40_no_rehearsal_s4"]


def tb_hyperparameters(run: str) -> dict:
    for f in glob.glob(str(REPO / "runs" / "gym_imagination" / run / "events.out.tfevents.*")):
        for ev in EventFileLoader(f).Load():
            if ev.HasField("summary"):
                for v in ev.summary.value:
                    if v.tag == "run/hyperparameters/text_summary":
                        return json.loads(v.tensor.string_val[0].decode().split("```json")[1].split("```")[0])
    raise KeyError(run)


text = dict(common.run_text())
text.update({run: {"hyperparameters": tb_hyperparameters(run)} for run in EXTRA})
common.run_text = lambda: text
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name = "__main__")
