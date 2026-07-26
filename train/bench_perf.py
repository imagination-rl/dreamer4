"""Compatibility entry point for the canonical compile benchmark."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.benchmark_halfcheetah_compile import *  # noqa: F401,F403,E402


if __name__ == "__main__":
    import fire

    fire.Fire(main)
