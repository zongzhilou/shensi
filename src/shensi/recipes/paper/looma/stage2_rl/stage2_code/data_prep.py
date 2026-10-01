#!/usr/bin/env python3
"""stage2_code：prompts → parquet（verl RLVR schema）。

python data_prep.py --prepare --config default
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.stage2_rl.common import prep_main

STAGE = "stage2_code"

if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent))
