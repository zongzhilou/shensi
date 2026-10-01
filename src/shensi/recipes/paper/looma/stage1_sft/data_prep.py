#!/usr/bin/env python3
"""SFT 语料准备：parquet / jsonl → messages jsonl。

python data_prep.py --prepare --config default
python data_prep.py --prepare --config agent
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.stage1_sft.common import prep_main

STAGE = "stage1_sft"

if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent))
