#!/usr/bin/env python3
"""PT 语料准备：LCS-558k → messages jsonl。

python data_prep.py --discover
python data_prep.py --prepare                    # 论文口径（558k 全量）
python data_prep.py --prepare --config tiny      # 小切片（本地验证）
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.deeprecur.common.prep import prep_main

STAGE = "stage0_pt"

if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent, "data_blend_lcs558k.json"))
