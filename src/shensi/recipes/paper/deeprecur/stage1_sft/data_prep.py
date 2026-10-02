#!/usr/bin/env python3
"""SFT 语料准备：LLaVA-mixed-665k（或 HD 的 748K 混合）→ messages jsonl。

python data_prep.py --discover
python data_prep.py --prepare                    # 论文口径（665k）
python data_prep.py --blend data_blend_hd.json --prepare   # DeepStack-HD 的 748K 混合
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.deeprecur.common.prep import prep_main

STAGE = "stage1_sft"

if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent, "data_blend_665k.json"))
