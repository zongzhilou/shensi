#!/usr/bin/env python3
"""中训练段（能力强化 → 长文档）：语料准备（配比 json → Megatron bin/idx）。

python data_prep.py --discover --config default
python data_prep.py --prepare --config tiny             # 小样本，几分钟出真实 bin/idx
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.stage0_pretrain.common import prep_main

STAGE = "stage2_midtrain"


if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent))
