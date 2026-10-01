#!/usr/bin/env python3
"""预训练段（PT-1 stable → PT-2 decay）：训练入口。

python train.py --config config/default.yaml            # 或 --profile default
python train.py --smoke                                 # tiny 档 5 步
python train.py --tokens 9e9 --model-algo qwen3_gdar_paper
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.stage0_pretrain.common import train_main

STAGE = "stage1_pretrain"


if __name__ == "__main__":
    sys.exit(train_main(STAGE, Path(__file__).resolve().parent))
