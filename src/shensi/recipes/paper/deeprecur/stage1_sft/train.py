#!/usr/bin/env python3
"""SFT（instruction tuning）：训练入口。

python train.py --smoke
python train.py --profile debug                      # 真实数据小切片 + tiny 几何
python train.py --tokens 0.9e9                       # 论文口径：665k × 1 epoch（估）
python train.py --config sft_v --tokens 1.1e9        # DeepStack-V/HD 变体（视觉编码器也训）
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.deeprecur.common.train import train_main

STAGE = "stage1_sft"

if __name__ == "__main__":
    sys.exit(train_main(STAGE, Path(__file__).resolve().parent))
