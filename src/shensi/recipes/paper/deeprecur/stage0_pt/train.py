#!/usr/bin/env python3
"""PT（feature alignment）：训练入口。

python train.py --smoke
python train.py --profile debug          # 真实数据小切片 + tiny 几何
python train.py --tokens 0.14e9          # 论文口径：LCS-558k × 1 epoch ≈ 0.14B tokens
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.deeprecur.common.train import train_main

STAGE = "stage0_pt"

if __name__ == "__main__":
    sys.exit(train_main(STAGE, Path(__file__).resolve().parent))
