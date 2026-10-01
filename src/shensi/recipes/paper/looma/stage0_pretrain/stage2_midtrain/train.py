#!/usr/bin/env python3
"""中训练（Mid-1 能力强化 → Mid-2 长文档）：训练入口。

python train.py --smoke
python train.py --tokens 5e8
python train.py --config decay --load <上一段 ckpt>
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.stage0_pretrain.common import train_main

STAGE = "stage2_midtrain"

if __name__ == "__main__":
    sys.exit(train_main(STAGE, Path(__file__).resolve().parent))
