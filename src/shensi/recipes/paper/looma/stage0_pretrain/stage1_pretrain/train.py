#!/usr/bin/env python3
"""PT-1 stable → PT-2 decay：训练入口。

python train.py --smoke
python train.py --tokens 9e9
python train.py --config decay --load <上一段 ckpt>
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.stage0_pretrain.common import train_main

STAGE = "stage1_pretrain"

if __name__ == "__main__":
    sys.exit(train_main(STAGE, Path(__file__).resolve().parent))
