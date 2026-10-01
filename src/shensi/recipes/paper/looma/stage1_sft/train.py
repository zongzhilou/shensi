#!/usr/bin/env python3
"""SFT（deep-thinking → hybrid → agent）：训练入口。

python train.py --smoke
python train.py --tokens 2e9 --load <中训练 ckpt>
python train.py --config sft3_agent --load <SFT-2 ckpt>
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.common.train_sft import train_main

STAGE = "stage1_sft"

if __name__ == "__main__":
    sys.exit(train_main(STAGE, Path(__file__).resolve().parent))
