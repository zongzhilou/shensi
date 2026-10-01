#!/usr/bin/env python3
"""stage2_agent：RL teacher 训练入口（verl GRPO + Megatron actor）。

python train.py --dry-run
python train.py --load <SFT ckpt>
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.common.launch_rl import launch_main

STAGE = "stage2_agent"

if __name__ == "__main__":
    sys.exit(launch_main(STAGE, Path(__file__).resolve().parent))
