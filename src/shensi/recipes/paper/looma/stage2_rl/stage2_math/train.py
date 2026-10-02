#!/usr/bin/env python3
"""数学方向的 RL 训练入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.common.train_rl import launch_main

STAGE = "stage2_math"

if __name__ == "__main__":
    sys.exit(launch_main(STAGE, Path(__file__).resolve().parent))
