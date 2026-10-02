#!/usr/bin/env python3
"""写作方向的 RL 训练入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.common.launch_rl import launch_main

STAGE = "stage2_writing"

if __name__ == "__main__":
    sys.exit(launch_main(STAGE, Path(__file__).resolve().parent))
