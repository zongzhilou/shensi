#!/usr/bin/env python3
"""PT 训练入口：feature alignment（只训 projector）。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.deeprecur.common.train import train_main

if __name__ == "__main__":
    sys.exit(train_main("stage0_pt", Path(__file__).resolve().parent))
