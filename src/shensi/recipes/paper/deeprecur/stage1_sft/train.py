#!/usr/bin/env python3
"""SFT 训练入口：instruction tuning（解冻 LLM + projector）。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.deeprecur.common.train import train_main

if __name__ == "__main__":
    sys.exit(train_main("stage1_sft", Path(__file__).resolve().parent))
