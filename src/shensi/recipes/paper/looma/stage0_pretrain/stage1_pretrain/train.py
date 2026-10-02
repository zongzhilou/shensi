#!/usr/bin/env python3
"""预训练档的训练入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.common.train_pt import train_main

STAGE = "stage1_pretrain"

if __name__ == "__main__":
    sys.exit(train_main(STAGE, Path(__file__).resolve().parent))
