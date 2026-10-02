#!/usr/bin/env python3
"""写作方向的语料准备入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.common.prep_rl import prep_main

STAGE = "stage2_writing"

if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent))
