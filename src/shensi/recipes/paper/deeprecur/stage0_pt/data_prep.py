#!/usr/bin/env python3
"""PT 语料准备入口：LCS-558k → messages jsonl。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.deeprecur.common.prep import prep_main

if __name__ == "__main__":
    sys.exit(prep_main("stage0_pt", Path(__file__).resolve().parent, "data_blend_raw.json"))
