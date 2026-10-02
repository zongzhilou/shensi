#!/usr/bin/env python3
"""SFT 语料准备入口：LLaVA-mixed-665k（或 HD 混合）→ messages jsonl。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.deeprecur.common.prep import prep_main

if __name__ == "__main__":
    sys.exit(prep_main("stage1_sft", Path(__file__).resolve().parent, "data_blend_raw.json"))
