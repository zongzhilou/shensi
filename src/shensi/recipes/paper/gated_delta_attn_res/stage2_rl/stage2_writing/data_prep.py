#!/usr/bin/env python3
"""写作方向的 RL 语料准备：prompts jsonl → verl 的 train/val parquet。
配比与参数在 `config/data_prep/`；实现见 `stage2_rl/common/prep.py`。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.stage2_rl.common import prep

STAGE = "stage2_writing"
ZEN = "写作"


if __name__ == "__main__":
    sys.exit(prep.main(STAGE, Path(__file__).resolve().parent))
