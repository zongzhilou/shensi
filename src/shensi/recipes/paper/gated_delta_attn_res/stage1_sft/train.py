#!/usr/bin/env python3
"""监督微调段（deep-thinking → hybrid → agent）：训练入口。公共核在配方 `common`，本段开关见 `stage1_sft/common/train.py`。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.stage1_sft.common import train_main

STAGE = "stage1_sft"


if __name__ == "__main__":
    sys.exit(train_main(STAGE, Path(__file__).resolve().parent))
