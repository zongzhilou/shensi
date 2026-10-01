#!/usr/bin/env python3
"""OPD（四 teacher 蒸馏回发布模型）：训练入口。公共核在配方 `common`，本段开关见 `stage3_opd/common/train.py`。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.stage3_opd.common import train_main

STAGE = "stage3_opd"


if __name__ == "__main__":
    sys.exit(train_main(STAGE, Path(__file__).resolve().parent))
