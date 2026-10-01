#!/usr/bin/env python3
"""OPD（四 teacher 蒸馏回发布模型）：语料准备。实现见 `stage3_opd/common/prep.py`。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.stage3_opd.common import prep_main

STAGE = "stage3_opd"


if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent))
