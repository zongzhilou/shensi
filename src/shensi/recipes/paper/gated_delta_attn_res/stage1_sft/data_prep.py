#!/usr/bin/env python3
"""监督微调段（deep-thinking → hybrid → agent）：语料准备。实现见 `stage1_sft/common/prep.py`。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.stage1_sft.common import prep_main

STAGE = "stage1_sft"


if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent))
