#!/usr/bin/env python3
"""写作方向的 RL teacher（verl + Megatron actor）。用法见 `stage2_rl/README.md`；
公共流程在 `stage2_rl/common/launch.py`。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.stage2_rl.common import launch_main

STAGE = "stage2_writing"


if __name__ == "__main__":
    sys.exit(launch_main(STAGE, Path(__file__).resolve().parent))
