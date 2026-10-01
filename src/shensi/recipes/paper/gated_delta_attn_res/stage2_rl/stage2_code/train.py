"""强化学习段的训练入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.stage2_rl.common import launch_main

STAGE = "stage2_code"


if __name__ == "__main__":
    sys.exit(launch_main(STAGE, Path(__file__).resolve().parent))
