"""强化学习段的语料准备入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.common import prep_rl as prep

STAGE = "stage2_agent"
ZEN = "Agent"


if __name__ == "__main__":
    sys.exit(prep.main(STAGE, Path(__file__).resolve().parent))
