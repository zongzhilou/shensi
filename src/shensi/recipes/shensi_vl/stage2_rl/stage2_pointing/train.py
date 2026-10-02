#!/usr/bin/env python3
"""pointing 家族 RL 入口（F_TwP → E_TwP）。"""

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi_vl.stage2_rl import launch

if __name__ == "__main__":
    raise SystemExit(launch.launch("stage2_pointing"))
