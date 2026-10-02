"""预训练档的训练入口（stable 与 decay 共用）。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.common.train_pt import train_main

STAGE = "stage1_pretrain"


if __name__ == "__main__":
    sys.exit(train_main(STAGE, Path(__file__).resolve().parent))
