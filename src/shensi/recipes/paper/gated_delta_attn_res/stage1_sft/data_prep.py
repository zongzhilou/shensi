"""监督微调段的语料准备入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res.common.prep_sft import prep_main

STAGE = "stage1_sft"


if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent))
