#!/usr/bin/env python3
"""PT-1/PT-2 语料准备：配比 json → Megatron bin/idx。

python data_prep.py --discover --config default
python data_prep.py --prepare --config tiny
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.stage0_pretrain.common import prep_main

STAGE = "stage1_pretrain"

if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent))
