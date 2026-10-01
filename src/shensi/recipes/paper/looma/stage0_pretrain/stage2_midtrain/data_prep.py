#!/usr/bin/env python3
"""中训练语料准备：配比 json → Megatron bin/idx。

python data_prep.py --discover --config default
python data_prep.py --prepare --config tiny
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.common.prep import prep_main

STAGE = "stage2_midtrain"

if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent))
