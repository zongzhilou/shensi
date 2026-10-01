"""OPD 段的语料准备入口。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi.recipes.paper.looma.stage3_opd.common import prep_main

STAGE = "stage3_opd"


if __name__ == "__main__":
    sys.exit(prep_main(STAGE, Path(__file__).resolve().parent))
