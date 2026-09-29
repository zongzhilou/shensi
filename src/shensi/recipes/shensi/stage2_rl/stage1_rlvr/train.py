#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rl_common  # noqa: E402

STAGE = "stage1_rlvr"

if __name__ == "__main__":
    raise SystemExit(rl_common.launch(STAGE))
