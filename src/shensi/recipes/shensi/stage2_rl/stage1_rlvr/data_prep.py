"""RLVR 语料准备（rl.prepare 的薄封装）。"""

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import rl

STAGE = "stage1_rlvr"

if __name__ == "__main__":
    raise SystemExit(rl.prepare(STAGE))
