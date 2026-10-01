"""对齐语料准备（判分规格进 ground_truth）。"""

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import rl

STAGE = "stage3_align"

if __name__ == "__main__":
    raise SystemExit(rl.prepare(STAGE))
