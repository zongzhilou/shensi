"""agentic 入口（真机档 / Sim 档）。"""

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import rl

STAGE = "stage2_agentic"

if __name__ == "__main__":
    raise SystemExit(rl.launch(STAGE))
