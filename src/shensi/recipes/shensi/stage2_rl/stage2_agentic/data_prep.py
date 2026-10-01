"""agentic 语料准备（带 agent_ref 与 verifier）。"""

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import rl

STAGE = "stage2_agentic"

if __name__ == "__main__":
    raise SystemExit(rl.prepare(STAGE))
