from shensi import runtime  # noqa: F401
from shensi.recipes.shensi import rl

STAGE = "stage1_rlvr"

if __name__ == "__main__":
    raise SystemExit(rl.prepare(STAGE))
