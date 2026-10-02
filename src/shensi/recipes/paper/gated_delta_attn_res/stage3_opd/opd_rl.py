"""OPD 段的 opd_rl.py 模块。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res import common
from shensi.recipes.shensi.common import rl

STAGE = "stage3_opd_rl"


def main() -> int:
    here = Path(__file__).resolve().parent
    cfg_path = here / "config/opd_rl.yaml"

    cfg = rl._load_with_base(cfg_path)
    model_path = (cfg.get("model") or {}).get("path")
    for item in sys.argv[1:]:
        if isinstance(item, str) and item.startswith("model.path="):
            model_path = item.partition("=")[2]
    argv: list[str] = list(sys.argv[1:])
    for item in common.agent_overrides(
        model_path, tool_config=str(here.parent / "stage2_rl/config/tools/harness.yaml")
    ):
        argv += ["--set", item]

    dry = "--dry-run" in argv
    watch = None if dry else common.early_stop_plan("stage2_rl", cfg)
    argv = [*argv, "--config", str(cfg_path)]
    return common.run_verl(STAGE, here, here / "opd_reward.py", argv, watch=watch)


if __name__ == "__main__":
    sys.exit(main())
