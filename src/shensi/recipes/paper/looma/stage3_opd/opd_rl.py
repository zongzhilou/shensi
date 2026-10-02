"""RL 式 OPD 的启动入口：读 ``config/opd_rl.yaml``，按域路由 teacher 拉起 verl。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.looma import common
from shensi.recipes.shensi.common import rl

STAGE = "stage3_opd_rl"


def main() -> int:
    """入口：配置 + agent 覆写 → 带早停看门狗拉起 verl（reward 用本目录的 opd_reward.py）。"""
    here = Path(__file__).resolve().parent
    cfg_path = here / "config/opd_rl.yaml"

    cfg = rl._load_with_base(cfg_path)
    model_path = (cfg.get("model") or {}).get("path")
    for item in sys.argv[1:]:
        if isinstance(item, str) and item.startswith("model.path="):
            model_path = item.partition("=")[2]
    argv: list[str] = list(sys.argv[1:])
    for item in common.agent_overrides(
        model_path, tool_config=str(here.parent / "stage2_rl" / "config" / "tools" / "harness.yaml")
    ):
        argv += ["--set", item]

    watch = None if "--dry-run" in argv else common.early_stop_plan("stage2_rl", cfg)
    argv = [*argv, "--config", str(cfg_path)]
    return common.run_verl(STAGE, here, here / "opd_reward.py", argv, watch=watch)


if __name__ == "__main__":
    raise SystemExit(main())
