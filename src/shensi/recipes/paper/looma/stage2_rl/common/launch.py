"""RL 各臂共用的启动流程（verl GRPO + Megatron actor）。"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.looma import common
from shensi.recipes.shensi.common import rl

__all__ = ["main"]


def main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    """起一次 RL；起点的模型类型决定 rollout 驱动（命中工具族时开多轮 + 工具配置）。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = rl._load_with_base(here / "config/default.yaml")
    model_path = (cfg.get("model") or {}).get("path")
    for item in argv:
        if isinstance(item, str) and item.startswith("model.path="):
            model_path = item.partition("=")[2]
    explicit = {item.split("=", 1)[0] for item in argv if isinstance(item, str) and "=" in item}
    overrides = common.agent_overrides(
        model_path, tool_config=str(here.parent / "config" / "tools" / "harness.yaml")
    )
    # 显式选择优先：命令行给过的键，自动接线的值不再盖上去。
    overrides = [item for item in overrides if item.split("=", 1)[0] not in explicit]
    if cfg.get("critic"):
        overrides = [*overrides, f"critic.model.path={model_path}"]
    for item in overrides:
        argv += ["--set", item]
    watch = None if "--dry-run" in argv else common.early_stop_plan("stage2_rl", cfg)
    return common.run_verl(stage, here, here / "reward.py", argv, watch=watch)
