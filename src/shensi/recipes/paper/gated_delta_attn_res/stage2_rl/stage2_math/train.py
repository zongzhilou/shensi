#!/usr/bin/env python3
"""GDAR 配方 · 数学 RL teacher（verl GRPO + Megatron actor，从 SFT-2 ckpt 起）。

复用 shensi 配方的 rl.launch（同一套 yaml → verl CLI 映射与启动）：

    python train.py --dry-run
    python train.py --load <SFT-2 ckpt>     # 起训（数据先 data_prep.py --prepare）

GDAR 在 verl 侧的注册在 `stage2_rl/gdar_bridge.py`（导入即注册：model_type → 连接层规格
provider → 由 convert/tables.py 生成的权重表）；run_verl 用 VERL_USE_EXTERNAL_MODULES 让每个
verl 进程都加载它。端到端闸门：`python -m ...stage2_rl.test_gdar_bridge`。
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res import common
from shensi.recipes.shensi import rl

STAGE = "stage2_math"


def main() -> int:
    here = Path(__file__).resolve().parent
    # 起点的模型类型决定 rollout 驱动：命中外部 harness 族时打开多轮 + 工具配置，
    # 其余档不加任何覆盖（默认单轮纯模型）。
    cfg = rl._load_with_base(here / "config/default.yaml")
    model_path = (cfg.get("model") or {}).get("path")
    for item in sys.argv[1:]:  # --set model.path=… 与 rl.launch 的覆写同序：CLI 优先
        if isinstance(item, str) and item.startswith("model.path="):
            model_path = item.partition("=")[2]
    over = common.agent_overrides(
        model_path, tool_config=str(here.parent / "config/tools/harness.yaml")
    )
    if cfg.get("critic"):  # critic 档：起点与 actor 同源，直接注入（避免子档插值）
        over = [*over, f"critic.model.path={model_path}"]
    argv: list[str] = list(sys.argv[1:])  # 用户给的 --dry-run/--set/--profile 照旧透传
    for item in over:
        argv += ["--set", item]
    dry = "--dry-run" in argv
    rc = (
        common.run_verl(STAGE, here, here / "reward.py", argv, watch=None)
        if dry
        else common.run_verl(
            STAGE, here, here / "reward.py", argv, watch=common.early_stop_plan("stage2_rl", cfg)
        )
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
