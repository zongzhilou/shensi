#!/usr/bin/env python3
"""RL 式 OPD 的启动器：学生 rollout + teacher 在线打分（reward = −reverse KL）。

与 `stage2_rl/*/train.py` 同一套 verl 栈与启动器（命令组装、进程环境、早停看门狗都在
`common.run_verl` 里），差别只有两处：config 是本目录的 `config/opd_rl.yaml`，reward 是本目录的
`opd_reward.py`（学生与 teacher 的 logprob 由两个 vLLM 端点提供，见该文件 docstring）。

    python opd_rl.py --dry-run                 # 打印 verl 命令（不连端点）
    python opd_rl.py                            # 起训（需要 OPD_STUDENT_URL / OPD_TEACHER_URL）
    python opd_rl.py --set model.path=<学生 HF 目录> --set data.train_batch_size=64
"""

from __future__ import annotations

import sys
from pathlib import Path

from shensi import runtime  # noqa: F401  导入即登记第三方要的东西
from shensi.recipes.paper.gated_delta_attn_res import common
from shensi.recipes.shensi import rl

STAGE = "stage3_opd_rl"


def main() -> int:
    here = Path(__file__).resolve().parent
    cfg_path = here / "config/opd_rl.yaml"

    # 学生起点决定 rollout 驱动：与 RL 各臂同一条规则（命中外部 harness 族时开多轮 + 工具配置）
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
