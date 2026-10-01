#!/usr/bin/env python3
"""stage1_pretrain 的集成测试：tiny 几何 + 真实 bin/idx 数据，跑 5 步并校验收尾。

判据见 `shensi.recipes.shensi.common.tiny_test.run_stage_tiny`：rc=0、跑到最后一次 iteration、
出现 `[after training is done]`、无 Traceback。

跑法：
    cd stage0_pretrain/stage1_pretrain && python test_train.py
    python test_train.py --profile adamw --iters 3   # 换档（adamw / lion / muon / ademamix）；
                                                  # 优化器档的 train_iters 是生产值，冒烟要 --iters 压住
    python test_train.py --mtp 1                  # 带上 1 层 MTP（>1 层见 recipes README 的局限）
"""

import argparse

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import tiny_test

STAGE = "stage1_pretrain"


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Shensi {STAGE} 集成测试（tiny 几何）")
    ap.add_argument("--profile", default="debug")
    ap.add_argument(
        "--iters", type=int, default=None, help="覆盖 train_iters（默认用 profile 自己的值）"
    )
    ap.add_argument(
        "--mtp",
        type=int,
        default=0,
        help="MTP 层数（0/1/2；可与 mHC 同开，极小档实跑过 1 层与 2 层）",
    )
    ap.add_argument("--set", dest="override", action="append", default=[])
    args = ap.parse_args()
    return tiny_test.run_stage_tiny(
        STAGE, args.profile, override=args.override, iters=args.iters, mtp_layers=args.mtp
    )


if __name__ == "__main__":
    raise SystemExit(main())
