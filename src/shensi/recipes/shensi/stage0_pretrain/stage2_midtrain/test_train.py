#!/usr/bin/env python3
"""stage2_midtrain 的集成测试：tiny 几何 + 中训档（稀疏路径 + DSA indexer loss），跑 5 步。

比 stage1 多覆盖的东西：`csa_dense_mode=false`（indexer 真的在算）、`dsa_warmup.yaml` 的
冻主干语义（`--profile dsa_warmup` 时）。

跑法：`cd stage0_pretrain/stage2_midtrain && python test_train.py [--profile dsa_warmup]`
"""

import argparse

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import tiny_test

STAGE = "stage2_midtrain"


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
