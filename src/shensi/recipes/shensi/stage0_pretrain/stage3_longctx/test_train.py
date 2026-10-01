#!/usr/bin/env python3
"""stage3_longctx 的集成测试：tiny 几何 + 本 stage 的档，跑 5 步并校验收尾。

比前两段多覆盖的东西：长序列的 YaRN 位置编码（tiny 档把长度压到 128，但走的是同一套 rope 参数路径）与 `context_parallel_size` 的配置面；数据同 stage2。

跑法：`cd stage0_pretrain/stage3_longctx && python test_train.py`
"""

import argparse

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi import tiny_test

STAGE = "stage3_longctx"


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Shensi {STAGE} 集成测试（tiny 几何）")
    ap.add_argument("--profile", default="debug")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--mtp", type=int, default=0, help="MTP 层数（我们的 MTP 目前只支持 0/1）")
    ap.add_argument("--set", dest="override", action="append", default=[])
    args = ap.parse_args()
    return tiny_test.run_stage_tiny(
        STAGE, args.profile, override=args.override, iters=args.iters, mtp_layers=args.mtp
    )


if __name__ == "__main__":
    raise SystemExit(main())
