#!/usr/bin/env python3
import argparse
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common

STAGE = "stage3_longctx"


def main() -> int:
    ap = argparse.ArgumentParser(description="Shensi stage3_longctx（128K → 1M 长上下文）")
    # 档位不写死：config/ 下每份 yaml 都是一档（debug.yaml 也在内），名字给错由 build_config 报清楚
    ap.add_argument("--profile", default="default")
    ap.add_argument(
        "--config",
        default=None,
        help="直接给配置档路径（与 --profile 等价，例：config/tiny.yaml）",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--tokens", type=int, default=None, help="token 预算（报告 128K 段 500e9、1M 段 50e9）"
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--set", dest="override", action="append", default=[])
    ap.add_argument(
        "--early-stop",
        type=int,
        default=3,
        help="早停耐心（验证指标连续多少次不改善就收尾；默认 3，0 或负数=不看门狗）",
    )
    ap.add_argument(
        "--no-early-stop",
        action="store_true",
        help="关掉早停看门狗（按 profile 的 train_iters 跑满）",
    )
    ap.add_argument(
        "--early-stop-grace",
        type=float,
        default=600.0,
        help="宽限秒数：这段时间内不判耐心（跑过预热再判）",
    )
    args = ap.parse_args()
    if args.config:
        args.profile = args.config
    if args.smoke:
        return common.smoke()
    paths = common.env_paths()
    data_dir = Path(args.data_dir or paths["data"] / STAGE)
    cfg = common.build_config(STAGE, args.profile, args.override, data_dir, tokens=args.tokens)
    patience = 0 if args.no_early_stop else args.early_stop
    return common.run(
        cfg,
        STAGE,
        args.profile,
        args.dry_run,
        watch=common.watchdog_spec(patience, metric="lm loss value", grace=args.early_stop_grace),
    )


if __name__ == "__main__":
    raise SystemExit(main())
