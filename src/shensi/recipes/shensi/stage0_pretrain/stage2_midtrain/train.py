#!/usr/bin/env python3
import argparse
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi import common

STAGE = "stage2_midtrain"


def main() -> int:
    ap = argparse.ArgumentParser(description="Shensi stage2_midtrain（DSA 引入 + 中训练）")
    # 档位不写死：config/ 下每份 yaml 都是一档（debug.yaml 也在内），名字给错由 build_config 报清楚
    ap.add_argument("--profile", default="default")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--tokens", type=int, default=None, help="token 预算（sparse adaptation 报告用 20e9）"
    )
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--set", dest="override", action="append", default=[])
    ap.add_argument("--early-stop", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        return common.smoke()
    paths = common.env_paths()
    data_dir = Path(args.data_dir or paths["data"] / STAGE)
    cfg = common.build_config(STAGE, args.profile, args.override, data_dir, tokens=args.tokens)
    rc = common.run(cfg, STAGE, args.profile, args.dry_run)
    if args.early_stop and not args.dry_run:
        rc = common.watch(cfg, args.early_stop)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
