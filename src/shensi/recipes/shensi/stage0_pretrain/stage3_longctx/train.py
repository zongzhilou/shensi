#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402

STAGE = "stage3_longctx"


def main() -> int:
    ap = argparse.ArgumentParser(description="Shensi stage3_longctx（128K → 1M 长上下文）")
    ap.add_argument("--profile", default="default", choices=("default", "1m"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tokens", type=int, default=None, help="token 预算（报告 128K 段 500e9、1M 段 50e9）")
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
