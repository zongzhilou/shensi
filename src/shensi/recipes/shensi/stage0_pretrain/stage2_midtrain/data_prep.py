#!/usr/bin/env python3
"""中训练语料准备（继承上一段配比）。"""

import argparse
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common

STAGE = "stage2_midtrain"


def main() -> int:
    ap = argparse.ArgumentParser(description="Shensi stage2_midtrain 语料准备")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--include-metadata-only", action="store_true")
    ap.add_argument(
        "--blend", default=None, help="换一份配比 json（默认 config/data_prep/data_blend_raw.json）"
    )
    common.add_common_args(ap)
    args = ap.parse_args()
    args = common.resolve_prep_config(args, Path(__file__).parent)
    if not (args.discover or args.prepare):
        ap.error("至少给一个：--discover / --prepare")
    paths = common.env_paths()
    root = Path(args.root or paths["pre"])
    out = Path(args.out or paths["data"] / STAGE)
    spec = common.load_blend_spec(
        Path(args.blend)
        if args.blend
        else Path(__file__).parent / "config/data_prep/data_blend_raw.json"
    )
    if args.discover:
        print(f"[data_prep] 语料根：{root}（stage2 口径：长文档优先，见 min_chars）")
        common.discover(root, spec)
    if args.prepare:
        common.prepare(
            spec,
            root,
            out,
            args.tokenizer or paths["tokenizer"],
            args.limit,
            args.workers,
            args.include_metadata_only,
            args.only,
            args.skip_missing,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
