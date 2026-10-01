#!/usr/bin/env python3
"""主预训练语料准备（--discover / --prepare / --codev3）。"""

import argparse
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import codev3, common

STAGE = "stage1_pretrain"


def main() -> int:
    ap = argparse.ArgumentParser(description="Shensi stage1_pretrain 语料准备")
    ap.add_argument("--discover", action="store_true", help="只扫描并打印语料面貌")
    ap.add_argument(
        "--blend", default=None, help="换一份配比 json（默认 config/data_prep/data_blend_raw.json）"
    )
    ap.add_argument("--prepare", action="store_true", help="产出 .bin/.idx 与 blend.json")
    ap.add_argument(
        "--include-metadata-only",
        action="store_true",
        help="把'只有元数据'的数据集当硬报错（提醒先去回捞文本）",
    )
    ap.add_argument(
        "--hf-sample", type=int, default=None, help="配合 --codev3：v1/v2/v3 各取这么多行元数据"
    )
    codev3.add_args(ap)
    common.add_common_args(ap)
    args = ap.parse_args()
    args = common.resolve_prep_config(args, Path(__file__).parent)

    paths = common.env_paths()
    root = Path(args.root or paths["pre"])
    out = Path(args.out or paths["data"] / STAGE)
    if args.codev3:
        args.out = str(out)
        codev3.run(args)
        return 0
    if not (args.discover or args.prepare):
        ap.error("至少给一个：--discover / --prepare / --codev3")

    spec = common.load_blend_spec(
        Path(args.blend)
        if args.blend
        else Path(__file__).parent / "config/data_prep/data_blend_raw.json"
    )
    if args.discover:
        print(f"[data_prep] 语料根：{root}")
        common.discover(root, spec, out)
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
