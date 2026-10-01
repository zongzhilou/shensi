#!/usr/bin/env python3
"""GDAR 配方 · stage1_pretrain 语料准备（bin/idx，Qwen3 tokenizer）。

    python data_prep.py --discover                    # 看数据面貌
    python data_prep.py --prepare                     # Mid-1 能力强化混合
    python data_prep.py --prepare --blend decay.json  # Mid-2 长文档混合
    python data_prep.py --prepare --limit 1000        # 调试档

产物落在 `$SHENSI_FS/shensi/data/gated_delta_attn_res/stage1_pretrain/`，train.py 自动拾取。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common

STAGE = "stage2_midtrain"


def main() -> int:
    ap = argparse.ArgumentParser(description="GDAR stage2_midtrain 语料准备（bin/idx）")
    ap.add_argument("--discover", action="store_true", help="只看数据面貌，不产出")
    ap.add_argument("--prepare", action="store_true", help="产出 bin/idx + blend.json")
    ap.add_argument(
        "--blend",
        default="default.json",
        help="配比 json（default=Mid-1 能力强化，mid2.json=Mid-2 长文档）",
    )
    ap.add_argument("--limit", type=int, default=None, help="每数据集最多多少条（调试用）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--data-dir",
        default=None,
        help="产物目录（默认 <FS>/shensi/data/gated_delta_attn_res/<stage>）",
    )
    ap.add_argument("--only", default=None, help="只处理名字含这个子串的数据集")
    args = ap.parse_args()

    paths = common.env_paths()
    blend_path = common.stage_dirs(STAGE)[1] / "data_prep" / args.blend
    blend_spec = common.load_blend_spec(blend_path)
    root = paths["pre"]
    out = Path(args.data_dir or paths["data"] / STAGE)

    if args.discover:
        report = common.base.discover(root, blend_spec, out)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.prepare:
        common.base.prepare(
            blend_spec,
            root,
            out,
            tokenizer=paths["tokenizer"],
            limit=args.limit,
            workers=args.workers,
            only=args.only,
        )
        print(f"[gdar] 产物在 {out}（blend.json + *_text_document.bin/.idx）")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
