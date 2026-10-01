#!/usr/bin/env python3
"""GDAR 配方 · stage3_opd 语料准备：学生 rollout → bin/idx（Qwen3 tokenizer）。

    python data_prep.py --prepare --blend math.json   # 某方向的学生 rollout → bin/idx
    python data_prep.py --prepare                     # 四方向均分（default.json）

学生 rollout 的 jsonl（每行含 prompt+response，或 response/text 字段）由 README ① 步产出；
teacher 的 logprob 缓存目录（README ② 步）在 train.py --teacher-cache 接，不经过这里。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common

STAGE = "stage3_opd"


def main() -> int:
    ap = argparse.ArgumentParser(description="GDAR stage3_opd 语料准备（rollout → bin/idx）")
    ap.add_argument("--discover", action="store_true", help="只看数据面貌，不产出")
    ap.add_argument("--prepare", action="store_true", help="产出 bin/idx + blend.json")
    ap.add_argument("--blend", default="default.json", help="配比 json（default=四方向均分）")
    ap.add_argument("--limit", type=int, default=None, help="每数据集最多多少条（调试用）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    paths = common.env_paths()
    blend_path = common.stage_dirs(STAGE)[1] / "data_prep" / args.blend
    blend_spec = common.load_blend_spec(blend_path)
    root = paths["data"] / STAGE / "rollouts"  # rollout jsonl 的约定落位
    out = Path(paths["data"]) / STAGE

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
        )
        print(f"[gdar] 产物在 {out}（blend.json + *_text_document.bin/.idx）")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
