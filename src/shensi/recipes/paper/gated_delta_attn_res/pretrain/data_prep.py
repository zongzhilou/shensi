#!/usr/bin/env python3
"""GDAR 配方 · 语料准备（bin/idx，Qwen3 tokenizer）。

用法与 shensi 配方一致（复用 recipes/shensi/common 的 discover/prepare 实现）：

    python data_prep.py --discover                 # 看数据面貌（有哪些文件、缺什么）
    python data_prep.py --prepare                  # 全量混合 → bin/idx + blend.json
    python data_prep.py --prepare --limit 1000     # 调试档（每数据集 1000 条）

配比在 `config/data_prep/{default,debug_sample}.json`；产物（bin/idx/blend.json）落在
`$SHENSI_FS/shensi/data/gated_delta_attn_res/`，`train.py` 会自动拾取那里的 blend.json。
tokenizer 固定 Qwen3 同款（`$SHENSI_GDAR_TOKENIZER` 可覆盖，默认配方根 tokenizer/Qwen3-0.6B）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common


def main() -> int:
    ap = argparse.ArgumentParser(description="GDAR 配方语料准备（bin/idx）")
    ap.add_argument("--discover", action="store_true", help="只看数据面貌，不产出")
    ap.add_argument("--prepare", action="store_true", help="产出 bin/idx + blend.json")
    ap.add_argument("--blend", default=None, help="配比 json（默认 config/data_prep/default.json）")
    ap.add_argument("--limit", type=int, default=None, help="每数据集最多多少条（调试用）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="只处理名字含这个子串的数据集")
    args = ap.parse_args()

    paths = common.env_paths()
    blend_path = common.CONFIG / "data_prep" / (args.blend or "default.json")
    blend_spec = common.load_blend_spec(blend_path)
    root = paths["pre"]
    out = Path(paths["data"])

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
