"""预训练段共用的语料准备实现。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.paper.looma.common import gdar as common


def prepare(
    *,
    stage: str,
    blend: str = "data_blend_raw.json",
    limit: int | None = None,
    workers: int = 8,
    only: str | None = None,
    data_dir: Path | None = None,
    discover: bool = False,
) -> int:
    paths = common.env_paths()
    blend_path = common.stage_dirs(stage)[1] / "data_prep" / blend
    if not blend_path.is_file():
        raise SystemExit(f"[gdar] 没有这个配比文件：{blend_path}")
    blend_spec = common.load_blend_spec(blend_path)
    out = Path(data_dir or paths["data"] / stage)
    if discover:
        print(
            json.dumps(
                common.base.discover(paths["pre"], blend_spec, out),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0
    common.base.prepare(
        blend_spec,
        paths["pre"],
        out,
        tokenizer=paths["tokenizer"],
        limit=limit,
        workers=workers,
        only=only,
    )
    print(f"[gdar] 产物在 {out}（blend.json + *_text_document.bin/.idx）")
    return 0


def prep_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{stage} 语料准备（bin/idx）")
    ap.add_argument("--discover", action="store_true", help="只看数据面貌，不产出")
    ap.add_argument("--prepare", action="store_true", help="产出 bin/idx + blend.json")
    ap.add_argument("--config", default=None, help="config/data_prep/<名字>.yaml")
    ap.add_argument("--blend", default=None, help="配比 json（相对 config/data_prep/）")
    ap.add_argument("--limit", type=int, default=None, help="每数据集最多多少条（调试用）")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--only", default=None, help="只处理名字含该子串的数据集")
    ap.add_argument(
        "--data-dir",
        default=None,
        help="产物目录（默认 <FS>/shensi/data/gated_delta_attn_res/<stage>）",
    )
    args = ap.parse_args(argv)
    if not (args.discover or args.prepare):
        ap.print_help()
        return 1
    cfg = common.dataprep_config_for(here, args.config, args.blend)
    return prepare(
        stage=stage,
        blend=args.blend or cfg.get("blend") or "data_blend_raw.json",
        limit=args.limit if args.limit is not None else cfg.get("limit"),
        workers=args.workers if args.workers is not None else (cfg.get("workers") or 8),
        only=args.only if args.only is not None else cfg.get("only"),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        discover=args.discover,
    )
