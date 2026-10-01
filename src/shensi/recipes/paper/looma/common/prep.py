"""语料准备：配比 json → Megatron bin/idx。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.shensi.common import common as base

from .config import dataprep_config, profile_from_args
from .paths import env_paths, stage_dirs


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
    """按配比产出 bin/idx；``discover=True`` 只打印数据面貌。"""
    paths = env_paths()
    blend_path = stage_dirs(stage) / "config" / "data_prep" / blend
    if not blend_path.is_file():
        raise SystemExit(f"[looma] 没有这个配比文件：{blend_path}")
    blend_spec = base.load_blend_spec(blend_path)
    out = Path(data_dir or paths["data"] / stage)
    if discover:
        report = base.discover(paths["pre"], blend_spec, out)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0
    base.prepare(
        blend_spec,
        paths["pre"],
        out,
        tokenizer=paths["tokenizer"],
        limit=limit,
        workers=workers,
        only=only,
    )
    print(f"[looma] 产物在 {out}（blend.json + *_text_document.bin/.idx）")
    return 0


def prep_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    """命令行入口：``--discover`` / ``--prepare`` 二选一，参数缺省从 data_prep 配置取。"""
    parser = argparse.ArgumentParser(description=f"{stage} 语料准备（bin/idx）")
    parser.add_argument("--discover", action="store_true", help="只看数据面貌，不产出")
    parser.add_argument("--prepare", action="store_true", help="产出 bin/idx + blend.json")
    parser.add_argument("--config", default=None, help="config/data_prep/<名字>.yaml")
    parser.add_argument("--blend", default=None, help="配比 json（相对 config/data_prep/）")
    parser.add_argument("--limit", type=int, default=None, help="每数据集最多多少条")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--only", default=None, help="只处理名字含该子串的数据集")
    parser.add_argument("--data-dir", default=None, help="产物目录")
    args = parser.parse_args(argv)
    if not (args.discover or args.prepare):
        parser.print_help()
        return 1
    name = profile_from_args(args.config, "default", stage)
    cfg = dataprep_config(here / "config" / "data_prep" / f"{name}.yaml")
    return prepare(
        stage=stage,
        blend=args.blend or cfg.get("blend") or "data_blend_raw.json",
        limit=args.limit if args.limit is not None else cfg.get("limit"),
        workers=args.workers if args.workers is not None else (cfg.get("workers") or 8),
        only=args.only if args.only is not None else cfg.get("only"),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        discover=args.discover,
    )
