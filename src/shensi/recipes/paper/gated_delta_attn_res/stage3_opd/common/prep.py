"""OPD 语料准备：学生 rollout 的 jsonl → Megatron bin/idx。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common


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
    spec = common.load_blend_spec(blend_path)
    root = paths["data"] / stage / "rollouts"
    out = Path(data_dir or paths["data"] / stage)
    if not root.is_dir():
        raise SystemExit(f"[gdar] 没有 rollout 目录：{root}（先跑 rollout.py 产 jsonl）")
    if discover:
        print(
            json.dumps(
                common.base.discover(root, spec, out), ensure_ascii=False, indent=2, default=str
            )
        )
        return 0
    common.base.prepare(
        spec, root, out, tokenizer=paths["tokenizer"], limit=limit, workers=workers, only=only
    )
    print(f"[gdar] 产物在 {out}（blend.json + *_text_document.bin/.idx）")
    return 0


def prep_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{stage} 语料准备（学生 rollout → bin/idx）")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--config", default=None, help="config/data_prep/<名字>.yaml")
    ap.add_argument("--blend", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    if not (args.discover or args.prepare):
        ap.print_help()
        return 1
    name = common.profile_from_args(args.config, "default", stage)
    cfg = common.dataprep_config(here / "config/data_prep" / f"{name}.yaml")
    return prepare(
        stage=stage,
        blend=args.blend or cfg.get("blend") or "data_blend_raw.json",
        limit=args.limit if args.limit is not None else cfg.get("limit"),
        workers=args.workers if args.workers is not None else (cfg.get("workers") or 8),
        only=args.only if args.only is not None else cfg.get("only"),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        discover=args.discover,
    )
