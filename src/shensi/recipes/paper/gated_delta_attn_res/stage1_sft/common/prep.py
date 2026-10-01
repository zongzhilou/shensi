"""SFT 语料准备：UltraData-SFT 的 parquet/jsonl → messages jsonl（98/2 切）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common

#: 各子段的输出文件名（`--config agent` → `sft_train_agent.jsonl`）
SUFFIX = {"default": "", "hybrid": "_hybrid", "agent": "_agent"}


def _rows_from(spec: dict, root: Path, limit: int | None) -> list[dict]:
    import pandas as pd

    rows: list[dict] = []
    for item in spec.get("datasets", []):
        name = item["name"]
        sub = item.get("config")
        d = root / name / sub if sub else root / name
        if not d.is_dir():
            d = root / name
        if not d.is_dir():
            continue
        for f in sorted(d.glob("**/*")):
            if f.suffix == ".parquet":
                df = pd.read_parquet(f)
                for _, r in df.iterrows():
                    rows.append(
                        {"messages": list(r["messages"])}
                        if "messages" in df.columns
                        else {"messages": json.loads(r.to_json())["messages"]}
                    )
            elif f.suffix in (".jsonl", ".json"):
                with open(f, encoding="utf-8") as fh:
                    for ln in fh:
                        if ln.strip():
                            row = json.loads(ln)
                            if "messages" in row:
                                rows.append({"messages": row["messages"]})
            if limit and len(rows) >= limit:
                return rows[:limit]
    return rows


def prepare(
    *,
    stage: str,
    blend: str = "data_blend_raw.json",
    limit: int | None = None,
    val_frac: float = 0.02,
    root: Path | None = None,
    out_dir: Path | None = None,
    suffix: str = "",
) -> int:
    """写 `sft_train<suffix>.jsonl` / `sft_val<suffix>.jsonl`，返回条数。"""
    paths = common.env_paths()
    root = Path(root or paths["post"])
    blend_path = common.stage_dirs(stage)[1] / "data_prep" / blend
    spec = common.load_blend_spec(blend_path)
    rows = _rows_from(spec, root, limit)
    if not rows:
        raise SystemExit(
            f"[gdar] 没有可用 SFT 数据：{root}（配比 {blend}）。语料落位或 --root 指对后重跑。"
        )
    out = Path(out_dir or paths["data"] / stage)
    out.mkdir(parents=True, exist_ok=True)
    n_val = max(1, int(len(rows) * val_frac))
    for name, subset in (
        (f"sft_val{suffix}.jsonl", rows[:n_val]),
        (f"sft_train{suffix}.jsonl", rows[n_val:]),
    ):
        (out / name).write_text(
            chr(10).join(json.dumps(r, ensure_ascii=False) for r in subset) + chr(10),
            encoding="utf-8",
        )
    print(
        f"[gdar] {out}/sft_train{suffix}.jsonl（{len(rows) - n_val}）+ sft_val{suffix}.jsonl（{n_val}）"
    )
    return len(rows)


def prep_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{stage} 语料准备（messages jsonl）")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--config", default=None, help="config/data_prep/<名字>.yaml")
    ap.add_argument("--blend", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    if not args.prepare:
        ap.print_help()
        return 1
    name = common.profile_from_args(args.config, "default", stage)
    cfg = common.dataprep_config(here / "config/data_prep" / f"{name}.yaml")
    blend = args.blend or cfg.get("blend") or "data_blend_raw.json"
    return (
        0
        if prepare(
            stage=stage,
            blend=blend,
            limit=args.limit if args.limit is not None else cfg.get("limit"),
            val_frac=args.val_frac if args.val_frac is not None else (cfg.get("val_frac") or 0.02),
            root=Path(args.root) if args.root else None,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            suffix=SUFFIX.get(Path(blend).stem.replace("data_blend_", ""), ""),
        )
        >= 0
        else 1
    )
