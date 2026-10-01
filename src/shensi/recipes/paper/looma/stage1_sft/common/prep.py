"""SFT 语料准备：parquet / jsonl → messages jsonl（按 val_frac 切分）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shensi.recipes.paper.looma.common import (
    dataprep_config,
    env_paths,
    load_blend_spec,
    profile_from_args,
    stage_dirs,
)

#: 各子段的文件名后缀（``--config agent`` → ``sft_train_agent.jsonl``）
SUFFIX = {"default": "", "hybrid": "_hybrid", "agent": "_agent"}


def _rows_from(spec: dict, root: Path, limit: int | None) -> list[dict]:
    """读 parquet / jsonl 里的 messages，返回统一格式的行。"""
    import pandas as pd

    rows: list[dict] = []
    for item in spec.get("datasets", []):
        name, sub = item["name"], item.get("config")
        directory = root / name / sub if sub else root / name
        if not directory.is_dir():
            directory = root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("**/*")):
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
                for _, record in frame.iterrows():
                    if "messages" in frame.columns:
                        rows.append({"messages": list(record["messages"])})
                    else:
                        rows.append({"messages": json.loads(record.to_json())["messages"]})
            elif path.suffix in (".jsonl", ".json"):
                with open(path, encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            row = json.loads(line)
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
    """写 ``sft_train<suffix>.jsonl`` 与 ``sft_val<suffix>.jsonl``，返回总条数。"""
    paths = env_paths()
    root = Path(root or paths["post"])
    blend_path = stage_dirs(stage) / "config" / "data_prep" / blend
    rows = _rows_from(load_blend_spec(blend_path), root, limit)
    if not rows:
        raise SystemExit(f"[looma] 没有可用 SFT 数据：{root}（配比 {blend}）")
    out = Path(out_dir or paths["data"] / stage)
    out.mkdir(parents=True, exist_ok=True)
    n_val = max(1, int(len(rows) * val_frac))
    for name, subset in (
        (f"sft_val{suffix}.jsonl", rows[:n_val]),
        (f"sft_train{suffix}.jsonl", rows[n_val:]),
    ):
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in subset) + "\n"
        (out / name).write_text(text, encoding="utf-8")
    print(
        f"[looma] {out}/sft_train{suffix}.jsonl（{len(rows) - n_val}）+ sft_val{suffix}.jsonl（{n_val}）"
    )
    return len(rows)


def prep_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=f"{stage} 语料准备（messages jsonl）")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--config", default=None, help="config/data_prep/<名字>.yaml")
    parser.add_argument("--blend", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    if not args.prepare:
        parser.print_help()
        return 1
    name = profile_from_args(args.config, "default", stage)
    cfg = dataprep_config(here / "config" / "data_prep" / f"{name}.yaml")
    blend = args.blend or cfg.get("blend") or "data_blend_raw.json"
    prepare(
        stage=stage,
        blend=blend,
        limit=args.limit if args.limit is not None else cfg.get("limit"),
        val_frac=args.val_frac if args.val_frac is not None else (cfg.get("val_frac") or 0.02),
        root=Path(args.root) if args.root else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        suffix=SUFFIX.get(Path(blend).stem.replace("data_blend_", ""), ""),
    )
    return 0
