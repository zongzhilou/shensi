"""RL 语料准备：prompts jsonl → verl 的 train/val parquet（配比驱动）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from shensi.recipes.paper.looma.common import gdar as common

PROMPT_KEYS = ("prompt", "question", "problem", "instruction", "messages")


def _source_dirs(root: Path, blend_spec: dict | None) -> list[Path]:
    if not blend_spec:
        return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    out = []
    for spec in blend_spec.get("datasets", []):
        name = spec["name"]
        sub = spec.get("config")
        d = root / name / sub if sub else root / name
        if not d.is_dir():
            d = root / name
        if d.is_dir():
            out.append(d)
    return out


def prepare(
    *,
    stage: str,
    blend: str = "data_blend_raw.json",
    limit: int | None = None,
    val_frac: float = 0.02,
    root: Path | None = None,
    out: Path | None = None,
) -> int:
    paths = common.env_paths()
    root = Path(root or paths["post"])
    out = Path(out or (paths["data"] / stage))
    blend_path = common.stage_dirs(stage)[1] / "data_prep" / blend
    blend_spec = common.load_blend_spec(blend_path) if blend_path.is_file() else None
    rows: list[dict] = []
    for d in _source_dirs(root, blend_spec):
        for f in sorted(d.glob("**/*")):
            if f.suffix not in (".jsonl", ".json"):
                continue
            with open(f, encoding="utf-8") as fh:
                for ln in fh:
                    if not ln.strip():
                        continue
                    row = json.loads(ln)
                    prompt = next((row[k] for k in PROMPT_KEYS if k in row), None)
                    if prompt:
                        rows.append({"prompt": prompt})
                    if limit and len(rows) >= limit:
                        break
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise SystemExit(
            f"[gdar] 没有可用 prompt：{root}（配比 {blend}）。语料落位或 --root 指对目录后重跑。"
        )
    n_val = max(1, int(len(rows) * val_frac))
    out.mkdir(parents=True, exist_ok=True)
    for name, subset in (("val.parquet", rows[:n_val]), ("train.parquet", rows[n_val:])):
        pq.write_table(
            pa.table(
                {
                    "prompt": [r["prompt"] for r in subset],
                    "data_source": [stage] * len(subset),
                    "ability": [stage.split("_")[-1]] * len(subset),
                    "reward_model": [{"style": "rule", "ground_truth": ""} for _ in subset],
                    # 带 prompt：pyarrow 写不了空 struct；RL 式 OPD 的 reward 也从这一列取
                    "extra_info": [{"prompt": r["prompt"]} for r in subset],
                }
            ),
            out / name,
        )
    print(f"[gdar] {out}/train.parquet（{len(rows) - n_val}）+ val.parquet（{n_val}）")
    return len(rows)


def main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{stage} 的 RL 数据准备")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--config", default=None, help="config/data_prep/<name>.yaml")
    ap.add_argument("--blend", default=None, help="配比 json（相对 config/data_prep/）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument(
        "--root", default=None, help="prompts 根目录（默认 <FS>/datasets/llm/post-training）"
    )
    args = ap.parse_args(argv if argv is not None else None)
    if not args.prepare:
        ap.print_help()
        return 1
    cfg = common.dataprep_config_for(here, args.config, args.blend)
    return (
        0
        if prepare(
            stage=stage,
            blend=args.blend or cfg.get("blend") or "data_blend_raw.json",
            limit=args.limit if args.limit is not None else cfg.get("limit"),
            val_frac=args.val_frac,
            root=Path(args.root) if args.root else None,
        )
        >= 0
        else 1
    )
