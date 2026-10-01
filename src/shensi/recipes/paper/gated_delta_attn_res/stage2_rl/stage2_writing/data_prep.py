#!/usr/bin/env python3
"""写作 的 RL 数据准备：prompts jsonl → verl 的 train/val parquet。

输入：`$SHENSI_FS/datasets/llm/post-training/<数据集>/`（jsonl/parquet，含 prompt 字段）；
输出：`$SHENSI_FS/shensi/data/gated_delta_attn_res/stage2_writing/train.parquet + val.parquet`
（verl RLVR schema：prompt / data_source / ability / reward_model / extra_info）。
"""

from __future__ import annotations

import argparse
import json

import pyarrow as pa
import pyarrow.parquet as pq

from shensi.recipes.paper.gated_delta_attn_res import common

STAGE = "stage2_writing"
PROMPT_KEYS = ("prompt", "question", "problem", "instruction", "messages")


def main() -> int:
    ap = argparse.ArgumentParser(description="写作 RL 数据准备")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()
    if not args.prepare:
        ap.print_help()
        return 1

    paths = common.env_paths()
    root = paths["post"]
    out = paths["data"] / STAGE
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        for f in sorted(d.glob("**/*")):
            if f.suffix not in (".jsonl", ".json"):
                continue
            with open(f, encoding="utf-8") as fh:
                lines = [ln for ln in fh if ln.strip()]
            for ln in lines:
                row = json.loads(ln)
                prompt = next((row[k] for k in PROMPT_KEYS if k in row), None)
                if prompt:
                    rows.append({"prompt": prompt})
                if args.limit and len(rows) >= args.limit:
                    break
            if args.limit and len(rows) >= args.limit:
                break
    if not rows:
        raise SystemExit(f"[gdar] 没有可用 prompt 数据：{root} 落位后重跑。")
    n_val = max(1, int(len(rows) * args.val_frac))
    out.mkdir(parents=True, exist_ok=True)
    for name, subset in (("val.parquet", rows[:n_val]), ("train.parquet", rows[n_val:])):
        table = pa.table(
            {
                "prompt": [r["prompt"] for r in subset],
                "data_source": [STAGE] * len(subset),
                "ability": [STAGE.split("_")[-1]] * len(subset),
                "reward_model": [{"style": "rule", "ground_truth": ""} for _ in subset],
                "extra_info": [{} for _ in subset],
            }
        )
        pq.write_table(table, out / name)
    print(f"[gdar] {out}/train.parquet（{len(rows) - n_val}）+ val.parquet（{n_val}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
