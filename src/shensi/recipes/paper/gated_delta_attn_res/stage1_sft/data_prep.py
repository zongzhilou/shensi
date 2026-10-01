#!/usr/bin/env python3
"""GDAR 配方 · stage1_sft 语料准备：UltraData-SFT → messages jsonl。

mcore 的 `--sft`（SFTTokenizer + Qwen3 chat 模板）直读 jsonl，每行一个
`{"messages": [{"role": ..., "content": ...}, ...]}`；本脚本把 UltraData-SFT-2605 /
UltraData-SFT-Agent-2609（parquet/jsonl，落位 `$SHENSI_FS/datasets/llm/post-training/<name>/`）
规整成这个格式，按 98/2 切 sft_train.jsonl / sft_val.jsonl。

    python data_prep.py --discover                    # 看数据面貌
    python data_prep.py --prepare                     # SFT-1（UltraData-SFT-2605）
    python data_prep.py --prepare --blend agent.json  # SFT-2（UltraData-SFT-Agent-2609）
    python data_prep.py --prepare --limit 1000        # 调试档

thinking 字段（deep-thinking 的 reasoning_content）原样保留在 messages 里，
SFTTokenizer 按 chat 模板处理；Qwen3 模板自己的 think 标签由 tokenizer 侧负责。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common

STAGE = "stage1_sft"
MESSAGE_KEYS = ("messages", "conversations", "conversation")
TEXT_KEYS = ("text", "content", "response")


def to_messages(row: dict) -> list[dict] | None:
    """把数据集的一行规整成 messages；认不出对话结构就返回 None（跳过）。"""
    for key in MESSAGE_KEYS:
        msgs = row.get(key)
        if isinstance(msgs, list) and msgs:
            out = []
            for m in msgs:
                if not isinstance(m, dict):
                    return None
                role = m.get("role") or m.get("from") or m.get("sender")
                content = m.get("content") or m.get("value") or m.get("text")
                if role is None or content is None:
                    return None
                entry = {"role": role, "content": content}
                if m.get("reasoning_content"):
                    entry["reasoning_content"] = m["reasoning_content"]
                out.append(entry)
            return out or None
    # 单文本字段（问答拼接）：按 user/assistant 两轮兜底
    for key in TEXT_KEYS:
        text = row.get(key)
        if isinstance(text, str) and len(text) > 16:
            return [
                {"role": "user", "content": text[: len(text) // 2]},
                {"role": "assistant", "content": text[len(text) // 2 :]},
            ]
    return None


def _files(root: Path, name: str) -> list[Path]:
    d = root / name
    if not d.is_dir():
        return []
    return [f for f in sorted(d.glob("**/*")) if f.suffix in (".parquet", ".jsonl", ".json")]


def main() -> int:
    ap = argparse.ArgumentParser(description="GDAR stage1_sft 语料准备（messages jsonl）")
    ap.add_argument("--discover", action="store_true", help="只看数据面貌，不产出")
    ap.add_argument("--prepare", action="store_true", help="产出 sft_train.jsonl / sft_val.jsonl")
    ap.add_argument(
        "--blend", default="default.json", help="配比 json（default=SFT-1，agent.json=SFT-2）"
    )
    ap.add_argument("--limit", type=int, default=None, help="每数据集最多多少条（调试用）")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = common.env_paths()
    blend_path = common.stage_dirs(STAGE)[1] / "data_prep" / args.blend
    blend_spec = common.load_blend_spec(blend_path)
    root = paths["post"]
    out = paths["data"] / STAGE

    if args.discover:
        for spec in blend_spec["datasets"]:
            files = _files(root, spec["name"])
            print(
                f"  {spec['name']}: 文件 {len(files)}"
                + (f"（如 {files[0].name}）" if files else "（未落位）")
            )
        return 0
    if not args.prepare:
        ap.print_help()
        return 1

    out.mkdir(parents=True, exist_ok=True)
    rows: list[list[dict]] = []
    for spec in blend_spec["datasets"]:
        n = 0
        for f in _files(root, spec["name"]):
            if f.suffix == ".parquet":
                import pyarrow.parquet as pq

                batches = pq.ParquetFile(f).iter_batches(batch_size=512)
                it = (r for batch in batches for r in batch.to_pylist())
            else:
                with open(f, encoding="utf-8") as fh:
                    lines = [ln for ln in fh if ln.strip()]
                it = (json.loads(ln) for ln in lines)
            for row in it:
                msgs = to_messages(row)
                if msgs:
                    rows.append(msgs)
                    n += 1
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break
        print(f"  [data_prep] {spec['name']}: {n} 条")
    if not rows:
        raise SystemExit(
            f"[gdar] 没有可用数据：{root} 下没有 SFT 集。UltraData-SFT-2605 / Agent-2609 落位后重跑。"
        )
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    with open(out / "sft_val.jsonl", "w", encoding="utf-8") as fh:
        for msgs in rows[:n_val]:
            fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    with open(out / "sft_train.jsonl", "w", encoding="utf-8") as fh:
        for msgs in rows[n_val:]:
            fh.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(
        f"[gdar] mcore --sft 口径：{out}/sft_train.jsonl（{len(rows) - n_val} 行）+ sft_val.jsonl（{n_val} 行）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
