#!/usr/bin/env python3
"""批量生成：拿本地检查点直接采样（OPD ① 的 plain 路径，也可当一般批采样工具）。

不起 OpenAI 服务：在建好的 vLLM 引擎上把 prompts jsonl 一次采完，落成 jsonl。prompts
每行给 ``prompt``（或 ``text`` / ``question`` / ``instruction`` / ``query``）字符串，
或给 ``messages`` 消息列表（走检查点里的 chat 模板）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .register_model import register_all

_STRING_KEYS = ("prompt", "text", "question", "instruction", "query")


def load_prompts(path: Path, key: str | None = None) -> list[dict[str, Any]]:
    """读 prompts jsonl，每行归成 ``{"text": …, "messages": …}`` 两种形态之一。"""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"[looma·vllm] {path}:{lineno} 不是 JSON 对象")
            messages = row.get("messages")
            if key:
                text = row.get(key)
            else:
                text = next((row[k] for k in _STRING_KEYS if isinstance(row.get(k), str)), None)
            if messages is None and not isinstance(text, str):
                raise SystemExit(
                    f"[looma·vllm] {path}:{lineno} 既没有 messages，也没有可认的字符串字段"
                    f"（{list(_STRING_KEYS)}；要换字段名用 --prompt-key）"
                )
            rows.append({"text": text if isinstance(text, str) else None, "messages": messages})
    return rows


def cap_max_model_len(ckpt: Path, want: int) -> int:
    """按检查点自带的 ``max_position_embeddings`` 收窄 ``max_model_len``。"""
    cfg_path = ckpt / "config.json"
    if not cfg_path.is_file():
        return want
    try:
        limit = int(
            json.loads(cfg_path.read_text(encoding="utf-8")).get("max_position_embeddings") or 0
        )
    except (ValueError, OSError):
        return want
    if limit and want > limit:
        print(f"[looma·vllm] max_model_len {want} > 模型上限 {limit}，压到 {limit}")
        return limit
    return want


def _row_out(rows_in: list[dict[str, Any]], index: int, responses: list[dict]) -> dict:
    row = {"index": index, "responses": responses}
    if rows_in[index]["text"] is not None:
        row["prompt"] = rows_in[index]["text"]
    if rows_in[index]["messages"] is not None:
        row["messages"] = rows_in[index]["messages"]
    return row


def main(argv: list[str] | None = None) -> int:
    """批量生成入口：登记实现 → 建引擎 → 采样 → 写 jsonl。"""
    ap = argparse.ArgumentParser(description="Looma：本地检查点的批量生成")
    ap.add_argument("--ckpt", required=True, help="HF 目录（export_hf.py 或 tiny_checkpoint 的产物）")
    ap.add_argument("--prompts", required=True, help="prompts jsonl")
    ap.add_argument("--out", required=True, help="输出 jsonl（每行一条 prompt + 它的样本）")
    ap.add_argument("--prompt-key", default=None, help="指定 prompt 字段名（默认自动认几个常见名）")
    ap.add_argument("--n", type=int, default=1, help="每条采几个样本")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认关掉 torch.compile / CUDAGraph：连接的形状有数据依赖，编译过不去",
    )
    ap.add_argument("--limit", type=int, default=None, help="只采前 N 条（调试）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不建引擎")
    args = ap.parse_args(argv)

    ckpt = Path(args.ckpt)
    if not (ckpt / "config.json").is_file():
        raise SystemExit(f"[looma·vllm] {ckpt} 不是 HF 目录（缺 config.json）")
    rows = load_prompts(Path(args.prompts), args.prompt_key)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"[looma·vllm] {args.prompts} 里没有可采的 prompt")
    n_chat = sum(1 for row in rows if row["messages"] is not None)
    print(
        f"[looma·vllm] {ckpt}：{len(rows)} 条 prompt（{len(rows) - n_chat} 文本 + {n_chat} messages）"
        f"× n={args.n}，temperature={args.temperature}"
    )
    if args.dry_run:
        return 0

    register_all()
    from vllm import LLM, SamplingParams

    max_model_len = cap_max_model_len(ckpt, args.max_model_len)
    llm = LLM(
        model=str(ckpt),
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    texts = [(i, row["text"]) for i, row in enumerate(rows) if row["text"] is not None]
    chats = [(i, row["messages"]) for i, row in enumerate(rows) if row["messages"] is not None]
    done: dict[int, list[dict]] = {}
    for batch, call in ((texts, llm.generate), (chats, llm.chat)):
        if not batch:
            continue
        outs = call([item for _, item in batch], sp)
        for (index, _), out in zip(batch, outs):
            done[index] = [
                {
                    "text": candidate.text,
                    "token_ids": list(candidate.token_ids),
                    "finish_reason": candidate.finish_reason,
                }
                for candidate in out.outputs
            ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for index in range(len(rows)):
            handle.write(json.dumps(_row_out(rows, index, done[index]), ensure_ascii=False) + "\n")
    n_tokens = sum(len(c["token_ids"]) for responses in done.values() for c in responses)
    print(f"[looma·vllm] 生成 {n_tokens} token，写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
