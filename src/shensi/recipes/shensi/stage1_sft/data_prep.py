#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage0_pretrain"))

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common

STAGE = "stage1_sft"
MESSAGE_KEYS = ("messages", "conversations", "conversation")
INSTRUCTION_KEYS = ("instruction", "question", "prompt", "query", "problem")


def to_messages(row: dict) -> list[dict] | None:
    for key in MESSAGE_KEYS:
        val = row.get(key)
        if isinstance(val, list) and val and isinstance(val[0], dict):
            out = []
            for m in val:
                role = str(m.get("role") or m.get("from") or "user")
                content = m.get("content") or m.get("value") or ""
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant"):
                    role = "assistant"
                out.append({"role": role, "content": str(content)})
            return out or None
    instr = next((row[k] for k in INSTRUCTION_KEYS if isinstance(row.get(k), str)), None)
    if instr:
        answer = (
            row.get("output") or row.get("response") or row.get("answer") or row.get("completion")
        )
        if isinstance(answer, str):
            return [{"role": "user", "content": instr}, {"role": "assistant", "content": answer}]
    return None


def iter_rows(files: list[Path], limit: int | None):
    n = 0
    for f in files:
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    yield row
                    n += 1
                    if limit and n >= limit:
                        return
        elif f.suffix in (".jsonl", ".json"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
                        n += 1
                        if limit and n >= limit:
                            return


def _encoding_dsv4():
    """`encoding_dsv4` 与本文件同目录（官方那份原样放在这儿）。

    别的 stage 会按**文件路径**加载本模块（stage4_world_model 复用 SFT 口径、stage3_eval 复用
    `to_messages`），那种加载方式不会把本目录放进 `sys.path`，所以这里自己补一下。
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import encoding_dsv4

    return encoding_dsv4


def render(tok, messages: list[dict], mode: str | None = None) -> tuple[list[int], list[int]]:
    """按 DeepSeek-V4 的 chat 编码（官方 encoding/encoding_dsv4.py）拼 token；loss 只算 assistant 段。"""
    _enc = _encoding_dsv4()
    ASSISTANT_SP_TOKEN, encode_messages = _enc.ASSISTANT_SP_TOKEN, _enc.encode_messages

    thinking_mode = mode or (
        "thinking" if any(m.get("reasoning_content") for m in messages) else "chat"
    )
    ids = tok(encode_messages(messages, thinking_mode=thinking_mode), add_special_tokens=False)[
        "input_ids"
    ]
    a_id = tok(ASSISTANT_SP_TOKEN, add_special_tokens=False)["input_ids"][0]
    think_close = tok("</think>", add_special_tokens=False)["input_ids"]
    mask = [0] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i] != a_id:
            i += 1
            continue
        j = i + 1
        if ids[j : j + len(think_close)] == think_close:
            j += len(think_close)  # chat 形态的 </think> 是脚手架，不参训
        else:
            j += 1  # thinking 形态的 <think> 同样不参训（与 mcore 的 assistant_prefix_len=2 对齐）
        while j < len(ids) and ids[j] != tok.eos_token_id:
            mask[j] = 1
            j += 1
        if j < len(ids):
            mask[j] = 1  # 结尾的 <｜end▁of▁sentence｜> 参训，让模型学会收尾
        i = j + 1
    return ids, mask


# 三种口径一次写全：messages jsonl（mcore --sft）/ messages parquet（verl）/ packed+loss_mask parquet
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shensi stage1_sft 语料准备")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument(
        "--blend", default=None, help="换一份配比 json（默认 config/data_prep/data_blend_raw.json）"
    )
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument(
        "--pack", action="store_true", default=True, help="同时产出 packed+loss_mask 版本"
    )
    ap.add_argument("--no-pack", dest="pack", action="store_false")
    ap.add_argument(
        "--root", default=None, help="SFT 语料根，默认 $SHENSI_FS/datasets/llm/post-training"
    )
    ap.add_argument("--out", default=None, help="产物目录，默认 $SHENSI_FS/shensi/data/stage1_sft")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="只处理名字含该子串的数据集")
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--val-ratio", type=float, default=0.005)
    ap.add_argument(
        "--min-response-chars",
        type=int,
        default=None,
        help="只留应答不短于这个字符数的样本（长输出能力，LongWriter 口径）；不给就不筛",
    )
    ap.add_argument(
        "--thinking-mode",
        default="auto",
        choices=("auto", "chat", "thinking"),
        help="DeepSeek-V4 的编码模式；auto = 有 reasoning_content 就用 thinking",
    )
    args = ap.parse_args(argv)

    paths = common.env_paths()
    root = Path(args.root or paths["post"])
    out = Path(args.out or paths["data"] / STAGE)
    spec = common.load_blend_spec(
        Path(args.blend)
        if args.blend
        else Path(__file__).parent / "config/data_prep/data_blend_raw.json"
    )
    datasets = [d for d in spec["datasets"] if not args.only or args.only in d["name"]]
    if args.discover:
        print(f"[sft] 语料根：{root}")
        for d in datasets:
            files = _files(root, d)
            print(
                f"  {'✅' if files else '❌'} {d['name']:<44} 文件 {len(files):<4} weight={d.get('weight')}"
            )
        return 0
    if not args.prepare:
        ap.error("至少给一个：--discover / --prepare")

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or paths["tokenizer"])
    rows_all: list[dict] = []
    for d in datasets:
        files = _files(root, d)
        if not files:
            msg = f"{d['name']}: 在 {root} 下没找到文件"
            if args.skip_missing:
                print(f"[sft] 跳过（--skip-missing）：{msg}")
                continue
            raise SystemExit(f"{msg}，先跑 --discover 看看")
        n = 0
        for row in iter_rows(files, args.limit):
            msgs = to_messages(row)
            if msgs:
                first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
                last_assistant = next(
                    (m["content"] for m in reversed(msgs) if m["role"] == "assistant"), ""
                )
                if args.min_response_chars and len(last_assistant) < args.min_response_chars:
                    continue
                rows_all.append(
                    {
                        "messages": msgs,
                        "source": d["name"],
                        # 兼容列：部分 fork 的 SFT dataset 直接读 question/answer
                        "question": first_user,
                        "answer": last_assistant,
                    }
                )
                n += 1
        print(f"[sft] {d['name']}: {n} 条")
    if not rows_all:
        raise SystemExit("[sft] 一条都没解析出来：看看 --discover 打出来的字段名")

    out.mkdir(parents=True, exist_ok=True)
    # mcore 的 --sft 直接读 jsonl（每行一个 {"messages": [...]}），见
    # megatron/training/datasets/sft_dataset.py::SFTLowLevelDataset
    n_val_j = max(2, int(len(rows_all) * args.val_ratio))
    with open(out / "sft_train.jsonl", "w", encoding="utf-8") as fh:
        for r in rows_all[n_val_j:]:
            fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    with open(out / "sft_val.jsonl", "w", encoding="utf-8") as fh:
        for r in rows_all[:n_val_j]:
            fh.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    print(
        f"[sft] mcore --sft 口径：{out}/sft_train.jsonl（{len(rows_all) - n_val_j} 行）"
        f" + sft_val.jsonl（{n_val_j} 行）"
    )
    # 至少 2 行：verl 的 sft_dataset 对单行 DataFrame 会 squeeze 成标量（实测 .tolist() 报错）
    n_val = max(2, int(len(rows_all) * args.val_ratio))
    table = pa.Table.from_pylist(rows_all)
    pq.write_table(table.slice(n_val, len(rows_all) - n_val), out / "train.parquet")
    pq.write_table(table.slice(0, n_val), out / "test.parquet")
    print(
        f"[sft] messages 口径：{out}/train.parquet（{len(rows_all) - n_val} 行）+ test.parquet（{n_val} 行）"
    )

    if args.pack:
        packed_dir = out / "packed"
        packed_dir.mkdir(exist_ok=True)
        packed = []
        for r in rows_all:
            ids, mask = render(
                tok, r["messages"], None if args.thinking_mode == "auto" else args.thinking_mode
            )
            packed.append({"input_ids": ids, "loss_mask": mask, "seq_length": len(ids)})
        pq.write_table(pa.Table.from_pylist(packed), packed_dir / "train.parquet")
        print(
            f"[sft] packed+loss_mask 口径：{packed_dir}/train.parquet（{len(packed)} 行，"
            f"平均 {sum(p['seq_length'] for p in packed) // max(len(packed), 1)} token/条）"
        )
    return 0


def _files(root: Path, d: dict) -> list[Path]:
    base = root / d["name"]
    if not base.is_dir():
        return []
    out = []
    for f in sorted(base.glob("**/*")):
        if f.is_dir() or f.suffix not in (".parquet", ".jsonl", ".json"):
            continue
        rel = str(f.relative_to(base))
        if d.get("config") and d["config"] not in rel:
            continue
        out.append(f)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
