"""受控检索题的生成器：按长度与键数网格合成题目。"""


from __future__ import annotations

import argparse
import os
import json
import random
import re
import string
from pathlib import Path


_BUILTIN_SENTENCES = [
    "The archive catalog lists several unrelated entries from the same period.",
    "Records were reorganized by the staff during the winter closure.",
    "A short note in the margin describes the condition of the binding.",
    "The index volume was rebound and the page numbers were preserved.",
    "Consultation requests are processed in the order they are received.",
    "The reading room closes thirty minutes before the building does.",
    "Several boxes were moved to the annex to make room for new arrivals.",
    "The catalogue cards were digitized and checked against the originals.",
]
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def load_filler(data_dir: Path | None, want_chars: int, rng: random.Random) -> str:
    chunks: list[str] = []
    if data_dir is not None and data_dir.exists():
        files = sorted(data_dir.glob("*/**/*.jsonl"))[:4]
        for f in files:
            try:
                with f.open() as fh:
                    for line in fh:
                        row = json.loads(line).get("row", {})
                        text = row.get("text") or row.get("content") or ""
                        if isinstance(text, str) and len(text) > 200:
                            chunks.append(text)
                        if sum(map(len, chunks)) > want_chars:
                            break
            except Exception:
                continue
            if sum(map(len, chunks)) > want_chars:
                break
    if not chunks:
        while sum(map(len, chunks)) < want_chars:
            rng.shuffle(_BUILTIN_SENTENCES)
            chunks.append(" ".join(_BUILTIN_SENTENCES))
    return "\n".join(chunks)


def make_question(rng: random.Random, filler: str, length: int, k_pairs: int, repeats: int) -> dict:
    """合成一道受控检索题（给定长度与键数）。"""
    keys = [f"code-{rng.randrange(1000, 9999)}" for _ in range(k_pairs)]
    query_key = keys[0]
    other_keys = keys[1:]

    def value() -> str:
        return f"{rng.randrange(100, 999)}-{''.join(rng.choice(string.ascii_uppercase) for _ in range(3))}"

    writes = [value() for _ in range(repeats)]
    correct = writes[-1]
    distractors_pool = list(writes[:-1])
    for key in other_keys:
        distractors_pool.append(value())

    facts = [(query_key, v) for v in writes]
    facts += [(key, value()) for key in other_keys]
    rng.shuffle(facts)

    span = max(400, length)
    if len(filler) > span:
        start = rng.randrange(0, len(filler) - span)
        body = filler[start : start + span]
    else:
        body = filler[:span]
    sentences = [f"The current {key} is {val}." for key, val in facts]
    cuts = sorted(rng.randrange(0, len(body)) for _ in facts)
    out, prev = [], 0
    for cut, sent in zip(cuts, sentences):
        out.append(body[prev:cut])
        out.append(" " + sent + " ")
        prev = cut
    out.append(body[prev:])
    context = "".join(out)
    question = f"\n\nQuestion: What is the current value of {query_key}?\nAnswer:"

    rng.shuffle(distractors_pool)
    choices = [correct] + distractors_pool[:3]
    while len(choices) < 4:
        choices.append(value())
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [choices[i] for i in order]
    answer_idx = shuffled.index(correct)

    return {
        "context": context + question,
        "query_key": query_key,
        "k_pairs": k_pairs,
        "repeats": repeats,
        "context_chars": len(context),
        "choices": shuffled,
        "answer_idx": answer_idx,
        "correct": correct,
        "chance": 0.25,
    }


def main() -> int:
    """题目生成入口：按长度 × 键数网格批量产出。"""
    ap = argparse.ArgumentParser(description="生成 depth-retrieval（最新值检索）评测集")
    ap.add_argument("--out", required=True, help="输出 JSONL 路径")
    ap.add_argument("--n", type=int, default=1000, help="题量（默认 1000；审稿人要求 >=1000）")
    ap.add_argument("--lengths", default="1024,2048,4096", help="上下文长度分层（字符数近似）")
    ap.add_argument("--ks", default="1,2,4,8", help="每题的 KV 对数分层（干扰强度）")
    ap.add_argument("--repeats", type=int, default=3, help="被查询键的写入次数（>=2 才有'最新'语义）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=os.environ.get("SHENSI_FS", "/root/work/filestorage") + "/datasets/llm/pre-training",
                    help="填充文本来源（默认 $SHENSI_FS/datasets/llm/pre-training；--filler-random 用内置文本）")
    ap.add_argument("--filler-random", action="store_true", help="强制使用内置填充文本（不读数据文件）")
    ap.add_argument("--emit-harness", default=None, help="额外写出一份 lm-evaluation-harness task YAML")
    args = ap.parse_args()

    if args.n < 1000:
        print(f"[warn] --n={args.n} < 1000：审稿人要求题量 >=1000，正式报告请用 >=1000")
    if args.repeats < 2:
        raise SystemExit("--repeats 必须 >= 2，否则没有'最新值'语义")

    rng = random.Random(args.seed)
    lengths = [int(x) for x in args.lengths.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    data_dir = None if args.filler_random else Path(args.data_dir)
    filler = load_filler(data_dir, max(lengths) + 2000, rng)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strata: dict[tuple[int, int], int] = {}
    with out_path.open("w") as fh:
        for i in range(args.n):
            length = lengths[i % len(lengths)]
            k = ks[(i // len(lengths)) % len(ks)]
            q = make_question(rng, filler, length, k, args.repeats)
            q["id"] = f"dr-{i:05d}"
            strata[(k, length)] = strata.get((k, length), 0) + 1
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")

    manifest = {
        "task": "depth-retrieval (latest-value retrieval)",
        "n": args.n,
        "chance": 0.25,
        "answer_format": "4-way multiple choice; correct = the *latest* written value of the queried key",
        "distractors": "earlier writes of the queried key, then values of the other keys",
        "seed": args.seed,
        "repeats_per_query_key": args.repeats,
        "strata": {f"K={k},L={L}": c for (k, L), c in sorted(strata.items())},
        "filler_source": "builtin" if args.filler_random or data_dir is None else str(data_dir),
        "file": str(out_path),
    }
    man_path = out_path.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    if args.emit_harness:
        yml = Path(args.emit_harness)
        yml.write_text(
            "task: depth_retrieval\n"
            "dataset_path: json\n"
            f"dataset_kwargs:\n  data_files:\n    test: {out_path}\n"
            "output_type: multiple_choice\n"
            "test_split: test\n"
            "doc_to_text: \"{{context}}\"\n"
            "doc_to_choice: \"{{choices}}\"\n"
            "doc_to_target: \"{{answer_idx}}\"\n"
            "metric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n"
            "  - metric: acc_norm\n    aggregation: mean\n    higher_is_better: true\n"
        )
        print(f"[harness] 已写出 {yml}")

    idx_counts = [0, 0, 0, 0]
    missing = 0
    for line in out_path.open():
        q = json.loads(line)
        idx_counts[q["answer_idx"]] += 1
        if q["context"].count(f"The current {q['query_key']} is {q['correct']}.") < 1:
            missing += 1
    total = sum(idx_counts)
    print(f"[self-check] answer_idx 分布 = {[round(c / total, 3) for c in idx_counts]}（应接近 [.25,.25,.25,.25]）")
    print(f"[self-check] 正确答案句缺失的题数 = {missing}（必须为 0）")
    if missing:
        raise SystemExit("生成有 bug：答案句不在上下文里")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

