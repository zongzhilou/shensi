#!/usr/bin/env python3
"""长上下文三类语料的构建（对齐 GLM-5 长上下文配方）。"""

import argparse
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi.common import common

STAGE = "stage3_longctx"

NEXT_LONG_NAME = "Shensi-Longctx-Synth-NextLong"
ENTROPY_LONG_NAME = "Shensi-Longctx-Synth-EntropyLong"
MRCR_NAME = "Shensi-Longctx-MRCR"

# 长文档源：自然长文为主（判例、长网页），够长且话题连续，适合拼接与埋针
DEFAULT_SOURCES = (
    "Nemotron-Pretraining-Legal-v1",
    "Nemotron-CC-v2.1",
    "Nemotron-CC-v2",
    "DCLM-Baseline",
    "Ultra-FineWeb",
)

DOC_SEPARATOR = "\n\n"


def iter_documents(
    root: Path,
    sources: Iterable[str],
    *,
    min_chars: int,
    limit: int | None = None,
) -> Iterator[str]:
    """按数据集逐个读长文档；只取够长的（短文档对长上下文段没有意义）。"""
    yielded = 0
    for name in sources:
        spec = {"name": name, "config": ""}
        for path in common._dataset_files(root, spec):  # noqa: SLF001 - 与 prepare 共用同一套落点规则
            for text in common._iter_records([path], None, None, min_chars):  # noqa: SLF001
                yield text
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def iter_jsonl_documents(path: Path, *, min_chars: int, limit: int | None = None) -> Iterator[str]:
    """从本地 jsonl 读文档（离线自检与单测用，不需要云端语料）。"""
    for index, text in enumerate(common._iter_records([path], None, None, min_chars)):  # noqa: SLF001
        if limit is not None and index >= limit:
            return
        yield text


def join_to_target(parts: Iterable[str], target_chars: int) -> str:
    """把片段拼到 target_chars 上下（不低于 target 就停，单个片段过长则截断到 target）。"""
    buf: list[str] = []
    size = 0
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if size + len(part) + len(DOC_SEPARATOR) > target_chars:
            room = target_chars - size - len(DOC_SEPARATOR)
            if room > 0:
                buf.append(part[:room])
            break
        buf.append(part)
        size += len(part) + len(DOC_SEPARATOR)
    return DOC_SEPARATOR.join(buf)


def synth_nextlong(documents: Iterator[str], target_chars: int, *, seed: int = 0) -> Iterator[str]:
    """NextLong 式：同源连续文档按原顺序拼接成一篇长文（话题连续，最接近「书/论文」的读感）。"""
    rng = random.Random(seed)
    buf: list[str] = []
    size = 0
    for doc in documents:
        buf.append(doc)
        size += len(doc) + len(DOC_SEPARATOR)
        if size >= target_chars:
            yield join_to_target(buf, target_chars)
            buf, size = [], 0
            rng.random()  # 保持随机流一致（留作后续打散用的钩子）


def synth_entropylong(
    documents: Iterator[str], target_chars: int, *, seed: int = 0
) -> Iterator[str]:
    """EntropyLong 式：文档切段后打散再拼（跨话题边界多、定位难度高）。"""
    rng = random.Random(seed)
    pool: list[str] = []
    for doc in documents:
        for start in range(0, len(doc), 4000):
            piece = doc[start : start + 4000].strip()
            if piece:
                pool.append(piece)
        rng.shuffle(pool)  # 片段顺序随机化：相邻片段跨文档、跨域
        while len(pool) >= 16:
            chunk, pool = pool[:16], pool[16:]
            yield join_to_target(chunk, target_chars)


def build_mrcr_item(
    documents: Iterator[str],
    *,
    target_chars: int,
    needles: int,
    seed: int = 0,
    index: int = 0,
) -> dict | None:
    """MRCR 类：把 needles 个针按序埋进长文，问「按出现顺序列出编号」。"""
    rng = random.Random(seed + index)
    body = join_to_target(documents, target_chars)
    if len(body) < target_chars // 2:
        return None
    facts = [
        f"记录 {index + 1}-{i + 1} 的内部编号是 {rng.randrange(10**6, 10**7)}"
        for i in range(needles)
    ]
    step = max(1, len(body) // (needles + 1))
    pieces: list[str] = []
    cursor = 0
    for i, fact in enumerate(facts):
        cut = step * (i + 1)
        pieces.append(body[cursor:cut])
        pieces.append(f"\n【记录】{fact}，只在本段出现一次。\n")
        cursor = cut
    pieces.append(body[cursor:])
    context = "".join(pieces)
    question = f"问题：上面按顺序出现过 {needles} 条【记录】，请按它们在材料里出现的顺序列出各自的编号，只回答编号本身。"
    answer = "、".join(
        f"记录 {index + 1}-{i + 1} 的编号是 {facts[i].rsplit(' ', 1)[-1]}" for i in range(needles)
    )
    return {
        "text": f"{context}\n\n{question}\n\n答：{answer}",
        "prompt": f"{context}\n\n{question}",
        "ground_truth": "、".join(fact.rsplit(" ", 1)[-1] for fact in facts),
        "capability": f"mrcr-{needles}needle-{target_chars // 1000}k",
    }


def _write_jsonl(path: Path, rows: Iterable[dict], what: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[longctx] {what} → {path}（{n} 条）")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Shensi stage3_longctx 三类语料构建")
    ap.add_argument("--step", choices=("synth", "mrcr", "all"), default="all")
    ap.add_argument(
        "--root", default=None, help="长文档源根目录（默认 $SHENSI_FS/datasets/llm/pre-training）"
    )
    ap.add_argument(
        "--out", default=None, help="产物目录（默认 $SHENSI_FS/shensi/data/stage3_longctx）"
    )
    ap.add_argument(
        "--eval-out", default=None, help="MRCR 评测集落点（默认 <out>/mrcr_eval.jsonl）"
    )
    ap.add_argument("--source", action="append", default=None, help="长文档源数据集名，可重复")
    ap.add_argument("--source-jsonl", default=None, help="改用本地 jsonl 当源（离线自检/单测）")
    ap.add_argument(
        "--target-chars", type=int, default=200_000, help="每篇长文的目标字符数（200K 段）"
    )
    ap.add_argument("--synth-target-chars", type=int, default=64_000, help="合成长文的目标长度")
    ap.add_argument("--needles", type=int, default=8, help="MRCR 每篇埋几个针")
    ap.add_argument("--items", type=int, default=2000, help="每类要产出多少篇")
    ap.add_argument("--min-chars", type=int, default=20_000, help="源文档的最短长度")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    paths = common.env_paths()
    out = Path(args.out or paths["data"] / STAGE)
    eval_out = Path(args.eval_out or out / "mrcr_eval.jsonl")

    if args.source_jsonl:

        def documents(limit=None):
            return iter_jsonl_documents(
                Path(args.source_jsonl), min_chars=args.min_chars, limit=limit
            )
    else:
        root = Path(args.root or paths["pre"])
        sources = args.source or list(DEFAULT_SOURCES)

        def documents(limit=None):
            return iter_documents(root, sources, min_chars=args.min_chars, limit=limit)

    if args.step in ("synth", "all"):
        _write_jsonl(
            out / f"{NEXT_LONG_NAME}.jsonl",
            (
                "\n".join([f"《长文 {i + 1}》", doc])
                for i, doc in enumerate(
                    _take(
                        synth_nextlong(documents(), args.synth_target_chars, seed=args.seed),
                        args.items,
                    )
                )
            ),
            "NextLong 式合成长文",
        )
        _write_jsonl(
            out / f"{ENTROPY_LONG_NAME}.jsonl",
            (
                "\n".join([f"《长文 {i + 1}》", doc])
                for i, doc in enumerate(
                    _take(
                        synth_entropylong(documents(), args.synth_target_chars, seed=args.seed),
                        args.items,
                    )
                )
            ),
            "EntropyLong 式合成长文",
        )

    if args.step in ("mrcr", "all"):
        rows = []
        evals = []
        for index in range(args.items):
            item = build_mrcr_item(
                documents(),
                target_chars=args.target_chars,
                needles=args.needles,
                seed=args.seed,
                index=index,
            )
            if item is None:
                break
            rows.append({"text": item["text"]})
            evals.append(
                {
                    "capability": item["capability"],
                    "prompt": item["prompt"],
                    "ground_truth": item["ground_truth"],
                }
            )
        _write_jsonl(out / f"{MRCR_NAME}.jsonl", rows, "MRCR 类长文（训练用整篇文本）")
        _write_jsonl(eval_out, evals, "MRCR 类评测集（stage3_eval 长文套件读它）")
    return 0


def _take(iterator: Iterator[str], n: int) -> Iterator[str]:
    for i, value in enumerate(iterator):
        if i >= n:
            return
        yield value


if __name__ == "__main__":
    raise SystemExit(main())
