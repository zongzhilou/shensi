#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shensi import runtime  # noqa: F401
from shensi.recipes.shensi import codev3


def main() -> int:
    ap = argparse.ArgumentParser(description="按 Nemotron Code 元数据从 GitHub 回捞代码")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--metadata", type=str, help="本地元数据目录/parquet 文件")
    src.add_argument("--hf-sample", type=int, help="直接从 HF 取这么多行元数据（调试用）")
    ap.add_argument("--dataset", default="nvidia/Nemotron-Pretraining-Code-v3")
    ap.add_argument("--config", default="Nemotron-Code-Metadata")
    ap.add_argument(
        "--out", type=str, default=None, help="输出归一化 jsonl（默认 <dataset>_code.jsonl）"
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-bytes", type=int, default=2_000_000)
    ap.add_argument("--languages", default="", help="逗号分隔的语言白名单（空=全部）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--proxy", default=os.environ.get("PROXY", "http://127.0.0.1:7897"))
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = ap.parse_args()

    op = codev3.opener(args.proxy or None)
    if args.hf_sample:
        rows = codev3.hf_rows(args.dataset, args.config, args.hf_sample, op, args.token)
    else:
        rows = codev3.local_rows(Path(args.metadata), args.limit)
    langs = {x.strip().lower() for x in args.languages.split(",") if x.strip()}
    if langs:
        rows = [r for r in rows if str(r.get("language", "")).lower() in langs]
    if args.limit:
        rows = rows[: args.limit]
    rows = [r for r in rows if codev3._key(r)]
    print(f"[fetch] 元数据 {len(rows)} 行待抓（{args.dataset} [{args.config}]）")

    if args.dry_run:
        for r in rows[:10]:
            print("   ", codev3.raw_url(r))
        return 0

    out_path = Path(args.out or f"{Path(args.dataset).name}_code.jsonl")
    ledger = codev3.materialize(
        rows,
        out_path,
        op,
        workers=args.workers,
        min_chars=args.min_chars,
        max_bytes=args.max_bytes,
        ledger_path=out_path.with_suffix(".ledger.json"),
    )
    print(f"[fetch] 写出 {out_path}：{ledger['stats']}")
    if ledger["stats"]["ok"] + ledger["stats"]["cache"] == 0:
        print("[fetch] 一篇都没抓到：检查网络/代理，或这批元数据指向的仓库已下线")
        return 1
    if ledger["failed_samples"]:
        print(
            f"[fetch] 抓不到的样例（前 {len(ledger['failed_samples'])} 个）：{ledger['failed_samples'][:5]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
