#!/usr/bin/env python3
import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DS_SERVER = "https://datasets-server.huggingface.co"
RAW = "https://raw.githubusercontent.com"
# HF 上这三代的元数据配置（核实过：v1 没有 language 列，v2/v3 有）
V1 = ("nvidia/Nemotron-Pretraining-Code-v1", "Nemotron-Code-Metadata")
V2 = ("nvidia/Nemotron-Pretraining-Code-v2", "Nemotron-Code-Metadata")
V3 = ("nvidia/Nemotron-Pretraining-Code-v3", "Nemotron-Code-Metadata")
TAGS = {"v1": V1, "v2": V2, "v3": V3}
# 二进制 / 生成的扩展名，不回捞
SKIP_EXT = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".svg",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".whl",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".pyc",
    ".class",
    ".jar",
    ".parquet",
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".onnx",
    ".mp3",
    ".mp4",
    ".mov",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".lock",
    ".min.js",
    ".min.css",
    ".map",
)
TEXT_KEYS = ("text", "content")


def opener(proxy: str | None):
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
    return urllib.request.build_opener(*handlers)


def http_get(url: str, op, timeout: int = 45, token: str = "") -> bytes | None:
    hdr = {"User-Agent": "shensi-recipe/1.0"}
    if token:
        hdr["Authorization"] = f"Bearer {token}"
    try:
        with op.open(urllib.request.Request(url, headers=hdr), timeout=timeout) as resp:
            return resp.read()
    except Exception:  # noqa: BLE001
        return None


def hf_rows(
    dataset: str, config: str, n: int, op, token: str = "", split: str = "train"
) -> list[dict]:
    rows: list[dict] = []
    while len(rows) < n:
        length = min(100, n - len(rows))
        url = f"{DS_SERVER}/rows?" + urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": len(rows),
                "length": length,
            }
        )
        body = http_get(url, op, timeout=90, token=token)
        if body is None:
            raise SystemExit(
                f"[codev3] 取不到 {dataset} [{config}] 的元数据（检查代理/token/配置名）"
            )
        page = [r["row"] for r in json.loads(body).get("rows", [])]
        if not page:
            break
        rows += page
    return rows[:n]


def local_rows(path: Path, limit: int | None = None) -> list[dict]:
    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.glob("**/*") if p.suffix in (".parquet", ".jsonl", ".json"))
    )
    out: list[dict] = []
    for f in files:
        if f.suffix == ".parquet":
            import pyarrow.parquet as pq

            for batch in pq.ParquetFile(f).iter_batches(batch_size=4096):
                out += batch.to_pylist()
        else:
            with open(f, encoding="utf-8") as fh:
                out += [json.loads(line) for line in fh if line.strip()]
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def load_meta(
    key: str,
    local: str | None,
    op,
    token: str,
    sample: int | None,
    limit: int | None = None,
) -> list[dict]:
    if local:
        rows = local_rows(Path(local), limit or sample)
    elif sample:
        dataset, config = TAGS[key]
        rows = hf_rows(dataset, config, limit or sample, op, token)
    else:
        raise SystemExit(
            f"[codev3] 缺 {key} 的元数据：给 --{key}-meta <目录/文件> 或 --hf-sample N"
        )
    print(
        f"[codev3] {key}: {len(rows)} 行元数据" + (f"（本地 {local}）" if local else "（HF 采样）")
    )
    return rows


def _key(row: dict) -> tuple[str, str] | None:
    repo = str(row.get("repo") or "").strip()
    rel = str(row.get("rel_path") or "").strip().lstrip("/")
    return (repo, rel) if repo and rel else None


# (repo, rel_path) 为键；v2 有 language，冲突时后写的覆盖
def build_basis(*row_sets: list[dict]) -> dict[tuple[str, str], dict]:
    basis: dict[tuple[str, str], dict] = {}
    for rows in row_sets:
        for r in rows:
            k = _key(r)
            if k is None:
                continue
            basis[k] = {
                "commit_id": str(r.get("commit_id") or ""),
                "language": str(r.get("language") or ""),
            }
    return basis


# commit 只区分“同一个文件换了版本”，判定以 (repo, rel_path) 为准
def classify(v3_rows: list[dict], basis: dict[tuple[str, str], dict]) -> dict:
    stats = {"rows": 0, "carried_same_commit": 0, "carried_new_commit": 0, "new": 0, "bad_row": 0}
    by_lang: dict[str, int] = {}
    carried: list[dict] = []
    new: list[dict] = []
    for r in v3_rows:
        stats["rows"] += 1
        k = _key(r)
        if k is None:
            stats["bad_row"] += 1
            continue
        base = basis.get(k)
        row = {**r, "_key": k}
        if base is None:
            stats["new"] += 1
            new.append(row)
        elif base["commit_id"] and base["commit_id"] == str(r.get("commit_id") or ""):
            stats["carried_same_commit"] += 1
            carried.append(row)
        else:
            stats["carried_new_commit"] += 1
            carried.append(row)
        lang = str(
            r.get("language") or base.get("language", "") if base else r.get("language") or ""
        )
        if lang:
            by_lang[lang] = by_lang.get(lang, 0) + 1
    stats["by_language_top"] = sorted(by_lang.items(), key=lambda kv: -kv[1])[:12]
    # 反向：v1/v2 里有多少键出现在 v3。v3 行数过亿时建键集很吃内存，所以只在 200 万行内算
    if len(v3_rows) <= 2_000_000:
        v3_keys = {k for k in (_key(r) for r in v3_rows) if k}
        hit = sum(1 for k in basis if k in v3_keys)
        stats["basis_in_v3"] = hit
        stats["basis_coverage"] = round(hit / max(len(basis), 1), 4)
    else:
        stats["basis_coverage"] = None
    return {"stats": stats, "carried": carried, "new": new}


# 已有文本先复用：重跑或别的来源抓过的都不再联网
def text_cache_index(dirs: list[str], max_records: int | None = None) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    n = 0
    for d in dirs:
        p = Path(d)
        files = (
            [p]
            if p.is_file()
            else sorted(f for f in p.glob("**/*") if f.suffix in (".parquet", ".jsonl", ".json"))
        )
        for f in files:
            if f.suffix == ".parquet":
                import pyarrow.parquet as pq

                for batch in pq.ParquetFile(f).iter_batches(batch_size=2048):
                    for row in batch.to_pylist():
                        _cache_add(index, row)
                        n += 1
            else:
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            try:
                                _cache_add(index, json.loads(line))
                            except json.JSONDecodeError:
                                continue
                            n += 1
            if max_records and n >= max_records:
                break
    if index:
        print(f"[codev3] 本地文本缓存：{len(index)} 个 (repo, rel_path) 键（扫了 {n} 条记录）")
    return index


def _cache_add(index: dict, row: dict) -> None:
    k = _key(row)
    if k is None or k in index:
        return
    for tk in TEXT_KEYS:
        v = row.get(tk)
        if isinstance(v, str) and v:
            index[k] = v
            return


# 路径必须 percent-encode——元数据里有空格/非 ASCII 路径，不转义会 InvalidURL
def raw_url(row: dict) -> str:
    repo = urllib.parse.quote(str(row.get("repo") or ""), safe="/")
    commit = urllib.parse.quote(str(row.get("commit_id") or ""), safe="")
    rel = urllib.parse.quote(str(row.get("rel_path") or "").lstrip("/"), safe="/")
    return f"{RAW}/{repo}/{commit}/{rel}"


def materialize(
    rows: list[dict],
    out_jsonl: Path,
    op,
    cache: dict[tuple[str, str], str] | None = None,
    workers: int = 8,
    min_chars: int = 200,
    max_bytes: int = 2_000_000,
    ledger_path: Path | None = None,
    header: bool = True,
) -> dict:
    cache = cache or {}
    stats = {
        "rows": 0,
        "cache": 0,
        "ok": 0,
        "missing": 0,
        "skip_ext": 0,
        "too_big": 0,
        "too_short": 0,
    }
    failed: list[str] = []

    def one(row: dict):
        stats["rows"] += 1
        rel = str(row.get("rel_path") or "")
        k = row.get("_key") or _key(row)
        if any(rel.lower().endswith(e) for e in SKIP_EXT):
            return "skip_ext", None
        if k in cache:
            text = cache[k]
            return ("cache" if len(text) >= min_chars else "too_short"), text
        body = http_get(raw_url(row), op)
        if body is None:
            return "missing", None
        if len(body) > max_bytes:
            return "too_big", None
        text = body.decode("utf-8", errors="replace")
        if len(text) < min_chars:
            return "too_short", None
        return "ok", text

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        open(out_jsonl, "w", encoding="utf-8") as fh,
    ):
        for row, (status, text) in zip(rows, pool.map(one, rows), strict=False):
            stats[status] += 1
            if text is None:
                if status == "missing" and len(failed) < 20:
                    failed.append(f"{row.get('repo')}/{row.get('rel_path')}")
                continue
            head = ""
            if header and row.get("repo"):
                rel = str(row.get("rel_path") or "").lstrip("/")
                head = f"# {row.get('repo')}/{rel} @ {row.get('commit_id')}\n"
            fh.write(json.dumps({"text": head + text}, ensure_ascii=False) + "\n")
    ledger = {
        "jsonl": str(out_jsonl),
        "stats": stats,
        "failed_samples": failed,
    }
    if ledger_path:
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8")
    return ledger


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--codev3", action="store_true", help="把 Code-v3 元数据在 v1/v2 基础上落地成可训练文本"
    )
    ap.add_argument(
        "--v1-meta", default=None, help="v1 元数据目录/文件（缺省配合 --hf-sample 用 HF 采样）"
    )
    ap.add_argument("--v2-meta", default=None, help="v2 元数据目录/文件")
    ap.add_argument("--v3-meta", default=None, help="v3 元数据目录/文件")
    ap.add_argument(
        "--text-cache", default="", help="逗号分隔的本地文本目录（含 repo/rel_path），命中就不回捞"
    )
    ap.add_argument(
        "--only-new", action="store_true", help="只回捞 v3 相对 v1/v2 的增量（--only-carried 取反）"
    )
    ap.add_argument("--only-carried", action="store_true", help="只回捞 v1/v2 里已有的那部分")
    ap.add_argument("--proxy", default=os.environ.get("PROXY", "http://127.0.0.1:7897"))
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--max-bytes", type=int, default=2_000_000)


def run(args) -> dict:
    paths_out = Path(args.out)
    paths_out.mkdir(parents=True, exist_ok=True)
    op = opener(args.proxy or None)
    sample = getattr(args, "hf_sample", None) or None

    v1 = load_meta("v1", args.v1_meta, op, args.token, sample)
    v2 = load_meta("v2", args.v2_meta, op, args.token, sample)
    v3 = load_meta("v3", args.v3_meta, op, args.token, sample)
    basis = build_basis(v1, v2)
    print(f"[codev3] v1/v2 基础清单：{len(basis)} 个 (repo, rel_path) 键")

    cls = classify(v3, basis)
    s = cls["stats"]
    print(
        f"[codev3] v3 分类：{s['rows']} 行 → 与 v1/v2 重叠 {s['carried_same_commit'] + s['carried_new_commit']}"
        f"（其中 commit 相同 {s['carried_same_commit']}、commit 变化 {s['carried_new_commit']}）"
        f" + v3 增量 {s['new']}；坏行 {s['bad_row']}"
    )
    if s["by_language_top"]:
        print(f"[codev3] 语言分布 top：{s['by_language_top'][:6]}")

    rows = (
        cls["new"]
        if args.only_new
        else (cls["carried"] if args.only_carried else cls["carried"] + cls["new"])
    )
    if not rows:
        print("[codev3] 选中的行为空：去掉 --only-new/--only-carried 再试")
    tag = f"{V3[0].split('/')[-1]}__{V3[1]}"
    jsonl = paths_out / f"{tag}.jsonl"
    cache = (
        text_cache_index([d for d in (args.text_cache or "").split(",") if d.strip()])
        if args.text_cache
        else {}
    )
    ledger = materialize(
        rows,
        jsonl,
        op,
        cache=cache,
        workers=args.workers,
        min_chars=args.min_chars,
        max_bytes=args.max_bytes,
        ledger_path=paths_out / f"{tag}.codev3.json",
    )
    ledger["basis"] = {
        "keys": len(basis),
        "v1_rows": len(v1),
        "v2_rows": len(v2),
        "v3_rows": len(v3),
    }
    ledger["classify"] = s
    ledger["selected"] = "new" if args.only_new else "carried" if args.only_carried else "all"
    (paths_out / "codev3_index.json").write_text(
        json.dumps(ledger, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    m = ledger["stats"]
    print(
        f"[codev3] 回捞：写 {m['ok'] + m['cache']} 篇（缓存命中 {m['cache']}）→ {jsonl}\n"
        f"         账本：404 {m['missing']} / 跳过扩展名 {m['skip_ext']} / 超体积 {m['too_big']}"
        f" / 太短 {m['too_short']}（明细 {paths_out / 'codev3_index.json'}）"
    )
    return ledger


def selftest() -> int:
    import tempfile

    ok = True

    def chk(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [判定] {name}: {'PASS' if cond else 'FAIL'} {detail}")

    a = {"repo": "r/A", "rel_path": "src/a.py", "commit_id": "aaa1111", "language": "Python"}
    b = {"repo": "r/B", "rel_path": "lib/b.go", "commit_id": "bbb2222", "language": "Go"}
    c = {"repo": "r/C", "rel_path": "x/c.rs", "commit_id": "ccc3333", "language": "Rust"}
    basis = build_basis([{**a, "commit_id": "old0000"}], [b])
    cls = classify([a, b, c], basis)
    s = cls["stats"]
    chk("v1 里有、commit 变了 → carried_new_commit", s["carried_new_commit"] == 1, f"stats={s}")
    chk("v2 里有、commit 相同 → carried_same_commit", s["carried_same_commit"] == 1)
    chk("v1/v2 都没有 → v3 增量", s["new"] == 1)
    chk(
        "缺 repo/rel_path 的行被挡掉",
        classify([{"repo": "", "rel_path": "x"}], basis)["stats"]["bad_row"] == 1,
    )
    chk("--only-new 只留增量", len(classify([a, b, c], basis)["new"]) == 1)

    u = raw_url({"repo": "r/A B", "commit_id": "aaa1111", "rel_path": "目录/x y.css"})
    chk(
        "URL 转义空格与非 ASCII（否则 GitHub 直接 InvalidURL）",
        " " not in u and "%20" in u and "%E7%9B%AE" in u,
        u,
    )

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cache_file = d / "cache.jsonl"
        cache_file.write_text(
            json.dumps(
                {"repo": "r/A", "rel_path": "src/a.py", "text": "x" * 300}, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )
        cache = text_cache_index([str(cache_file)])
        chk(
            "本地文本缓存按 (repo, rel_path) 建索引",
            ("r/A", "src/a.py") in cache,
            f"键 {list(cache)}",
        )
        rows = [{**r, "_key": _key(r)} for r in (a, b, c)]
        for r in rows[1:]:
            cache[r["_key"]] = "y" * 300
        led = materialize(
            rows, d / "out.jsonl", opener(None), cache=cache, ledger_path=d / "led.json"
        )
        chk(
            "命中缓存就不再联网回捞",
            led["stats"]["cache"] == 3 and led["stats"]["missing"] == 0,
            str(led["stats"]),
        )
        lines = [x for x in (d / "out.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        chk("jsonl 条数 = 命中数", len(lines) == 3)
        head = json.loads(lines[0])["text"].splitlines()[0]
        chk("正文首行是出处（repo/rel_path @ commit）", head == "# r/A/src/a.py @ aaa1111", head)
        chk(
            "跳过扩展名不计入文本",
            materialize([{**rows[0], "rel_path": "a.png"}], d / "o2.jsonl", opener(None))["stats"][
                "skip_ext"
            ]
            == 1,
        )

    print("\n  -> " + ("全部 PASS" if ok else "有 FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Nemotron-Pretraining-Code-v3 落地（在 v1/v2 基础上）")
    add_args(ap)
    ap.add_argument(
        "--hf-sample", type=int, default=None, help="v1/v2/v3 各取这么多行元数据（调试）"
    )
    ap.add_argument("--out", default=None, help="产物目录（jsonl + 账本）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--selftest", action="store_true", help="跑离线自检（不联网）")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.codev3:
        ap.error("加 --codev3（这个入口就是干这件事的），或 --selftest")
    if not args.out:
        ap.error("--codev3 需要 --out <产物目录>")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
