"""官方 MRCR：取数与判分（前缀哈希 + SequenceMatcher）。"""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path

# 官方分桶边界（token 数，含左右端点）
BINS: tuple[tuple[int, int], ...] = (
    (4096, 8192),
    (8192, 16384),
    (16384, 32768),
    (32768, 65536),
    (65536, 131072),
    (131072, 262144),
    (262144, 524288),
    (524288, 1048576),
)

NEEDLE_FILES = {
    2: ("2needle/2needle_0.parquet", "2needle/2needle_1.parquet"),
    4: ("4needle/4needle_0.parquet", "4needle/4needle_1.parquet"),
    8: ("8needle/8needle_0.parquet", "8needle/8needle_1.parquet"),
}


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """官方判分：前缀哈希必须在开头，随后按 SequenceMatcher 比值给分。"""
    if not response.startswith(random_string_to_prepend):
        return 0.0
    return float(
        SequenceMatcher(
            None,
            response.removeprefix(random_string_to_prepend),
            answer.removeprefix(random_string_to_prepend),
        ).ratio()
    )


def _n_tokens(messages: list[dict], encoder=None) -> int:
    """消息的 token 数：有 tiktoken 就用官方 o200k_base，没有就按 4 字符/token 估。"""
    text = "\n".join(str(m.get("content", "")) for m in messages)
    if encoder is not None:
        return len(encoder.encode(text))
    return len(text) // 4


def _token_encoder(tokenizer: str | None):
    try:
        import tiktoken

        return tiktoken.get_encoding(tokenizer or "o200k_base")
    except Exception:  # noqa: BLE001 - 没装 tiktoken 就用估算
        return None


def load_samples(
    *,
    needles: tuple[int, ...] = (2,),
    bins: tuple[tuple[int, int], ...] | None = None,
    per_bin: int = 1,
    limit: int | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    tokenizer: str | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict]:
    """按针数与 token 桶取样本；返回 [{messages, answer, prefix, n_needles, n_tokens, bin}]。"""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    encoder = _token_encoder(tokenizer)
    wanted_bins = bins if bins is not None else BINS
    out: list[dict] = []
    for n in needles:
        for rel in NEEDLE_FILES.get(n, ()):
            path = hf_hub_download("openai/mrcr", rel, repo_type="dataset", cache_dir=cache_dir)
            table = pq.read_table(path)
            seen = {b: 0 for b in wanted_bins}
            for row in table.to_pylist():
                messages = json.loads(row["prompt"])
                ntok = _n_tokens(messages, encoder)
                if min_tokens is not None and ntok < min_tokens:
                    continue
                if max_tokens is not None and ntok > max_tokens:
                    continue
                bucket = next((b for b in wanted_bins if b[0] <= ntok <= b[1]), None)
                if bucket is None:
                    continue
                if seen[bucket] >= per_bin:
                    continue
                seen[bucket] += 1
                out.append(
                    {
                        "messages": messages,
                        "answer": row["answer"],
                        "prefix": row["random_string_to_prepend"],
                        "n_needles": int(row["n_needles"]),
                        "n_tokens": ntok,
                        "bin": f"{bucket[0]}-{bucket[1]}",
                        "date_added": row.get("date_added"),
                    }
                )
                if limit is not None and len(out) >= limit:
                    return out
                if all(seen[b] >= per_bin for b in wanted_bins):
                    break
    return out


def selftest() -> int:
    """离线自检：判分器行为（满分 / 前缀缺失 0 分 / 打乱显著掉分）+ 取数（最小桶各 1 条）。"""
    answer = "AB12cd" + "the quick brown fox jumps over the lazy dog"
    prefix = "AB12cd"
    checks = [
        ("原样回答满分", grade(answer, answer, prefix) == 1.0),
        ("缺前缀 0 分", grade(answer.removeprefix(prefix), answer, prefix) == 0.0),
        (
            "打乱后掉分",
            grade(prefix + "dog lazy the over jumps fox brown quick the", answer, prefix) < 0.8,
        ),
        ("空回答 0 分", grade("", answer, prefix) == 0.0),
    ]
    ok = True
    for name, passed in checks:
        print(f"[mrcr] 判分自检：{name} → {'✓' if passed else '✗'}")
        ok &= passed

    rows = load_samples(needles=(2,), per_bin=1, limit=2, min_tokens=4096, max_tokens=8192)
    if rows:
        row = rows[0]
        print(
            f"[mrcr] 取数自检：{len(rows)} 条（针={row['n_needles']} "
            f"tokens≈{row['n_tokens']} 桶={row['bin']} 消息数={len(row['messages'])} "
            f"前缀={row['prefix']!r}）"
        )
        ok &= row["n_needles"] == 2 and len(row["messages"]) > 10
    else:
        print("[mrcr] 取数自检：没取到样本（网络？）→ 跳过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest())
