#!/usr/bin/env python3
"""预算化 data_prep 的闸门：按权重分配 token 预算、质量门、去重、记账与 blend 产物。

用合成语料（含刻意植入的重复与低质文档）跑一遍 `common.prepare(--tokens/--quality/--dedup)`，
校验：每源目标与实到、丢弃原因、budget_report.json / blend.json、以及 train 侧能直接读到它。
tokenizer 用配方里自带的 Qwen3-0.6B（几何无关，只做分词与 bin/idx 编码）。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from shensi.recipes.shensi.common import common  # noqa: E402

TOKENIZER = REPO / "src/shensi/recipes/paper/gated_delta_attn_res/common/tokenizer/Qwen3-0.6B"

FAILS: list[str] = []


def check(name: str, ok, extra: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {extra}", flush=True)
    if not ok:
        FAILS.append(name)


import random  # noqa: E402 - 与上面的 sys.path 注射同批（测试脚本）

VOCAB = [
    "sparse",
    "attention",
    "recipe",
    "keeps",
    "dense",
    "trunk",
    "stage",
    "model",
    "tokens",
    "mixture",
    "expert",
    "routing",
    "latent",
    "projection",
    "window",
    "block",
    "residual",
    "stream",
    "training",
    "budget",
    "data",
    "quality",
    "filter",
    "dedup",
    "shard",
    "document",
    "corpus",
    "language",
    "math",
    "code",
    "science",
    "web",
    "multilingual",
    "frontier",
]


def doc(rng: random.Random, n_words: int = 420) -> str:
    """确定性互异文档（同一词表的不同序列）——用于把预算填满而不触发近邻去重。"""
    return " ".join(rng.choice(VOCAB) for _ in range(n_words))


LOW_SHORT = "too short."
LOW_SYMBOLS = "|" * 1800
LOW_REPEAT = "same line\n" * 200


def make_sources(root: Path) -> dict:
    spec = {
        "datasets": [
            {"name": "srcA", "config": "v1", "weight": 0.5},
            {"name": "srcB", "config": "v1", "weight": 0.25},
            {"name": "srcC", "config": "v1", "weight": 0.25},
        ]
    }
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261010)
    base = doc(rng)
    near = (
        " ".join(base.split()[:200]) + " " + " ".join(base.split()[200:])
    )  # 同序（精确重复用 base）
    near = near.replace("sparse", "sparse-")  # 近邻重复：改一个 token
    docs_a = [base, near, base, LOW_SHORT, LOW_SYMBOLS] + [doc(rng) for _ in range(80)]
    docs_b = [LOW_REPEAT] + [doc(rng) for _ in range(40)]
    docs_c = [doc(rng) for _ in range(40)]
    for name, docs in (("srcA", docs_a), ("srcB", docs_b), ("srcC", docs_c)):
        d = root / name / "v1"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "shard.jsonl", "w", encoding="utf-8") as fh:
            for text in docs:
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return spec


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="shensi_data_prep_"))
    root, out = tmp / "raw", tmp / "bins"
    spec = make_sources(root)
    budget = 6000

    blend = common.prepare(
        spec,
        root,
        out,
        str(TOKENIZER),
        limit=None,
        workers=1,
        tokens=budget,
        quality="heuristic",
        dedup="near",
    )
    report = json.loads((out / "budget_report.json").read_text(encoding="utf-8"))
    sources = report["sources"]

    check(
        "每源按权重分到预算，且实到落在目标的一篇粒度内",
        all(
            s["target_tokens"] <= s["tokens"] <= s["target_tokens"] + 600  # 停止点精确到一篇
            for s in sources.values()
        )
        and {k.split("__")[0] for k in sources} == {"srcA", "srcB", "srcC"},
        json.dumps({k: (s["target_tokens"], s["tokens"]) for k, s in sources.items()}),
    )
    dropped = {k: v["dropped"] for k, v in sources.items()}
    check(
        "质量门挡掉短文档 / 符号噪声 / 重复行",
        any(v.get("too_short") for v in dropped.values())
        and any(v.get("letters_ratio") for v in dropped.values())
        and any(v.get("repeated_lines") for v in dropped.values()),
        json.dumps(dropped),
    )
    check(
        "去重挡掉精确与近邻重复",
        sum(v.get("exact_dup", 0) for v in dropped.values()) >= 1
        and sum(v.get("near_dup", 0) for v in dropped.values()) >= 1,
        json.dumps(dropped),
    )
    check(
        "总账 = 各源实到之和，且不超预算 + 一篇粒度",
        report["total_tokens"] == sum(s["tokens"] for s in sources.values())
        and report["total_tokens"] <= budget + 1800,  # 三源各一篇的粒度
        f"total={report['total_tokens']} / budget={budget}",
    )
    flat = json.loads((out / "blend.json").read_text(encoding="utf-8"))
    weights, prefixes = flat[0::2], flat[1::2]
    check(
        "blend.json 是扁平 [权重, bin 前缀, …] 且 bin/idx 落盘",
        len(weights) == 3
        and all(Path(p + ".bin").is_file() and Path(p + ".idx").is_file() for p in prefixes),
        f"weights={weights}",
    )

    # train 侧：build_config 会把 <out>/blend.json 注入 train.data.data_path（tiny 是 mock 档，按设计不注入）
    import os

    os.environ.setdefault("SHENSI_ROOT", str(REPO))
    os.environ.setdefault("SHENSI_FS", str(tmp))
    cfg = common.build_config("stage0_pretrain/stage1_pretrain", "default", [], out)
    produced = json.loads(Path(blend).read_text(encoding="utf-8"))
    check(
        "train 侧直接吃到处理后的数据（data_path = 权重×前缀列表）",
        cfg["train"]["data"].get("data_path") == produced and len(produced) == 6,
        str(cfg["train"]["data"].get("data_path"))[:60] + " …",
    )

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'SOME CHECKS FAILED'}（产物：{out}）")
    if FAILS:
        print("FAILED:", FAILS)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
