#!/usr/bin/env python3
"""评测数据落位：论文基准的数据集清单与离线放置指引（**不代下**）。

lmms-eval 自己会下数据集（默认 HF 缓存）；本脚本只做两件事：
1. ``--discover``：打印论文各套件 → 基准 → 数据集的登记表（离线准备/审计用）；
2. ``--check``：按 ``$SHENSI_FS/datasets/lmms-eval`` 与 HF 缓存核对是否到位，缺的显式列出。

python data_prep.py --discover
python data_prep.py --check --suite main
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from shensi.recipes.paper.deeprecur.stage2_eval.common.benchmarks import SUITES, suite_benchmarks

#: 论文基准 → 数据集（lmms-eval 的 task 会自下；这里登记 HF id 供离线预置与审计）
DATASETS: dict[str, str] = {
    "VQAv2": "lmms-lab/VQAv2",
    "GQA": "lmms-lab/GQA",
    "TextVQA": "lmms-lab/TextVQA",
    "DocVQA": "lmms-lab/DocVQA",
    "InfoVQA": "lmms-lab/InfoVQA",
    "SEED": "lmms-lab/SEED-Bench",
    "POPE": "lmms-lab/POPE",
    "MMMU": "lmms-lab/MMMU",
    "MM-Vet": "lmms-lab/MMVet",
    "ChartQA": "lmms-lab/ChartQA",
    "MultiDocVQA": "lmms-lab/MultiDocVQA",
    "EgoSchema": "lmms-lab/EgoSchema",
    "NextQA": "lmms-lab/NExTQA",
    "MSVD": "lmms-lab/MSVD",
    "ActivityNet": "lmms-lab/ActivityNet-QA",
}


def discover(suite: str | None) -> int:
    suites = [suite] if suite else sorted(SUITES)
    print("[deeprecur·eval] 论文基准 → 数据集（lmms-eval 自下；离线时预置到 HF 缓存或 LM_HOME）:")
    for name in suites:
        for bench in suite_benchmarks(name):
            print(f"  {name:<9} {bench.paper_name:<12} {DATASETS.get(bench.paper_name, '（待补）')}")
    print("  离线：先在有网机器缓存，再把 HF 缓存目录整体同步到训练机（HF_HOME 指向它）")
    return 0


def check(suite: str) -> dict:
    """核对数据集是否在 HF 缓存/本地目录里（缺的显式列出，不静默）。"""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = hf_home / "hub"
    local = Path(os.environ.get("SHENSI_FS", "/home/louzo/fsdata")) / "datasets" / "lmms-eval"
    report: dict = {"present": [], "missing": []}
    for bench in suite_benchmarks(suite):
        dataset = DATASETS.get(bench.paper_name)
        if not dataset:
            report["missing"].append(f"{bench.paper_name}（未登记 HF id）")
            continue
        slug = "datasets--" + dataset.replace("/", "--")
        if (hub / slug).is_dir() or (local / slug).is_dir() or (local / bench.paper_name).is_dir():
            report["present"].append(f"{bench.paper_name}（{dataset}）")
        else:
            report["missing"].append(f"{bench.paper_name}（{dataset}）")
    print(f"[deeprecur·eval] suite={suite} 数据核对：在 {len(report['present'])} / 缺 {len(report['missing'])}")
    for name in report["present"]:
        print(f"  ✓ {name}")
    for name in report["missing"]:
        print(f"  ✗ {name}（有网机器缓存后同步，或让 lmms-eval 首次运行时自下）")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="评测数据集落位/审计（不代下）")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--suite", default="main", choices=sorted(SUITES))
    args = parser.parse_args()
    if args.discover:
        return discover(args.suite if args.suite != "main" else None)
    if args.check:
        check(args.suite)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
