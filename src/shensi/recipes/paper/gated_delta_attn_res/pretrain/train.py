#!/usr/bin/env python3
"""GDAR 配方 · 预训练 stage 入口。

用法与 shensi 配方的 stage train.py 同一套：

    python train.py --smoke                 # tiny 几何 + mock 数据 + 5 步（不碰语料）
    python train.py --profile debug         # tiny 几何 + 真实语料的极小档
    python train.py --profile gdar --dry-run
    python train.py --profile gdar --tokens 10e9
    python train.py --profile gdar --seed 1 # 多 seed 的另外两臂

profile 表（pretrain/config/）：

| profile | 是什么 |
|---|---|
| default | base 对照臂（plain Qwen3-0.6B，无 --spec） |
| gdar    | 论文主行（gdar_layer_spec_paper） |
| ar/dar  | 恒等锚定的 AR / DAR 对照臂 |
| ablations/* | 设计消融 A1–A16 与 E3/E6 的对照行（每个只改一处） |
| debug   | tiny 几何（真实语料） |
| tiny    | mock 冒烟档（--smoke 用的就是它） |
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common


def main() -> int:
    ap = argparse.ArgumentParser(description="GDAR 配方预训练（base / GDAR / 消融臂）")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不启动")
    ap.add_argument("--smoke", action="store_true", help="跑仓库内 tiny 冒烟档 5 步")
    ap.add_argument("--tokens", type=int, default=None, help="token 预算，用来换算 train_iters")
    ap.add_argument("--data-dir", default=None, help="预处理产物目录（含 blend.json）")
    ap.add_argument(
        "--set", dest="override", action="append", default=[], help="点号键覆写，可多次"
    )
    ap.add_argument(
        "--early-stop",
        type=int,
        default=None,
        help="早停耐心（验证指标连续多少次不改善就收尾）；不给就不看门狗",
    )
    args = ap.parse_args()
    if args.smoke:
        return common.smoke()
    paths = common.env_paths()
    data_dir = Path(paths["data"])
    cfg = common.build_config(args.profile, args.override, data_dir, tokens=args.tokens)
    rc = common.run(cfg, args.dry_run)
    if args.early_stop and not args.dry_run:
        rc = common.watch(cfg, args.early_stop)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
