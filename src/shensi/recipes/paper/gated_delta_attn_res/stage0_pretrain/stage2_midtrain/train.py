#!/usr/bin/env python3
"""GDAR 配方 · stage2_midtrain（Mid-1 能力强化 / Mid-2 长文档）。

用法与 shensi 配方的 stage train.py 同一套：

    python train.py --smoke                       # tiny 几何 + mock 数据 + 5 步
    python train.py --tokens 5e8 --load <PT-2 ckpt>   # Mid-1 能力强化
    python train.py --profile mid2 --tokens 3e8 --load <Mid-1 ckpt>  # Mid-2 长文档
    python train.py --model-algo base --dry-run   # 对照臂（plain Qwen3，无 --spec）

模型算法（--model-algo）见根 README 的 MODEL_ALGOS 注册表；所有臂跑同一份段位配方。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common

STAGE = "stage2_midtrain"


def main() -> int:
    ap = argparse.ArgumentParser(description="GDAR stage2_midtrain（Mid-1/Mid-2 两段）")
    ap.add_argument("--profile", default="default")
    ap.add_argument(
        "--model-algo",
        default=None,
        help=f"模型算法（不给则用 {common.DEFAULT_ALGO}；profile 自带 spec 时用 profile 的）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不启动")
    ap.add_argument("--smoke", action="store_true", help="跑仓库内 tiny 配置 5 步")
    ap.add_argument(
        "--tokens",
        type=lambda v: int(float(v)),  # 认 1e9 这类科学计数法
        default=None,
        help="token 预算（认 1e9），换算 train_iters",
    )
    ap.add_argument("--data-dir", default=None, help="预处理产物目录（含 blend.json）")
    ap.add_argument("--load", default=None, help="接续上一段的 ckpt 目录（接续上一段 ckpt）")
    ap.add_argument(
        "--set", dest="override", action="append", default=[], help="点号键覆写，可多次"
    )
    ap.add_argument(
        "--early-stop",
        type=int,
        default=None,
        help="早停耐心（验证指标连续多少次不改善就收尾）；不给用配置默认（默认就开）",
    )
    ap.add_argument(
        "--no-early-stop",
        action="store_true",
        help="关掉早停看门狗（默认开：训练步数给无限大，靠早停及时收尾）",
    )
    args = ap.parse_args()
    if args.smoke:
        return common.smoke(STAGE)
    algo = common.apply_algo_or_die(args.model_algo)
    paths = common.env_paths()
    data_dir = Path(args.data_dir or paths["data"] / STAGE)
    cfg = common.build_config(
        STAGE,
        args.profile,
        args.override,
        data_dir,
        tokens=args.tokens,
        model_algo=algo,
        load_ckpt=args.load,
    )
    watch = None if args.no_early_stop else common.early_stop_plan(STAGE, cfg, args.early_stop)
    return common.run(cfg, args.dry_run, watch=watch)


if __name__ == "__main__":
    raise SystemExit(main())
