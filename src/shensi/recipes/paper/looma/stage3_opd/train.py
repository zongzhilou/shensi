#!/usr/bin/env python3
"""OPD 的训练入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from shensi.recipes.paper.looma import common

STAGE = "stage3_opd"


def main() -> int:
    ap = argparse.ArgumentParser(description="Looma stage3_opd（on-policy 蒸馏）")
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
        type=lambda v: int(float(v)),
        default=None,
        help="token 预算（认 1e9），换算 train_iters",
    )
    ap.add_argument("--data-dir", default=None, help="预处理产物目录（含 blend.json）")
    ap.add_argument("--load", default=None, help="学生起点 ckpt（默认 SFT-2）")
    ap.add_argument(
        "--teacher-cache",
        default=None,
        help="teacher 的 token 级 logprob 缓存目录（→ --logits-load-dir，mcore 原生 KD）",
    )
    ap.add_argument(
        "--set", dest="override", action="append", default=[], help="点号键覆写，可多次"
    )
    ap.add_argument("--early-stop", type=int, default=None, help="早停耐心；不给用配置默认")
    ap.add_argument("--no-early-stop", action="store_true", help="关掉早停看门狗（默认开）")
    args = ap.parse_args()
    if args.smoke:
        return common.smoke("stage1_pretrain", override=args.override)
    algo = common.apply_algo_or_die(args.model_algo)
    if args.teacher_cache and not Path(args.teacher_cache).is_dir():
        raise SystemExit(
            f"[looma] teacher 缓存目录不存在：{args.teacher_cache}（先跑 README 的 ①② 两步）"
        )
    override = []
    if args.teacher_cache:
        override.append(f"train.model.logits_load_dir={args.teacher_cache}")
    override += args.override
    paths = common.env_paths()
    data_dir = Path(args.data_dir or paths["data"] / STAGE)
    cfg = common.build_config(
        STAGE,
        args.profile,
        override,
        data_dir,
        tokens=args.tokens,
        model_algo=algo,
        load_ckpt=args.load,
    )
    watch = None if args.no_early_stop else common.early_stop_plan(STAGE, cfg, args.early_stop)
    return common.run(cfg, args.dry_run, watch=watch)


if __name__ == "__main__":
    raise SystemExit(main())
