#!/usr/bin/env python3
"""GDAR 配方 · stage3_opd 第②步：teacher 给学生 rollout 打分（mcore 原生 logits saver）。

走 mcore 自家的 `--logits-save-dir`（`megatron/training/distillation/logits_saver.py`）：
teacher 在 rollout 数据上做一次**冻结**前向（`--freeze-all-layers`，无反向），把每 token 的
top-K log-prob 落盘成 zstd 分片；student 训练时用 `--logits-load-dir` 读回做 KD。
top-K/top-P 截断即 MiniCPM5 的 "top-k logits 并集" 口径。

    # ① 学生 rollout（任意采样器；落成与 bin/idx 同源的 jsonl → data_prep）
    # ② teacher 打分（本脚本；每个方向一个 teacher、一个缓存目录）
    python score.py --load <teacher ckpt> --data-dir <rollout 数据目录> \
        --out <teacher logprob 缓存目录> --top-k 64
    # ③ 学生 KD 训练（stage3_opd/train.py --teacher-cache <缓存目录>）

说明：`--logits-save-dir` 要求 `--async-save` + `--use-persistent-ckpt-worker`（logits 走
checkpoint 队列异步落盘），本脚本已按该约定补齐这两个开关。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common

STAGE = "stage3_opd"


def main() -> int:
    ap = argparse.ArgumentParser(description="OPD teacher 打分（mcore 原生 logits saver）")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--model-algo", default=None, help="teacher 的模型算法（默认同训练）")
    ap.add_argument("--load", required=True, help="teacher 的 ckpt 目录")
    ap.add_argument("--data-dir", default=None, help="rollout 数据目录（含 blend.json）")
    ap.add_argument(
        "--out", required=True, help="logprob 缓存输出目录（学生训练时 --teacher-cache）"
    )
    ap.add_argument(
        "--top-k", type=int, default=64, help="每 token 保存的 top-K（MiniCPM5 的并集口径）"
    )
    ap.add_argument("--top-p", type=float, default=None, help="可选：top-P 截断（在 top-K 之后）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    algo = common.apply_algo_or_die(args.model_algo)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    override = [
        f"train.model.logits_save_dir={out}",
        f"train.model.logits_save_top_k={args.top_k}",
        "train.model.freeze_all_layers=true",
        "train.system.async_save=true",
        "train.system.use_persistent_ckpt_worker=true",
        # 打分是纯前向数据集遍历：不做评估、不留 ckpt
        "train.model.eval_iters=0",
        "train.model.eval_interval=1000000",
        "--no-save",
    ]
    if args.top_p is not None:
        override[2:2] = [f"train.model.logits_save_top_p={args.top_p}"]
    paths = common.env_paths()
    cfg = common.build_config(
        STAGE,
        args.profile,
        override,
        Path(args.data_dir or paths["data"] / STAGE),
        model_algo=algo,
        load_ckpt=args.load,
    )
    print(f"[gdar] teacher 打分：{args.load} → {out}（top-k={args.top_k}）")
    return common.run(cfg, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
