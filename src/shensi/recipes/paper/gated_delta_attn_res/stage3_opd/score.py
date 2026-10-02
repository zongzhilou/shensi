"""OPD 的 teacher 打分：为 rollout 产出 token 级对数概率缓存。"""

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
