"""OPD 训练入口：公共核 + teacher 缓存目录（`--teacher-cache`）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from shensi.recipes.paper.gated_delta_attn_res import common


def train_main(stage: str, here: Path, argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{stage}（on-policy 蒸馏：mcore 原生 KD）")
    common.add_common_train_args(ap)
    ap.add_argument(
        "--teacher-cache",
        default=None,
        help="teacher 的 token 级 logprob 缓存目录（→ --logits-load-dir）",
    )
    ap.add_argument(
        "--reverse-kl", action="store_true", help="KD 用 reverse KL（KL(student‖teacher)）"
    )
    args = ap.parse_args(argv)
    overrides: list[str] = []
    if args.teacher_cache:
        overrides.append(f"train.model.logits_load_dir={args.teacher_cache}")
    if args.reverse_kl:
        overrides.append("train.model.logits_load_reverse_kl=true")
    return common.train_from_args(stage, args, overrides=overrides)
